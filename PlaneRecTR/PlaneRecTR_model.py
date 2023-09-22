# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Modified by https://github.com/facebookresearch/Mask2Former
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher

from .utils.misc import get_coordinate_map

@META_ARCH_REGISTRY.register()
class PlaneRecTR(nn.Module):
    """
    Main class for plane segmentation and reconstruction architectures.
    """

    @configurable
    def __init__(        
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        k_inv_dot_xy1,
        predict_param: bool,
        predict_depth: bool,
        plane_mask_threshold: float,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            k_inv_dot_xy1:
            predict_param: bool,
            predict_depth: bool,
            plane_mask_threshold: float, 
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.k_inv_dot_xy1 = k_inv_dot_xy1
        
        self.predict_param = predict_param
        self.predict_depth = predict_depth
        self.plane_mask_threshold = plane_mask_threshold

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
            # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        param_l1_weight = cfg.MODEL.MASK_FORMER.PARAM_L1_WEIGHT
        param_cos_weight = cfg.MODEL.MASK_FORMER.PARAM_COS_WEIGHT
        q_weight = cfg.MODEL.MASK_FORMER.Q_WEIGHT
        center_weight = cfg.MODEL.MASK_FORMER.CENTER_WEIGHT
        plane_depths_weight = cfg.MODEL.MASK_FORMER.PLANE_DEPTHS_WEIGHT
        whole_depth_weight = cfg.MODEL.MASK_FORMER.WHOLE_DEPTH_WEIGHT
        
        # predict bool
        predict_param = cfg.MODEL.MASK_FORMER.PREDICT_PARAM
        predict_depth = cfg.MODEL.MASK_FORMER.PREDICT_DEPTH

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_param = param_l1_weight,  
            cost_depth = plane_depths_weight,
            predict_param = predict_param,
            predict_depth = predict_depth,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight,
                        "loss_param_l1": param_l1_weight, "loss_param_cos": param_cos_weight,
                        "loss_Q": q_weight, "loss_center_instance": center_weight, 
                        "loss_plane_depths": plane_depths_weight, "loss_whole_depth": whole_depth_weight,
                        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = [
                    "labels", 
                    "masks",
                    'plane_depths',
                ]

        if predict_param:
            losses.extend([
                'params', 
                'Q'
            ])
          
        if predict_depth:
            losses.extend([
                'plane_depths',
            ])

        
        k_inv_dot_xy1 = get_coordinate_map(cfg.INPUT.DATASET_MAPPER_NAME, torch.device("cpu"))
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            k_inv_dot_xy1 = k_inv_dot_xy1, 
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "k_inv_dot_xy1": k_inv_dot_xy1,
            "predict_param": predict_param,
            "predict_depth": predict_depth,
            "plane_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.PLANE_MASK_THRESHOLD,
            
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs] 
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor) 
        
        outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            K_inv_dot_xy_1s = [x["K_inv_dot_xy_1"].to(self.device) for x in batched_inputs]
            random_scales = [x["random_scale"].to(self.device) for x in batched_inputs]
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images, K_inv_dot_xy_1s, random_scales)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict: # {'loss_ce': 2.0, 'loss_mask': 5.0, 'loss_dice': 5.0, 'loss_ce_0': 2.0, 'loss_mask_0': 5.0, 'loss_dice_0': 5.0, 'loss_ce_1': 2.0, 'loss_mask_1': 5.0, 'loss_dice_1': 5.0, 'loss_ce_2': 2.0, 'loss_mask_2': 5.0, 'loss_dice_2': 5.0, 'loss_ce_3': 2.0, 'loss_mask_3': 5.0, ...}
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"] # torch.Size([b, num_queries, num_classes + 1])
            mask_pred_results = outputs["pred_masks"] # torch.Size([b, num_queries, h/4, w/4])
            param_pred_results = outputs["pred_params"] # torch.Size([b, num_queries, 3])
            depth_pred_results = outputs["pred_depths"] # torch.Size([b, num_queries, h, w]
            # upsample masks
            if not mask_pred_results.shape[-1] == images.tensor.shape[-1]:
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                ) # torch.Size([b,num_queries,h,w])

            if not depth_pred_results.shape[-1] == images.tensor.shape[-1]:
                depth_pred_results = F.interpolate(
                    depth_pred_results,
                    size = (images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode = "bilinear",
                    align_corners=False,
                ) # torch.Size([b,num_queries,h,w])

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, param_pred_result, depth_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, param_pred_results, depth_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0]) # ep 349
                width = input_per_image.get("width", image_size[1]) # ep 640
                processed_results.append({})
                # Return semantic segmentation predictions in the original resolution.
                # if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                ) 
                mask_cls_result = mask_cls_result.to(mask_pred_result) # torch.Size([num_queries, num_classes, num_classes + 1])

                # plane inference
                if self.semantic_on:
                    
                    plane_seg, inferred_planes_depth, inferred_seg_depth, valid_param = retry_if_cuda_oom(self.plane_inference)(mask_cls_result, mask_pred_result, param_pred_result, depth_pred_result)
                    processed_results[-1]["sem_seg"] = plane_seg
                    processed_results[-1]["planes_depth"] = inferred_planes_depth
                    processed_results[-1]["seg_depth"] = inferred_seg_depth
                    processed_results[-1]["valid_params"] = valid_param

            return processed_results

    def prepare_targets(self, targets, images, K_inv_dot_xy_1s, random_scales):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image, K_inv_dot_xy_1, scale in zip(targets, K_inv_dot_xy_1s, random_scales):
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "params": targets_per_image.gt_params,
                    "plane_depths": targets_per_image.gt_plane_depths,
                    "resize14_plane_depths": targets_per_image.gt_resize14_plane_depths,
                    "K_inv_dot_xy_1": K_inv_dot_xy_1,
                    "random_scale": scale,
                }
            )
        return new_targets
    
    def plane_inference(self, mask_cls, mask_pred, param_pred, depth_pred):
        mask_cls = F.softmax(mask_cls, dim=-1) # torch.Size([num_queries, num_classes + 1 = 3])
        score, labels = mask_cls.max(dim=-1)
        labels[labels != 1] = 0 # [num_queries]
        label_mask = labels > 0  # [num_queries]
        if sum(label_mask) == 0:
            _, max_pro_idx = mask_cls[:, 1].max(dim=0)
            label_mask[max_pro_idx] = 1
        valid_param = param_pred[label_mask, :]  # valid_plane_num, 3
        
        mask_pred = mask_pred.sigmoid() # torch.Size([num_queries, h, w])
        valid_mask_pred = mask_pred[label_mask] # [valid_plane_num,h,w]
        tmp = torch.zeros((self.num_queries + 1 - valid_mask_pred.shape[0], valid_mask_pred.shape[1], valid_mask_pred.shape[2]),
                        dtype = valid_mask_pred.dtype, device = valid_mask_pred.device)
        
        non_plane_mask = (valid_mask_pred > self.plane_mask_threshold).sum(0) == 0
        tmp[-1][non_plane_mask] = 1 
        plane_seg = torch.cat((valid_mask_pred, tmp), dim = 0)
        plane_seg = plane_seg.sigmoid() # [num_queries, h, w]
  
        valid_num = valid_mask_pred.shape[0]
        inferred_planes_depth = None
        inferred_seg_depth = None
        if self.predict_param:
            # get depth map
            h, w = plane_seg.shape[-2:]
            
            depth_maps_inv = torch.matmul(valid_param, self.k_inv_dot_xy1.to(self.pixel_mean.device))
            depth_maps_inv = torch.clamp(depth_maps_inv, min=0.1, max=1e4)
            depth_maps = 1. / depth_maps_inv  # (valid_plane_num, h*w)
            inferred_planes_depth = depth_maps.t()[range(h*w), plane_seg[:valid_num].argmax(dim=0).view(-1)] # plane depth [h,w]
            inferred_planes_depth = inferred_planes_depth.view(h, w)
            inferred_planes_depth[non_plane_mask] = 0.0 # del non-plane regions
        
        if self.predict_depth:
            valid_depth_pred = depth_pred[label_mask] # [valid_plane_num, h, w]
            segmentation = (plane_seg[:valid_num].argmax(dim=0)[:,:,None] == torch.arange(valid_num).to(plane_seg)).permute(2, 0, 1) # [h, w, 1] == []  -> [valid_plane_num, h, w]
            inferred_seg_depth = (segmentation * valid_depth_pred).sum(0) # [h, w]
            inferred_seg_depth[non_plane_mask] = 0.0 # del non-plane regions
        
        
        return plane_seg, inferred_planes_depth, inferred_seg_depth, valid_param
