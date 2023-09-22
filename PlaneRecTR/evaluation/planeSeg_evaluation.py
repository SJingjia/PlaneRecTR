# https://github.com/facebookresearch/Mask2Former
# 

import itertools
import json
import logging
import numpy as np
import os
from os.path import join as pjoin
from collections import OrderedDict
import torch

from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.evaluator import DatasetEvaluator


_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False

from ..utils.disp import (
    visualizationBatch, 
    plot_depth_recall_curve,
    plot_normal_recall_curve,
    plot_offset_recall_curve,
)
from ..utils.misc import get_coordinate_map
from ..utils.metrics import (
    evaluateMasks, 
    eval_plane_recall_depth, 
    eval_plane_recall_normal,
    eval_plane_recall_offset,
)
from ..utils.metrics_de import evaluateDepths

from ..utils.metrics_onlyparams import eval_plane_bestmatch_normal_offset

class PlaneSegEvaluator(DatasetEvaluator):
    """
    Evaluate plane segmentation metrics.
    """
    eval_iter = 0

    def __init__(
        self,
        dataset_name,
        output_dir=None,
        *,
        num_planes=None,
        vis = False,
        vis_period = 50,
        eval_period = 500,
    ):
        self._logger = logging.getLogger(__name__)
        if num_planes is not None:
            self._logger.warn(
                "PlaneSegEvaluator(num_planes) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name.split('_')[1] + '_' + dataset_name.split('_')[2]
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._num_planes = num_planes
        self._num_queries = num_planes + 1 if "npr" in dataset_name else num_planes # TODO: add npr
        self.vis = vis
        self.vis_period = vis_period
        self.eval_period = eval_period
        self.k_inv_dot_xy1 = get_coordinate_map(dataset_name, self._cpu_device).numpy()

    def reset(self):
        
        self.RI_VI_SC = []
        self.pixelDepth_recall_curve_of_GTpd = np.zeros((13))
        self.planeDepth_recall_curve_of_GTpd = np.zeros((13, 3))
    
        self.pixelNorm_recall_curve = np.zeros((13))
        self.planeNorm_recall_curve = np.zeros((13, 3))

        self.pixelOff_recall_curve = np.zeros((13))
        self.planeOff_recall_curve = np.zeros((13, 3))

        self.bestmatch_normal_errors = []
        self.bestmatch_offset_errors = []

        if "nyuv2_plane" in self._dataset_name:
                self.depth_estimation_metrics = np.zeros((8)) # rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3
 
        if self.vis:
            self.vis_dicts = []
            self.gt_vis_dicts = []
            self.file_names = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            sem_seg = output["sem_seg"].argmax(dim=0).to(self._cpu_device) # torch.Size([480, 640]) # sem_seg 21, 192, 256
            pred = np.array(sem_seg, dtype=np.int) # (480, 640)
            plane_depth = output["planes_depth"].to(self._cpu_device).numpy()
            seg_depth = output["seg_depth"].to(self._cpu_device).numpy()
            valid_params = output["valid_params"].to(self._cpu_device)

            if self._dataset_name=="scannetv1_plane" or self._dataset_name=="nyuv2_plane":
                gt_filename = input["npz_file_name"]
                npz_data = np.load(gt_filename)
                gt = npz_data["segmentation"]

                gt_plane_depth = npz_data["depth"][0] # b#??, h, w
                gt_raw_depth = npz_data["raw_depth"]
                gt_params = npz_data["plane"]           
            else:
                print(self._dataset_name)

            if self.vis:
                if self._dataset_name=="scannetv1_plane" or self._dataset_name=="nyuv2_plane":
                    image = npz_data["image"] #BGR
                    file_name = os.path.split(input["npz_file_name"])[-1].split(".")[0]
                    
                self.gt_vis_dicts.append({
                    'image': image, 
                    'segmentation': gt,
                    'depth_GTplane': gt_plane_depth,
                    'K_inv_dot_xy_1': self.k_inv_dot_xy1,
                    })
                self.vis_dicts.append({
                    'image': image,
                    'segmentation': pred,
                    'depth_predplane': plane_depth,
                    'K_inv_dot_xy_1': self.k_inv_dot_xy1,
                })
                self.file_names.append(file_name)

            self.RI_VI_SC.append(evaluateMasks(pred, gt, device = "cuda",  pred_non_plane_idx = self._num_planes+1, gt_non_plane_idx=self._num_planes, printInfo=False))

            # ----------------------------------------------------- evaluation
            # 1 evaluation: plane segmentation
            valid_plane_num = len(valid_params)
            pixelStatistics, planeStatistics = eval_plane_recall_depth(
                pred, gt, plane_depth, gt_plane_depth, valid_plane_num)
            self.pixelDepth_recall_curve_of_GTpd += np.array(pixelStatistics)
            self.planeDepth_recall_curve_of_GTpd += np.array(planeStatistics)

            # 2 evaluation: plane segmentation
            instance_param = valid_params.cpu().numpy()
            plane_recall, pixel_recall = eval_plane_recall_normal(pred, gt,
                                        instance_param, gt_params,
                                        )
            self.pixelNorm_recall_curve += pixel_recall
            self.planeNorm_recall_curve += plane_recall


            # 3 evaluation: plane offset
            instance_param = valid_params.cpu().numpy()
            plane_recall, pixel_recall = eval_plane_recall_offset(pred, gt,
                                        instance_param, gt_params,
                                        )
            self.pixelOff_recall_curve += pixel_recall
            self.planeOff_recall_curve += plane_recall

            instance_param = valid_params.numpy()
            normal_error, offset_error = eval_plane_bestmatch_normal_offset(instance_param, gt_params)
            self.bestmatch_normal_errors.append(normal_error)
            self.bestmatch_offset_errors.append(offset_error)

            if "nyuv2_plane" in self._dataset_name:
                # self.depth_estimation_metrics += evaluateDepths(plane_depth, gt_raw_depth, pred, gt, file_name, True)
                self.depth_estimation_metrics += evaluateDepths(plane_depth, gt_raw_depth, pred, gt, None, False)
            

    def evaluate(self):

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)

        PlaneSegEvaluator.eval_iter += self.eval_period
        res = {}
        
        # print("len(RI_VI_SC)",len(self.RI_VI_SC))
        res_RI_VI_SC = np.sum(self.RI_VI_SC, axis = 0)/len(self.RI_VI_SC)
        res["RI"] = res_RI_VI_SC[0]
        res["VI"] = res_RI_VI_SC[1]
        res["SC"] = res_RI_VI_SC[2]
        if "nyuv2_plane" in self._dataset_name: # rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3
            res_depth_estimation_metrics = self.depth_estimation_metrics/len(self.RI_VI_SC)
            res["DE_rel"], res["DE_rel_sqr"], res["DE_log10"], res["DE_rmse"], \
            res["DE_rmse_log"], res["DE_accuracy_1"], res["DE_accuracy_2"], res["DE_accuracy_3"] = res_depth_estimation_metrics
        
        if self._output_dir:
            
            file_path = pjoin(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)

            if self.vis:

                vis_path = pjoin(self._output_dir, "vis_" + str(PlaneSegEvaluator.eval_iter))

                if not os.path.exists(vis_path):
                    os.makedirs(vis_path)
                    
                for i in range(len(self.vis_dicts)):
                    
                    if i % self.vis_period == 0:
                        
                        visualizationBatch(root_path = vis_path, idx = self.file_names[i], info = "gt",
                        data_dict = self.gt_vis_dicts[i], num_queries = self._num_queries, save_image = True, save_segmentation = True,
                        save_depth = True, save_ply = True, save_cloud = False) 
                        visualizationBatch(root_path = vis_path, idx = self.file_names[i], info = "pred",
                        data_dict = self.vis_dicts[i], num_queries = self._num_queries, save_image = True, save_segmentation = True,
                        save_depth = True, save_ply = True, save_cloud = False)

                recall_curve_save_path = pjoin(vis_path, "recall_curve")
                if not os.path.exists(recall_curve_save_path):
                    os.makedirs(recall_curve_save_path)
                
                mine_recalls_pixel = {"PlaneRecTR (Ours)": self.pixelDepth_recall_curve_of_GTpd / len(self.RI_VI_SC) * 100}
                mine_recalls_plane = {"PlaneRecTR (Ours)": self.planeDepth_recall_curve_of_GTpd[:, 0] / self.planeDepth_recall_curve_of_GTpd[:, 1] * 100}
                res['per_pixel_depth_01'] = mine_recalls_pixel["PlaneRecTR (Ours)"][2]
                res['per_pixel_depth_06'] = mine_recalls_pixel["PlaneRecTR (Ours)"][-1]
                res['per_plane_depth_01'] = mine_recalls_plane["PlaneRecTR (Ours)"][2]
                res['per_plane_depth_06'] = mine_recalls_plane["PlaneRecTR (Ours)"][-1]
                # print("mine_recalls_pixel (pred_planed vs gt_planed)", mine_recalls_pixel)
                # print("mine_recalls_plane (pred_planed vs gt_planed)", mine_recalls_plane)
                plot_depth_recall_curve(mine_recalls_pixel, type='pixel (pred_planed vs gt_planed)', save_path=recall_curve_save_path)
                plot_depth_recall_curve(mine_recalls_plane, type='plane (pred_planed vs gt_planed)', save_path=recall_curve_save_path)

                normal_recalls_pixel = {"PlaneRecTR": self.pixelNorm_recall_curve / len(self.RI_VI_SC) * 100}
                normal_recalls_plane = {"PlaneRecTR": self.planeNorm_recall_curve[:, 0] / self.planeNorm_recall_curve[:, 1] * 100}
                res['per_pixel_normal_5'] = normal_recalls_pixel["PlaneRecTR"][2]
                res['per_pixel_normal_30'] = normal_recalls_pixel["PlaneRecTR"][-1]
                res['per_plane_normal_5'] = normal_recalls_plane["PlaneRecTR"][2]
                res['per_plane_normal_30'] = normal_recalls_plane["PlaneRecTR"][-1]
                # print("normal_recalls_pixel", normal_recalls_pixel)
                # print("normal_recalls_plane", normal_recalls_plane)
                plot_normal_recall_curve(normal_recalls_pixel, type='pixel', save_path=recall_curve_save_path)
                plot_normal_recall_curve(normal_recalls_plane, type='plane', save_path=recall_curve_save_path)

                offset_recalls_pixel = {"PlaneRecTR": self.pixelOff_recall_curve / len(self.RI_VI_SC) * 100}
                offset_recalls_plane = {"PlaneRecTR": self.planeOff_recall_curve[:, 0] / self.planeOff_recall_curve[:, 1] * 100}
                res['per_pixel_offset_17'] = offset_recalls_pixel["PlaneRecTR"][2]
                res['per_pixel_offset_100'] = offset_recalls_pixel["PlaneRecTR"][-1]
                res['per_plane_offset_17'] = offset_recalls_plane["PlaneRecTR"][2]
                res['per_plane_offset_100'] = offset_recalls_plane["PlaneRecTR"][-1]
                # print("offset_recalls_pixel", offset_recalls_pixel)
                # print("offset_recalls_plane", offset_recalls_plane)
                plot_offset_recall_curve(offset_recalls_pixel, type='pixel', save_path=recall_curve_save_path)
                plot_offset_recall_curve(offset_recalls_plane, type='plane', save_path=recall_curve_save_path)

                res["normal_error"] = np.mean(self.bestmatch_normal_errors)
                res["offset_error"] = np.mean(self.bestmatch_offset_errors)
                # print("bestmatch_normal_errors", res["normal_error"])
                # print("bestmatch_offset_errors", res["offset_error"])
                
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)

        return results
