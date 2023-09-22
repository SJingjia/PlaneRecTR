from detectron2.config import CfgNode as CN 

def add_PlaneRecTR_config(cfg):
    """
    Add config for PlaneRecTR.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    # cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic" #!
    cfg.INPUT.DATASET_MAPPER_NAME = "scannetv1_plane"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    cfg.INPUT.RESIZE = True
    cfg.INPUT.BRIGHT_COLOR_CONTRAST = False

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0
    cfg.MODEL.MASK_FORMER.PARAM_L1_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.PARAM_COS_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.Q_WEIGHT = 2.0
    cfg.MODEL.MASK_FORMER.CENTER_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.PLANE_DEPTHS_WEIGHT = 2.0
    cfg.MODEL.MASK_FORMER.WHOLE_DEPTH_WEIGHT = 2.0


    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8 # !
    # cfg.MODEL.MASK_FORMER.NHEADS = 4
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 20

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.MASK_FORMER.TEST.PLANE_MASK_THRESHOLD = 0.5

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # pixel depth decoder config
    cfg.MODEL.SEM_SEG_HEAD.DEPTH_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False



    # hrnet32
    cfg.MODEL.arch = "hrnet_32"
    cfg.MODEL.hrnet_w32 = CN()
    cfg.MODEL.hrnet_w32.PRETRAINED = './checkpoint/hrnetv2_w32_imagenet_pretrained_new.pth'
    cfg.MODEL.hrnet_w32.STAGE1 = CN()
    cfg.MODEL.hrnet_w32.STAGE1.NUM_MODULES=1
    cfg.MODEL.hrnet_w32.STAGE1.NUM_BRANCHES= 1
    cfg.MODEL.hrnet_w32.STAGE1.BLOCK= None
    cfg.MODEL.hrnet_w32.STAGE1.NUM_BLOCKS= [4]
    cfg.MODEL.hrnet_w32.STAGE1.NUM_CHANNELS= [64]
    cfg.MODEL.hrnet_w32.STAGE1.FUSE_METHOD= None
    cfg.MODEL.hrnet_w32.STAGE2 = CN()
    cfg.MODEL.hrnet_w32.STAGE2.NUM_MODULES=1
    cfg.MODEL.hrnet_w32.STAGE2.NUM_BRANCHES= 2
    cfg.MODEL.hrnet_w32.STAGE2.BLOCK= None
    cfg.MODEL.hrnet_w32.STAGE2.NUM_BLOCKS= [4,4]
    cfg.MODEL.hrnet_w32.STAGE2.NUM_CHANNELS= [32, 64]
    cfg.MODEL.hrnet_w32.STAGE2.FUSE_METHOD= None
    cfg.MODEL.hrnet_w32.STAGE3 = CN()
    cfg.MODEL.hrnet_w32.STAGE3.NUM_MODULES=4
    cfg.MODEL.hrnet_w32.STAGE3.NUM_BRANCHES= 3
    cfg.MODEL.hrnet_w32.STAGE3.BLOCK= None
    cfg.MODEL.hrnet_w32.STAGE3.NUM_BLOCKS= [4,4,4]
    cfg.MODEL.hrnet_w32.STAGE3.NUM_CHANNELS= [32,64,128]
    cfg.MODEL.hrnet_w32.STAGE3.FUSE_METHOD= None
    cfg.MODEL.hrnet_w32.STAGE4 = CN()
    cfg.MODEL.hrnet_w32.STAGE4.NUM_MODULES=3
    cfg.MODEL.hrnet_w32.STAGE4.NUM_BRANCHES= 4
    cfg.MODEL.hrnet_w32.STAGE4.BLOCK= None
    cfg.MODEL.hrnet_w32.STAGE4.NUM_BLOCKS= [4,4,4,4]
    cfg.MODEL.hrnet_w32.STAGE4.NUM_CHANNELS= [32, 64, 128, 256]
    cfg.MODEL.hrnet_w32.STAGE4.FUSE_METHOD= None
    # cfg.MODEL.hrnet_w32.WINDOW_SIZE = 7
    # cfg.MODEL.hrnet_w32.MLP_RATIO = 4.0
    # cfg.MODEL.hrnet_w32.QKV_BIAS = True
    # cfg.MODEL.hrnet_w32.QK_SCALE = None
    # cfg.MODEL.hrnet_w32.DROP_RATE = 0.0
    # cfg.MODEL.hrnet_w32.ATTN_DROP_RATE = 0.0
    # cfg.MODEL.hrnet_w32.DROP_PATH_RATE = 0.3
    # cfg.MODEL.hrnet_w32.APE = False
    # cfg.MODEL.hrnet_w32.PATCH_NORM = True
    cfg.MODEL.hrnet_w32.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    # cfg.MODEL.hrnet_32.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    # cfg.INPUT.IMAGE_SIZE = 1024 #!
    cfg.INPUT.IMAGE_SIZE = (192, 256)
    # cfg.INPUT.MIN_SCALE = 0.1
    # cfg.INPUT.MAX_SCALE = 2.0 # !
    cfg.INPUT.MIN_SCALE = 0.6
    cfg.INPUT.MAX_SCALE = 1.5

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    cfg.MODEL.MASK_FORMER.PREDICT_CENTER = False
    cfg.MODEL.MASK_FORMER.PREDICT_PARAM = True
    cfg.MODEL.MASK_FORMER.PREDICT_DEPTH = True

    cfg.TEST.VIS_PERIOD = 30



