from . import data  # register all new datasets
from . import modeling

# config
from .config import add_PlaneRecTR_config

# dataset loading
# from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
# from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
# from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
#     MaskFormerInstanceDatasetMapper,
# )
# from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
#     MaskFormerPanopticDatasetMapper,
# )
# from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
#     MaskFormerSemanticDatasetMapper,
# )
from .data.dataset_mappers.scannetv1_plane_dataset_mapper import (
    SingleScannetv1PlaneDatasetMapper,
)

from .data.dataset_mappers.nyuv2_plane_dataset_mapper import (
    SingleNYUv2PlaneDatasetMapper,
)
# models
from .PlaneRecTR_model import PlaneRecTR
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
# from .evaluation.instance_evaluation import InstanceSegEvaluator
from .evaluation.planeSeg_evaluation import PlaneSegEvaluator
