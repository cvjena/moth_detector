from moth_detector.core.dataset.bbox_dataset import BBoxDataset
from moth_detector.core.dataset.augmentations import Augmentations
from moth_detector.core.dataset.augmentations import is_bbox_ok


def kwargs(opts):
	return dict(
		dataset_cls=BBoxDataset,
		dataset_kwargs_factory=BBoxDataset.kwargs(opts),
	)
