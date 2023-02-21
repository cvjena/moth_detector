from moth_detector.core.detectors.ssd import Detector as SSD_Detector
from moth_detector.core.detectors.rcnn import Detector as FRCNN_Detector
from moth_detector.core.detectors.shallow import Detector as Shallow_Detector


__all__ = [
	"SSD_Detector",
	"FRCNN_Detector",

	"Shallow_Detector"
]


def model_kwargs(model_type):

	if model_type in ["shallow", "mcc"]:
		return {}

	return dict(
		n_fg_class=1,
		pretrained_model='imagenet',
	)


def kwargs(opts):

	_detectors = {
		"chainercv.SSD300": SSD_Detector,
		"chainercv.FasterRCNNVGG16": FRCNN_Detector,
		"shallow": Shallow_Detector,
		"mcc": Shallow_Detector,
	}
	model_type = opts.model_type

	assert model_type in _detectors, \
		f"Detector type not found: {model_type}"

	cls = _detectors.get(model_type)

	return dict(
		classifier_cls=cls,
		classifier_kwargs=dict(
			nms_thresh=opts.nms_threshold,
			score_thresh=opts.score_threshold,
			max_boxes=opts.max_boxes,
		),

		model_kwargs=model_kwargs(model_type),
	)
