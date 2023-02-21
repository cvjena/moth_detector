from moth_detector.core.models.ssd import Model as SSD_Model
from moth_detector.core.models.rcnn import Model as FRCNN_Model
from moth_detector.core.models.shallow import Model as Shallow_Model
from moth_detector.core.models.shallow import MCCModel

__all__ = [
	"SSD_Model",
	"FRCNN_Model",
	"Shallow_Model",
	"MCCModel",
	"get",
]


def get(model_type: str):

	_models = {
		"chainercv.SSD300": SSD_Model,
		"chainercv.FasterRCNNVGG16": FRCNN_Model,
		"shallow": Shallow_Model,
		"mcc": MCCModel,
	}

	assert model_type in _models, \
		f"Model type not found: {model_type}"

	return _models.get(model_type)
