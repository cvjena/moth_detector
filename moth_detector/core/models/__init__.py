from moth_detector.core.models.ssd import Model as SSD_Model
from moth_detector.core.models.rcnn import Model as FRCNN_Model
from moth_detector.core.models.shallow import Model as Shallow_Model
from moth_detector.core.models.shallow import MCCModel

__all__ = [
	"SSD_Model",
	"FRCNN_Model",
	"Shallow_Model",
	"MCCModel",
]
