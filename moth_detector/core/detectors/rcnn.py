import numpy as np

from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain as FRCNN_Detector

from moth_detector.core.models.base import BaseModel


class Detector(FRCNN_Detector, BaseModel):
	__name__ = "Fast R-CNN Detector"

	def __init__(self, model, *args, loss_func, **kwargs):
		super(Detector, self).__init__(faster_rcnn=model, *args, **kwargs)

	@property
	def model(self):
		return self.faster_rcnn

	def forward(self, X, boxes, y):
		scales = self.xp.ones_like(y)
		return super(Detector, self).forward(X, boxes, y, scales)
