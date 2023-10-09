from multiprocessing.dummy import Pool
from functools import partial

from moth_detector.core.detectors.base import BaseDetector



class Detector(BaseDetector):
	__name__ = "Non-Deep Detector"

	def __init__(self, model, *args, **kwargs):
		super().__init__(model, *args, **kwargs)

		self.multi_threaded = False

	def predict(self, X, preset="foo", **kw):

		bboxes, labels, scores = [], [], []

		model = partial(self.model, **kw)
		if self.multi_threaded and len(X) >= 2:

			with Pool() as pool:
				for bbox, label, score in pool.imap(model, X):
					bboxes.append(bbox)
					labels.append(label)
					scores.append(score)
		else:
			for bbox, label, score in map(model, X):
				bboxes.append(bbox)
				labels.append(label)
				scores.append(score)

		return bboxes, labels, scores

	def decode_inner(self, *args, **kwargs):
		import pdb; pdb.set_trace()

	def forward(self, *inputs):
		import pdb; pdb.set_trace()
