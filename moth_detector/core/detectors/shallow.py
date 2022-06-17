from multiprocessing.dummy import Pool

from moth_detector.core.detectors.base import BaseDetector



class Detector(BaseDetector):
	__name__ = "Non-Deep Detector"

	def __init__(self, model, *args, **kwargs):
		super().__init__(model, *args, **kwargs)

		self.multi_threaded = False

	def predict(self, X, preset="foo"):

		bboxes, labels, scores = [], [], []

		if self.multi_threaded and len(X) >= 2:

			with Pool() as pool:
				for bbox, label, score in pool.imap(self.model, X):
					bboxes.append(bbox)
					labels.append(label)
					scores.append(score)
		else:
			for bbox, label, score in map(self.model, X):
				bboxes.append(bbox)
				labels.append(label)
				scores.append(score)

		return bboxes, labels, scores

	def decode_inner(self, *args, **kwargs):
		import pdb; pdb.set_trace()

	def forward(self, *inputs):
		import pdb; pdb.set_trace()
