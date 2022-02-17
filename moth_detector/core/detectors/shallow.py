from multiprocessing.dummy import Pool

from moth_detector.core.detectors.base import BaseDetector



class Detector(BaseDetector):
	__name__ = "Non-Deep Detector"

	def __init__(self, model, *args, **kwargs):
		super().__init__(model)

		self.multi_threaded = True

	def predict(self, X, preset="foo"):

		bboxes, labels, scores = [], [], []

		if self.multi_threaded:

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

	def __call__(self, *inputs):
		import pdb; pdb.set_trace()
