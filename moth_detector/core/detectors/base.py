import abc
import chainer
import numpy as np
import typing as T

from chainer.backends.cuda import to_cpu
from cvmodelz import classifiers

from moth_detector.core import evaluations as evals

class BaseDetector(classifiers.Classifier):

	def __init__(self, *args, nms_thresh, score_thresh, max_boxes, **kwargs):
		super().__init__(*args, **kwargs)
		self.model.nms_thresh = nms_thresh
		self.model.score_thresh = score_thresh
		self.max_boxes = max_boxes

	def decode(self, *args, **kwargs):
		boxes, labels, scores = self.decode_inner(*args, **kwargs)

		if self.max_boxes <= 0:
			return boxes, labels, scores

		selected_boxes, selected_labels, selected_scores = [], [], []
		for _boxes, _labs, _scores in zip(boxes, labels, scores):
			order = np.argsort(_scores)[::-1]
			selected = order[:self.max_boxes]

			mask = np.zeros_like(_scores, dtype=bool)
			mask[selected] = 1

			selected_boxes.append(_boxes[mask])
			selected_labels.append(_labs[mask])
			selected_scores.append(_scores[mask])

		return selected_boxes, selected_labels, selected_scores

	@abc.abstractmethod
	def decode_inner(self, *args, **kwargs):
		pass

	def report_mAP(self,
		pred_bboxes: T.List[np.ndarray],
		pred_labels: T.List[np.ndarray],
		pred_scores: T.List[np.ndarray],
		gt_bboxes: T.List[np.ndarray],
		gt_labels: T.List[np.ndarray],
		*,
		thresholds: T.List[float] = [0.5, 0.75]) -> None:

		_bboxes = [ to_cpu(box) for box in gt_bboxes]
		_labels = [ to_cpu(_y) for _y in gt_labels]

		pred = evals.Records(pred_bboxes, pred_labels, pred_scores)
		gt = evals.Records(_bboxes, _labels)
		for thresh in thresholds:
			result = evals.VOCEvaluations.evaluate(
				pred, gt, iou_thresh=thresh)

			self.report(**{f"map@{int(thresh*100)}": result})
