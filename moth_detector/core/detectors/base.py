import abc
import chainer
import numpy as np

from cvmodelz import classifiers
from chainer.backends.cuda import to_cpu
from chainercv.evaluations import eval_detection_voc
from typing import List

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
		pred_bboxes: List[np.ndarray],
		pred_labels: List[np.ndarray],
		pred_scores: List[np.ndarray],
		gt_bboxes: List[np.ndarray],
		gt_labels: List[np.ndarray],
		*,
		thresholds: List[float] = [0.5, 0.75]) -> None:

		_bboxes = [ to_cpu(box) for box in gt_bboxes]
		_labels = [ to_cpu(_y) for _y in gt_labels]

		for thresh in thresholds:
			result = eval_detection_voc(
				pred_bboxes, pred_labels, pred_scores,
				_bboxes, _labels, iou_thresh=thresh)

			self.report(**{f"map@{int(thresh*100)}": result["map"]})
