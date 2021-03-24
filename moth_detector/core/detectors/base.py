import abc
import chainer
import numpy as np

from chainer.backends.cuda import to_cpu
from chainercv.evaluations import eval_detection_voc
from typing import List

class BaseDetector(abc.ABC):

	def report(self, **kwargs):
		return chainer.report(kwargs, self)


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
