import numpy as np
import typing as T

from chainercv import evaluations as Eval
from chainercv.utils.bbox.bbox_iou import bbox_iou

class Records(T.NamedTuple):
	bboxes: T.List[np.ndarray]
	labels: T.List[np.ndarray]
	scores: T.Optional[T.List[np.ndarray]] = None


class _Static:
	def __new__(cls):
		raise TypeError("Static class only!")

class APEvaluations(_Static):
	@classmethod
	def evaluate(cls, pred: Records, gt: Records, *, iou_thresh: float = 0.5, **kwargs):
		res = []
		for entries in zip(*pred, *gt[:-1]):
			pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels = entries
			mask = gt_labels != -1
			gt_bboxes = gt_bboxes[mask]
			gt_labels = gt_labels[mask]

			selec = np.zeros(gt_bboxes.shape[0], dtype=bool)
			iou = bbox_iou(pred_bboxes, gt_bboxes)
			gt_index = iou.argmax(axis=1)
			# set -1 if there is no matching ground truth
			gt_index[iou.max(axis=1) < iou_thresh] = -1
			matched = []

			for idx in gt_index:
				if idx == -1:
					matched.append(0)
					continue

				if selec[idx]:
					matched.append(0)
					continue

				matched.append(1)
				selec[idx] = True

			#res.append(np.mean(matched))
			res.append(np.mean(selec))

		return np.array(res)

class VOCEvaluations(_Static):

	@classmethod
	def prec_rec(cls, pred: Records, gt: Records, *args, **kwargs):
		return Eval.calc_detection_voc_prec_rec(
				pred.bboxes, pred.labels, pred.scores,
				gt.bboxes, gt.labels,
				*args, **kwargs)

	@classmethod
	def avg_prec(cls, prec, rec, *args, **kwargs):
		return Eval.calc_detection_voc_ap(prec, rec, *args, **kwargs)

	@classmethod
	def evaluate(cls, pred: Records, gt: Records, **kwargs):

		for scores in pred.scores:
			if -1 not in scores:
				continue

			# we land here if a -1 is found
			return cls.evaluate_other(pred, gt, **kwargs)

		else:
			# if not a single -1 is found, use
			# default VOC metrics
			return cls.evaluate_voc(pred, gt, **kwargs)

	@classmethod
	def evaluate_voc(cls, pred: Records, gt: Records, *,
					 iou_thresh: float = 0.5, **kwargs):

		results = []
		for entries in zip(*pred, *gt[:-1]):
			pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels = entries
			result = Eval.eval_detection_voc(
					[pred_bboxes], [pred_labels], [pred_scores],
					[gt_bboxes], [gt_labels],
					iou_thresh=iou_thresh,
					**kwargs)
			results.append(result["map"])
		return np.mean(results)

	@classmethod
	def evaluate_other(cls, pred: Records, gt: Records, *,
					   iou_thresh: float = 0.5, **kwargs):
		return cls.evaluate_voc(pred, gt, iou_thresh=iou_thresh, **kwargs)
		results = []
		for entries in zip(*pred, *gt[:-1]):
			pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels = entries
			prec, rec = Eval.calc_detection_voc_prec_rec(
				[pred_bboxes], [pred_labels], [pred_scores],
				[gt_bboxes], [gt_labels],
				iou_thresh=iou_thresh)


			import pdb; pdb.set_trace()

class COCOEvaluations(_Static):

	@classmethod
	def evaluate(cls, pred: Records, gt: Records, *args, **kwargs):

		result = Eval.eval_detection_coco(
				pred.bboxes, pred.labels, pred.scores,
				gt.bboxes, gt.labels,
				*args, **kwargs)
		return {key.replace("/", ", "): float(value) for key, value in result.items()
			if key not in ("coco_eval", "existent_labels")}
