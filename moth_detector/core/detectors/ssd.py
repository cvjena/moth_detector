import chainer
import numpy as np

from chainer.backends.cuda import to_cpu
from chainercv.evaluations import eval_detection_voc
from chainercv.links.model.ssd import multibox_loss

def _unpack(arr):
	""" in case of chainer.Variable, return the actual array
		otherwise return itself
	"""
	return getattr(arr, "array", arr)

class Detector(chainer.Chain):

	def __init__(self, model, *, loss_func, k=3, alpha=1):

		super(Detector, self).__init__()
		with self.init_scope():
			self.model = model

			self.add_persistent("k", k)
			self.add_persistent("alpha", alpha)

		self.loss_func = multibox_loss

	def decode(self, loc, conf):
		return self.model.coder.decode(
			_unpack(loc), _unpack(conf),
			self.model.nms_thresh,
			self.model.score_thresh)

	def decode_all(self, locs, confs):
		bboxes, labels, scores = [], [], []

		for loc, conf in zip(_unpack(locs), _unpack(confs)):
			bbox, label, score = self.decode(loc, conf)
			bboxes.append(to_cpu(bbox))
			labels.append(to_cpu(label))
			scores.append(to_cpu(score))

		return bboxes, labels, scores

	def encode(self, box, label):
		box = box.astype(np.float32)
		return self.model.coder.encode(box, label)

	def encode_all(self, boxes, labels):
		locs, confs = [], []

		for box, label in zip(boxes, labels):
			loc, conf = self.encode(box, label)
			locs.append(loc)
			confs.append(conf)

		return self.xp.stack(locs), self.xp.stack(confs)

	def mask_real_boxes(self, box, labels, *, value=-1):
		if labels.ndim == 2:
			real_boxes, real_labels = [], []
			for lab, b in zip(labels, box):
				mask = lab != value
				real_boxes.append(b[mask])
				real_labels.append(lab[mask])
			return real_boxes, real_labels

		elif labels.ndim == 1:
			mask = labels != value
			return box[mask], labels[mask]

	def predict(self, X, preset="evaluate"):

		X = self.xp.array(X)

		if preset is not None:
			self.model.use_preset(preset)

		with chainer.using_config("train", False), chainer.no_backprop_mode():
			return self.decode_all(*self.model(X))

	def report(self, **kwargs):
		return chainer.report(kwargs, self)

	def __call__(self, *inputs):

		X, boxes, y = inputs
		mb_locs, mb_confs = self.model(X)

		boxes, y = self.mask_real_boxes(boxes, y)

		gt_mb_locs, gt_mb_confs = self.encode_all(boxes, y)
		loc_loss, conf_loss = self.loss_func(
			mb_locs, mb_confs, gt_mb_locs, gt_mb_confs, self.k)
		loss = loc_loss * self.alpha + conf_loss


		pred_bboxes, pred_labels, pred_scores = self.decode_all(mb_locs, mb_confs)
		_boxes = [ to_cpu(box) for box in boxes]
		_labels = [ to_cpu(_y) for _y in y]

		for thresh in [ 0.5, 0.75 ]:
			result = eval_detection_voc(
				pred_bboxes, pred_labels, pred_scores,
				_boxes, _labels, iou_thresh=thresh)

			self.report(**{f"map@{int(thresh*100)}": result["map"]})

		self.report(
			loss=loss,
			loc_loss=loc_loss,
			conf_loss=conf_loss
		)

		return loss
