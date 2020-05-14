import chainer
import numpy as np

from chainer.backends.cuda import to_cpu
from chainer.serializers import npz
from chainercv.evaluations import eval_detection_voc
from chainercv.links.model import ssd

def _unpack(arr):
	""" in case of chainer.Variable, return the actual array
		otherwise return itself
	"""
	return getattr(arr, "array", arr)

class Model(ssd.SSD300):

	class meta:
		mean = np.array((123, 117, 104)).reshape((-1, 1, 1))
		input_size = 300
		feature_size = 4096

		def prepare_func(x, size=None, *args, **kwargs):
			import pdb; pdb.set_trace()

	def __init__(self, input_size, **kwargs):
		super(Model, self).__init__(**kwargs)
		self.input_size = input_size

	def load_for_finetune(self, weights, n_classes, *, path="", strict=False, headless=False, **kwargs):
		return self.load(weights, path=path, strict=strict, headless=headless)

	def load_for_inference(self, weights, n_classes, *, path="", strict=False, headless=False, **kwargs):
		return self.load(weights, path=path, strict=strict, headless=headless)

	def load(self, weights, *, path="", strict=False, headless=False):
		if weights in [None, "auto"]:
			logging.warning("Attempted to load default weights or no weights were given!")
			return

		npz.load_npz(weights, self, path=path, strict=strict)



	def reinitialize_clf(self, n_classes, feat_size=None, initializer=None):
		pass

class Detector(chainer.Chain):

	def __init__(self, model, *, loss_func, k=3, alpha=1):

		super(Detector, self).__init__()
		with self.init_scope():
			self.model = model

			self.add_persistent("k", k)
			self.add_persistent("alpha", alpha)

		self.loss_func = ssd.multibox_loss

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
		return self.model.coder.encode(box.astype(np.float32), label)

	def encode_all(self, boxes, labels):
		locs, confs = [], []

		for box, label in zip(boxes, labels):
			loc, conf = self.encode(box, label)
			locs.append(loc)
			confs.append(conf)

		return self.xp.stack(locs), self.xp.stack(confs)

	def predict(self, X, preset="evaluate"):

		X = self.xp.array(X)
		self.model.use_preset(preset)
		with chainer.using_config("train", False), chainer.no_backprop_mode():
			return self.decode_all(*self.model(X))

	def report(self, **kwargs):
		return chainer.report(kwargs, self)

	def __call__(self, *inputs):

		X, boxes, y = inputs
		mb_locs, mb_confs = self.model(X)

		gt_mb_locs, gt_mb_confs = self.encode_all(boxes, y)
		loc_loss, conf_loss = self.loss_func(
			mb_locs, mb_confs, gt_mb_locs, gt_mb_confs, self.k)
		loss = loc_loss * self.alpha + conf_loss


		pred_bboxes, pred_labels, pred_scores = self.decode_all(mb_locs, mb_confs)
		for thresh in [ 0.5, 0.75 ]:
			result = eval_detection_voc(
				pred_bboxes, pred_labels, pred_scores,
				to_cpu(boxes), to_cpu(y), iou_thresh=thresh)

			self.report(**{f"map@{int(thresh*100)}": result["map"]})

		self.report(
			loss=loss,
			loc_loss=loc_loss,
			conf_loss=conf_loss
		)

		return loss
