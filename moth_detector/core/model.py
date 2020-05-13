import chainer
import numpy as np

from chainercv.links.model import ssd
from chainer.serializers import npz

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

	def __call__(self, *inputs):

		X, gt_locs, gt_labs = inputs
		mb_locs, mb_confs = self.model(X)

		loc_loss, conf_loss = self.loss_func(
			mb_locs, mb_confs, gt_locs, gt_labs, self.k)

		loss = loc_loss * self.alpha + conf_loss
		chainer.report(dict(
			loss=loss,
			loc_loss=loc_loss,
			conf_loss=conf_loss
		), self)

		return loss
