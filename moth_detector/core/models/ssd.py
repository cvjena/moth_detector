import numpy as np

from chainer.serializers import npz
from chainercv.evaluations import eval_detection_voc
from chainercv.links.model.ssd import SSD300

class Model(SSD300):

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
