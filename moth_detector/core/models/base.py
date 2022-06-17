import abc

from chainer.serializers import npz


class BaseModel(abc.ABC):
	clf_layer_name = None

	def __init__(self, input_size, **kwargs):
		super(BaseModel, self).__init__(**kwargs)
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

	def preprocess(self, x, **kwargs):
		return x

	def use_preset(self, preset):
		# if preset != "visualize":
		# 	return super().use_preset(preset)

		self.nms_thresh = 0.45
		self.score_thresh = 0.1
