import chainer
import copy
import matplotlib.pyplot as plt
import numpy as np

from chainercv import transforms as tr
from chainercv.links.model.ssd import transforms as ssd_tr
from matplotlib.patches import Rectangle

from cvdatasets.dataset import AnnotationsReadMixin
from cvdatasets.dataset import ImageProfilerMixin
from cvdatasets.dataset import IteratorMixin
from cvdatasets.dataset import MultiBoxMixin
from cvdatasets.dataset import TransformMixin
from cvdatasets.dataset.image import Size

from moth_detector.core.dataset.augmentations import Augmentations
from moth_detector.core.dataset.augmentations import is_bbox_ok

class BBoxDataset(
	ImageProfilerMixin,
	TransformMixin,
	IteratorMixin,
	MultiBoxMixin,
	AnnotationsReadMixin):

	# ImageNet mean (we need this if we use InceptionV3 ???)
	mean = np.array((123, 117, 104), dtype=np.float32).reshape((-1, 1, 1))

	@classmethod
	def kwargs(cls, opts, subset):
		return dict(opts=opts)

	def __init__(self, *args, opts, prepare, center_crop_on_val, **kwargs):
		super(BBoxDataset, self).__init__(*args, **kwargs)

		self.prepare = prepare
		self._setup_augmentations(opts)

		# self.center_crop_on_val = center_crop_on_val


	def _setup_augmentations(self, opts):
		Aug = Augmentations
		self._train_augs = [
			Aug.scale_down(size=1000),
			Aug.random_distort(),
			Aug.random_expand(),
			Aug.random_crop(),

			Aug.scale(size=min(self.size)),
			Aug.random_crop(size=tuple(self._size)),
			Aug.random_flip(size=tuple(self._size),
				x_random=True, y_random=True),
		]

		self._val_augs = [
			Aug.scale(size=min(self.size)),
			Aug.center_crop(size=tuple(self._size)),

		]

	def prepare_back(self, img):
		img = img + self.mean
		img = img.transpose(1, 2, 0)
		return img.astype(np.uint8)


	def pad_bbox(self, bbox, labels, *, total_boxes=128):
		padded_bbox = np.full((total_boxes, 4), -1, dtype=bbox.dtype)
		padded_labels = np.full(total_boxes, -1, dtype=labels.dtype)
		padded_bbox[:len(bbox)] = bbox
		padded_labels[:len(labels)] = labels
		return padded_bbox, padded_labels


	def transform(self, im_obj):

		img, bbox, lab = self.preprocess(im_obj)
		img, bbox = self.augment(img, bbox)
		img, bbox, lab = self.postprocess(img, bbox, lab)

		return img, bbox, lab

	def preprocess(self, im_obj):
		# we have here only one class so far
		img, lab = im_obj.im_array, np.array([0])
		img = img.transpose(2, 0, 1)

		multi_box = self.multi_box(im_obj.uuid, keys=["y0", "x0", "y1", "x1"])
		bbox = np.array(multi_box, dtype=np.int32)
		self._profile_img(img, "Init image")

		return img, bbox, lab

	@property
	def augmentations(self):
		return self._train_augs if chainer.config.train else self._val_augs

	def call_augmentation(self, img, bbox, aug_func, augments_bbox: bool = True, **params):

		if augments_bbox:
			return aug_func(img, bbox, **params)

		return aug_func(img, **params), bbox

	def augment(self, img, bbox, n_tries=5):

		# we need these checks, because random crop crops sometimes without the bbox
		for i in range(n_tries):
			new_img, new_bbox = self._augment(img, bbox)
			if is_bbox_ok(new_bbox):
				return new_img, new_bbox

		# apply validation augmentations, they always work
		with chainer.using_config("train", False):
			return self._augment(img, bbox)

	def _augment(self, img, bbox):

		# fig, axs = plt.subplots(2)
		# axs[0].imshow(img.transpose(1,2,0).astype(np.uint8))
		# axs[0].axis("off")
		# axs[0].set_title("Original Image")
		# for y0, x0, y1, x1 in bbox:
		# 	axs[0].add_patch(Rectangle((x0,y0), x1-x0, y1-y0,
		# 		fill=False, linewidth=2))

		for aug in self.augmentations:
			img, bbox = aug(img, bbox)
			self._profile_img(img, aug.__name__)

		# axs[1].set_title("final image")
		# axs[1].axis("off")
		# axs[1].imshow(img.transpose(1,2,0).astype(np.uint8))
		# for (y0, x0, y1, x1) in bbox:
		# 	axs[1].add_patch(Rectangle((x0,y0), x1-x0, y1-y0,
		# 		fill=False, linewidth=2))
		# plt.tight_layout()
		# plt.show()
		# plt.close()

		return img, bbox

	def postprocess(self, img, bbox, lab):

		img = img - self.mean
		assert is_bbox_ok(bbox), \
			f"Ill-formed bounding box: {bbox}!"

		bbox, lab = self.pad_bbox(bbox, lab)

		return img, bbox, lab
