from moth_detector.core.dataset.bbox_dataset import BBoxDataset
from moth_detector.core.dataset.augmentations import Augmentations
from moth_detector.core.dataset.augmentations import is_bbox_ok


# import chainer
# import copy
# import numpy as np

# from chainercv import transforms as tr
# from chainercv.links.model.ssd import transforms as ssd_tr

# from cvdatasets.dataset import AnnotationsReadMixin
# from cvdatasets.dataset import ImageProfilerMixin
# from cvdatasets.dataset import IteratorMixin
# from cvdatasets.dataset import MultiBoxMixin
# from cvdatasets.dataset import TransformMixin
# from cvdatasets.dataset.image import Size

# import matplotlib.pyplot as plt

# from matplotlib.patches import Rectangle

# class BBoxDataset(
# 	ImageProfilerMixin,
# 	TransformMixin,
# 	IteratorMixin,
# 	MultiBoxMixin,
# 	AnnotationsReadMixin):

# 	# ImageNet mean (we need this if we use InceptionV3 ???)
# 	mean = np.array((123, 117, 104), dtype=np.float32).reshape((-1, 1, 1))

# 	@classmethod
# 	def kwargs(cls, opts, subset):
# 		return dict(opts=opts)

# 	def __init__(self, *args, opts, prepare, center_crop_on_val, **kwargs):
# 		super(BBoxDataset, self).__init__(*args, **kwargs)

# 		self.prepare = prepare
# 		self._pre_rescale = 1000
# 		self._setup_augmentations(opts)

# 		# self.center_crop_on_val = center_crop_on_val

# 	def is_bbox_ok(self, bbox) -> bool:
# 		return bbox.shape[0] >= 1 and bbox.shape[1] == 4

# 	def _setup_augmentations(self, opts):
# 		self._train_augs = [
# 			(self._scale_down, dict()),

# 			(ssd_tr.random_distort, dict(augments_bbox=False)),
# 			(self._random_expand, dict()),
# 			(self._random_crop, dict()),
# 			(self._scale, dict(size=min(self.size))),
# 			(self._random_crop, dict(size=tuple(self._size))),
# 			(self._random_flip, dict(size=tuple(self._size),
# 				x_random=True, y_random=True)),
# 		]
# 		self._val_augs = [
# 			(self._scale, dict(size=min(self.size))),
# 			(self._center_crop, dict(size=tuple(self._size))),
# 		]

# 	def prepare_back(self, img):
# 		img = img + self.mean
# 		img = img.transpose(1, 2, 0)
# 		return img.astype(np.uint8)


# 	def pad_bbox(self, bbox, labels, *, total_boxes=128):
# 		padded_bbox = np.full((total_boxes, 4), -1, dtype=bbox.dtype)
# 		padded_labels = np.full(total_boxes, -1, dtype=labels.dtype)
# 		padded_bbox[:len(bbox)] = bbox
# 		padded_labels[:len(labels)] = labels
# 		return padded_bbox, padded_labels

# 	def _scale(self, img, bbox, size):
# 		"""Resizing with keeping the aspect ratio"""
# 		_, *old_size = img.shape
# 		img = tr.scale(img, size, fit_short=True)
# 		_, *new_size = img.shape
# 		bbox = tr.resize_bbox(bbox, old_size, new_size)
# 		return img, bbox

# 	def _scale_down(self, img, bbox):
# 		""" scale down huge images """
# 		if max(img.shape) >= self._pre_rescale:
# 			img, bbox = self._scale(img, bbox, size=self._pre_rescale)
# 		return img, bbox

# 	def _center_crop(self, img, bbox, size):
# 		"""Center crop to square final size"""
# 		new_img, param = tr.center_crop(img,
# 			size,
# 			return_param=True)
# 		new_bbox = tr.crop_bbox(bbox,
# 			y_slice=param['y_slice'],
# 			x_slice=param['x_slice'],
# 			allow_outside_center=True)

# 		if not self.is_bbox_ok(new_bbox):
# 			_, *old_size = img.shape
# 			new_img = tr.resize(img, size)
# 			new_bbox = tr.resize_bbox(bbox, old_size, size)

# 		return new_img, new_bbox

# 	def _random_expand(self, img, bbox):
# 		# img, param = tr.random_expand(img,
# 		# 	max_ratio=2,
# 		# 	fill=self.mean,
# 		# 	return_param=True)

# 		# bbox = tr.translate_bbox(bbox,
# 		# 	y_offset=param['y_offset'],
# 		# 	x_offset=param['x_offset'])
# 		return img, bbox


# 	def _random_crop(self, img, bbox, size=None):

# 		if size is None:
# 			img, param = ssd_tr.random_crop_with_bbox_constraints(
# 				img=img, bbox=bbox,
# 				min_scale=0.3, max_scale=1,
# 				max_aspect_ratio=1,
# 				constraints=[(c,1) for c in reversed(np.linspace(0.1, 0.7, 7))],
# 				return_param=True)
# 		else:
# 			img, param = tr.random_crop(img, size, return_param=True)

# 		bbox = tr.crop_bbox(bbox,
# 			y_slice=param['y_slice'],
# 			x_slice=param['x_slice'],
# 			allow_outside_center=True)

# 		return img, bbox

# 	def _random_flip(self, img, bbox, size, **params):

# 		img, flip_params = tr.random_flip(img, return_param=True, **params)
# 		bbox = tr.flip_bbox(bbox, size, **flip_params)
# 		return img, bbox

# 	def transform(self, im_obj):

# 		img, bbox, lab = self.preprocess(im_obj)
# 		img, bbox = self.augment(img, bbox)
# 		img, bbox, lab = self.postprocess(img, bbox, lab)

# 		return img, bbox, lab

# 	def preprocess(self, im_obj):
# 		# we have here only one class so far
# 		img, lab = im_obj.im_array, np.array([0])
# 		img = img.transpose(2, 0, 1)

# 		multi_box = self.multi_box(im_obj.uuid, keys=["y0", "x0", "y1", "x1"])
# 		bbox = np.array(multi_box, dtype=np.int32)
# 		self._profile_img(img, "Init image")

# 		return img, bbox, lab

# 	@property
# 	def augmentations(self):
# 		return self._train_augs if chainer.config.train else self._val_augs

# 	def call_augmentation(self, img, bbox, aug_func, augments_bbox: bool = True, **params):

# 		if augments_bbox:
# 			return aug_func(img, bbox, **params)

# 		return aug_func(img, **params), bbox

# 	def augment(self, img, bbox, n_tries=5):
# 		bbox_ok = False
# 		# we need these checks, because random crop crops sometimes without the bbox
# 		for i in range(n_tries):
# 			new_img, new_bbox = self._augment(img, bbox)
# 			bbox_ok = self.is_bbox_ok(new_bbox)
# 			if bbox_ok:
# 				break

# 		if bbox_ok:
# 			return new_img, new_bbox

# 		# apply validation augmentations, they always work
# 		with chainer.using_config("train", False):
# 			return self._augment(img, bbox)

# 	def _augment(self, img, bbox):

# 		# fig, axs = plt.subplots(2)
# 		# axs[0].imshow(img.transpose(1,2,0).astype(np.uint8))
# 		# axs[0].axis("off")
# 		# axs[0].set_title("Original Image")
# 		# for y0, x0, y1, x1 in bbox:
# 		# 	axs[0].add_patch(Rectangle((x0,y0), x1-x0, y1-y0,
# 		# 		fill=False, linewidth=2))

# 		for aug, params in self.augmentations:
# 			img, bbox = self.call_augmentation(img, bbox, aug, **params)
# 			self._profile_img(img, aug.__name__)

# 		# axs[1].set_title("final image")
# 		# axs[1].axis("off")
# 		# axs[1].imshow(img.transpose(1,2,0).astype(np.uint8))
# 		# for (y0, x0, y1, x1) in bbox:
# 		# 	axs[1].add_patch(Rectangle((x0,y0), x1-x0, y1-y0,
# 		# 		fill=False, linewidth=2))
# 		# plt.tight_layout()
# 		# plt.show()
# 		# plt.close()

# 		return img, bbox

# 	def postprocess(self, img, bbox, lab):

# 		img = img - self.mean
# 		assert self.is_bbox_ok(bbox), \
# 			f"Ill-formed bounding box: {bbox}!"

# 		bbox, lab = self.pad_bbox(bbox, lab)

# 		return img, bbox, lab
