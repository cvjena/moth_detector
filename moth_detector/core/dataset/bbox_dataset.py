import chainer
import copy
import logging
import matplotlib.pyplot as plt
import numpy as np

from chainercv import transforms as tr
from chainercv.links.model.ssd import transforms as ssd_tr
from matplotlib.patches import Rectangle

from cvdatasets import dataset as ds

from moth_detector.core.dataset.augmentations import Augmentations
from moth_detector.core.dataset.augmentations import is_bbox_ok

class BBoxDataset(
	ds.ImageProfilerMixin,
	ds.TransformMixin,
	ds.IteratorMixin,
	ds.MultiBoxMixin,
	ds.AnnotationsReadMixin):

	# ImageNet mean (we need this if we use InceptionV3 ???)
	mean = np.array((123, 117, 104), dtype=np.float32).reshape((-1, 1, 1))


	@classmethod
	def kwargs(cls, opts):

		def inner(subset: str) -> dict:
			return dict(opts=opts)

		return inner

	def __init__(self, *args, opts, prepare, center_crop_on_val, **kwargs):
		super(BBoxDataset, self).__init__(*args, **kwargs)

		self._setup_augmentations(opts)

		self.return_scale = opts.model_type == "frcnn"
		self.max_boxes = opts.max_boxes
		self.area_threshold = opts.area_threshold

		if len(self.uuids) == 0:
			return

		counts = []
		for uuid in self.uuids:
			objects = self._annot.multi_boxes[uuid]["objects"]
			counts.append(len(objects))

		n, avg, std = sum(counts), np.mean(counts), np.std(counts)
		logging.info(f"Loaded {n} bounding boxes ({avg:.2f} +/- {std:.2f} | {min(counts)} - {max(counts)} per image )")

	def get_im_obj(self, i):
		return ds.AnnotationsReadMixin.get_example(self, i)

	def _setup_augmentations(self, opts):
		Aug = Augmentations

		rescale_size_tr = max(1000, min(self.size))
		rescale_size_val = max(1000, min(self._size))

		self._train_augs = [
			Aug.scale_down(size=rescale_size_tr),
			Aug.random_distort(),
			Aug.random_expand(),
			Aug.random_crop(),

			Aug.resize(size=tuple(self.size)),
			Aug.random_crop(size=tuple(self._size)),
			Aug.random_flip(size=tuple(self._size),
				x_random=True, y_random=True),
		]

		self._val_augs = [
			Aug.scale_down(size=rescale_size_val),
			Aug.resize(size=tuple(self._size)),
			# Aug.center_crop(size=tuple(self._size)),

		]

	def prepare_back(self, img):
		img = img + self.mean
		img = img.transpose(1, 2, 0)
		return img.astype(np.uint8)


	def transform(self, im_obj):

		img, bbox = self.preprocess(im_obj)
		img, bbox = self.augment(img, bbox)
		img, bbox, lab = self.postprocess(img, bbox)

		if self.return_scale:
			OW, OH = im_obj.im.size
			_, h, w = img.shape
			scale = h / OH
			return img, bbox, lab, np.float32(scale)
		else:
			return img, bbox, lab

	def preprocess(self, im_obj):
		img = im_obj.im_array.transpose(2, 0, 1)
		self._profile_img(img, "Init image")

		multi_box = self.multi_box(im_obj.uuid, keys=["y0", "x0", "y1", "x1"])

		bbox = np.array(multi_box, dtype=np.int32)

		return img, bbox

	@property
	def augmentations(self):
		return self._train_augs if chainer.config.train else self._val_augs

	def augment(self, img, bbox, n_tries=5):

		# we need these checks, because random crop crops sometimes without the bbox
		for i in range(n_tries):
			new_img, new_bbox = self._augment(img, bbox)
			if is_bbox_ok(new_bbox):
				return new_img, new_bbox

		# apply validation augmentations, they always work
		with chainer.using_config("train", False):
			return self._augment(img, bbox)

	def _augment(self, img, bbox, show_aug=False):

		orig = img, bbox

		for aug in self.augmentations:
			img, bbox = aug(img, bbox)
			self._profile_img(img, aug.__name__)

		if not show_aug:
			return img, bbox


		def plot(img, bbox, ax=None, title=None):
			ax = ax or plt.gca()
			if title is not None:
				ax.set_title(title)
			ax.axis("off")
			ax.imshow(img.transpose(1,2,0).astype(np.uint8))
			for y0, x0, y1, x1 in bbox:
				ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0,
					fill=False, linewidth=2))

		fig, axs = plt.subplots(2)

		plot(*orig, ax=axs[0], title="Original Image")
		plot(img, bbox, ax=axs[1], title="Augmented Image")

		plt.tight_layout()
		plt.show()
		plt.close()

		return img, bbox

	def postprocess(self, img, bbox):

		img = img - self.mean
		assert is_bbox_ok(bbox), \
			f"Ill-formed bounding box: {bbox}!"

		bbox, lab = self.pad_bbox(bbox)
		self._profile_img(img, "Final image")

		return img, bbox.astype(np.float32), lab

	def pad_bbox(self, bbox, *, total_boxes=None, box_dtype=chainer.config.dtype, lab_dtype=np.int32):
		total_boxes = total_boxes or self.max_boxes
		padded_bbox = np.full((total_boxes, 4), -1, dtype=box_dtype)
		padded_labels = np.full(total_boxes, -1, dtype=lab_dtype)

		padded_bbox[:len(bbox)] = bbox
		# we have here only one class so far
		padded_labels[:len(bbox)] = [(0 if self.check(*box) else -1) for box in bbox]

		return padded_bbox, padded_labels


	def check(self, y0, x0, y1, x1):
		w, h = x1-x0, y1-y0

		w_ratio, h_ratio = self._size / (w, h)

		area =  1/w_ratio * 1/h_ratio

		return area >= self.area_threshold
