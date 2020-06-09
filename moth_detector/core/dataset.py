import chainer
import copy
import numpy as np

from chainer_addons.utils import Size
from chainercv import transforms
from chainercv.links.model.ssd import transforms as ssd_transforms

from cvdatasets.dataset import AnnotationsReadMixin
from cvdatasets.dataset import MultiBoxMixin
from cvdatasets.dataset import IteratorMixin

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

class BBoxDataset(IteratorMixin, MultiBoxMixin, AnnotationsReadMixin):

	# ImageNet mean (we need this if we use InceptionV3 ???)
	mean = np.array((123, 117, 104), dtype=np.float32).reshape((-1, 1, 1))

	def __init__(self, *args, augment, preprocess, size, center_crop_on_val, **kwargs):
		super(BBoxDataset, self).__init__(*args, **kwargs)

		self.size = Size(size)
		self._augment = augment
		self._pre_rescale = 1000
		# self.preprocess = preprocess
		# self.center_crop_on_val = center_crop_on_val

	def is_bbox_ok(self, bbox) -> bool:
		return bbox.shape[0] >= 1 and bbox.shape[1] == 4

	def get_img_data(self, i):
		im_obj = super(BBoxDataset, self).get_example(i)
		# we have here only one class so far
		img, labels = im_obj.im_array, np.array([0])
		img = img.transpose(2, 0, 1)
		return img, labels

	def prepare_back(self, img):
		img = img + self.mean
		img = img.transpose(1, 2, 0)
		return img.astype(np.uint8)

	def get_example(self, i):
		img, labels = self.get_img_data(i)
		multi_box = self.multi_box(i, keys=["y0", "x0", "y1", "x1"])

		bbox = np.array(multi_box, dtype=np.int32)

		if self._augment and chainer.config.train:
			c = 0
			while True:
				new_img, new_bbox = self.augment(img, bbox)

				if self.is_bbox_ok(new_bbox) or c >= 5:
					break
				c += 1

			if self.is_bbox_ok(new_bbox):
				img, bbox = new_img, new_bbox
			else:
				img, bbox = self.val_augment(img, bbox)
		else:
			img, bbox = self.val_augment(img, bbox)

		img = img - self.mean
		assert self.is_bbox_ok(bbox), \
			f"Ill-formed bounding box: {bbox}!"

		bbox, labels = self.pad_bbox(bbox, labels)
		return img, bbox, labels

	def pad_bbox(self, bbox, labels, *, total_boxes=128):
		padded_bbox = np.full((total_boxes, 4), -1, dtype=bbox.dtype)
		padded_labels = np.full(total_boxes, -1, dtype=labels.dtype)
		padded_bbox[:len(bbox)] = bbox
		padded_labels[:len(labels)] = labels
		return padded_bbox, padded_labels

	def _scale(self, img, bbox, size):
		_, *old_size = img.shape
		img = transforms.scale(img, size, fit_short=True)
		_, *new_size = img.shape
		bbox = transforms.resize_bbox(bbox, old_size, new_size)
		return img, bbox

	def val_augment(self, img, bbox):
		# 1. Resizing with keeping the aspect ratio
		img, bbox = self._scale(img, bbox, min(self.size))

		# 2. Center crop to square final size
		final_size = tuple(self.size)
		new_img, param = transforms.center_crop(img,
			final_size,
			return_param=True)
		new_bbox = transforms.crop_bbox(bbox,
			y_slice=param['y_slice'],
			x_slice=param['x_slice'],
			allow_outside_center=True)

		if not self.is_bbox_ok(new_bbox):
			_, *old_size = img.shape
			new_img = transforms.resize(img, final_size)
			new_bbox = transforms.resize_bbox(bbox, old_size, final_size)

		return new_img, new_bbox


	def augment(self, img, bbox):
		# 0. scale down huge images
		if max(img.shape) >= self._pre_rescale:
			img, bbox = self._scale(img, bbox, size=self._pre_rescale)

		# fig, axs = plt.subplots(2)
		# axs[0].imshow(img.transpose(1,2,0).astype(np.uint8))
		# axs[0].set_title("Original Image")
		# for y0, x0, y1, x1 in bbox:
		# 	axs[0].add_patch(Rectangle((x0,y0), x1-x0, y1-y0,
		# 		fill=False, linewidth=2))

		# 1. Color augmentation
		img = ssd_transforms.random_distort(img)

		# 2. Random expansion
		if False:#np.random.randint(2):
			img, param = transforms.random_expand(
				img,
				max_ratio=2,
				fill=self.mean,
				return_param=True)

			bbox = transforms.translate_bbox(
				bbox,
				y_offset=param['y_offset'],
				x_offset=param['x_offset'])

		# 3. Random cropping
		img, param = ssd_transforms.random_crop_with_bbox_constraints(
			img=img, bbox=bbox,
			min_scale=0.3, max_scale=1,
			max_aspect_ratio=1,
			constraints=[(c,1) for c in reversed(np.linspace(0.1, 0.7, 7))],
			return_param=True)
		bbox = transforms.crop_bbox(bbox,
			y_slice=param['y_slice'],
			x_slice=param['x_slice'],
			allow_outside_center=True)

		# 4. Resizing with keeping the aspect ratio
		img, bbox = self._scale(img, bbox, min(self.size))

		# 5. Random crop to square final size
		final_size = tuple(self.size)
		img, param = transforms.random_crop(img,
			final_size,
			return_param=True)
		bbox = transforms.crop_bbox(bbox,
			y_slice=param['y_slice'],
			x_slice=param['x_slice'],
			allow_outside_center=True)

		# 6. Random horizontal flipping
		img, param = transforms.random_flip(
			img,
			x_random=True,
			return_param=True)
		bbox = transforms.flip_bbox(bbox,
			final_size,
			x_flip=param['x_flip'])

		# axs[1].set_title("final image")
		# axs[1].imshow(img.transpose(1,2,0).astype(np.uint8))
		# for (y0, x0, y1, x1) in bbox:
		# 	axs[1].add_patch(Rectangle((x0,y0), x1-x0, y1-y0,
		# 		fill=False, linewidth=2))
		# plt.show()
		# plt.close()

		return img, bbox
