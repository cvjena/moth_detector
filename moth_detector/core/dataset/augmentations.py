import enum
import numpy as np

from functools import partial
from functools import wraps
from typing import Tuple

from chainercv import transforms as tr
from chainercv.links.model.ssd import transforms as ssd_tr

class _wrapper(object):

	def __init__(self, augmentation, *args,
		augments_bbox: bool = True, **kwargs):

		self.augmentation = augmentation
		self.args = args
		self.kwargs = kwargs

		self._augments_bbox = augments_bbox
		self.__name__ = augmentation.name

	def __call__(self, img, bbox):
		func = self.augmentation.func
		try:
			if self._augments_bbox:
				return func(img, bbox, *self.args, **self.kwargs)

			return func(img, *self.args, **self.kwargs), bbox
		except:
			import pdb; pdb.set_trace()
			raise


	def __repr__(self):
		prefix = "Image and bbox" if self._augments_bbox else "Image"
		return f"{prefix} augmentation: \"{self.augmentation.name}\" with args {self.args} and kwargs {self.kwargs}"


def check_bbox(func):

	@wraps(func)
	def inner(img, bbox, size=None, *args, **kwargs):
		new_img, new_bbox = func(img, bbox, size, *args, **kwargs)

		if size is not None and not is_bbox_ok(new_bbox):
			_, *old_size = img.shape
			new_img = tr.resize(img, size)
			new_bbox = tr.resize_bbox(bbox, old_size, size)
		return new_img, new_bbox

	return inner

def _scale(img, bbox, size: int, upsample=True):
	"""
		Resizing with keeping the aspect ratio.
		If downsample is False, resize only images,
		that are greater than the given size.
	"""

	if max(img.shape) < size and not upsample:
		return img, bbox

	_, *old_size = img.shape
	img = tr.scale(img, size, fit_short=True)
	_, *new_size = img.shape
	bbox = tr.resize_bbox(bbox, old_size, new_size)
	return img, bbox

@check_bbox
def _center_crop(img, bbox, size: Tuple[int, int]):
	"""
		Center crop to given size.

		If the resulting bbox is outside of the crop,
		then resize the image without aspect ration
		preservation (ensured by @check_bbox).
	"""

	img, param = tr.center_crop(img, size, return_param=True)
	bbox = tr.crop_bbox(bbox,
		y_slice=param['y_slice'],
		x_slice=param['x_slice'],
		allow_outside_center=True)

	return img, bbox

@check_bbox
def _random_crop(img, bbox, size: Tuple[int, int] = None):

	"""
		Random crop to given size.

		If size is None, random crop to a random size
		between 30% and 100%.

		If the resulting bbox is outside of the crop,
		then resize the image without aspect ration
		preservation (ensured by @check_bbox).
	"""
	if size is None:
		img, param = ssd_tr.random_crop_with_bbox_constraints(
			img=img, bbox=bbox,
			min_scale=0.3, max_scale=1,
			max_aspect_ratio=1,
			constraints=[(c,1) for c in reversed(np.linspace(0.1, 0.7, 7))],
			return_param=True)
	else:
		img, param = tr.random_crop(img, size, return_param=True)

	bbox = tr.crop_bbox(bbox,
		y_slice=param['y_slice'],
		x_slice=param['x_slice'],
		allow_outside_center=True)

	return img, bbox

def _random_flip(img, bbox, size: Tuple[int, int], **params):
	img, flip_params = tr.random_flip(img, return_param=True, **params)
	bbox = tr.flip_bbox(bbox, size, **flip_params)
	return img, bbox

def _random_expand(img, bbox):
	return img, bbox

def is_bbox_ok(bbox) -> bool:
	return bbox.shape[0] >= 1 and bbox.shape[1] == 4

class Augmentations(enum.Enum):

	center_crop 		= (_center_crop, )
	scale 				= (_scale, )
	scale_down 			= (_scale, True, dict(upsample=False))
	random_crop 		= (_random_crop, )
	random_flip 		= (_random_flip, )
	random_distort 		= (ssd_tr.random_distort, False)
	random_expand 		= (_random_expand, )

	def __init__(self, func, augments_bbox=True, kwargs=dict()):
		self.func = func
		self.kwargs = kwargs
		self.augments_bbox = augments_bbox

	def __repr__(self):
		return f"<Augmentation \"{self.name}\">"

	def __call__(self, *args, **kwargs):
		kwargs = dict(self.kwargs, **kwargs)
		return _wrapper(self, augments_bbox=self.augments_bbox, *args, **kwargs)



if __name__ == '__main__':
	print(list(Augmentations))
	print(Augmentations.scale_down(size=300)(None, None))

