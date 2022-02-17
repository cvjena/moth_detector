import cv2
import numpy as np
import typing as T

from skimage import filters
from collections import namedtuple
from matplotlib import pyplot as plt

from moth_detector.core.models.base import BaseModel
from moth_detector.core.dataset.bbox_dataset import BBoxDataset


class BBox(namedtuple("BBox", "x0 y0 x1 y1")):
	__slots__ = ()

	@property
	def w(self):
		return abs(self.x1 - self.x0)

	@property
	def h(self):
		return abs(self.y1 - self.y0)


	@property
	def area(self):
		return self.h * self.w

	@property
	def ratio(self):
		return min(self.h, self.w) / max(self.h, self.w)

	def crop(self, im: np.ndarray, enlarge: bool = True):

		x0, y0, x1, y1 = self
		H, W, *_ = im.shape

		# translate from relative coordinates to pixel
		# coordinates for the given image

		x0, x1 = int(x0 * W), int(x1 * W)
		y0, y1 = int(y0 * H), int(y1 * H)

		# enlarge to a square extent
		if enlarge:
			h, w = int(self.h * H), int(self.w * W)
			size = max(h, w)
			dw, dh = (size - w) / 2, (size - h) / 2
			x0, y0 = max(int(x0 - dw), 0), max(int(y0 - dh), 0)
			x1, y1 = int(x0 + size), int(y0 + size)


		if im.ndim == 2:
			return im[y0:y1, x0:x1]

		elif im.ndim == 3:
			return im[y0:y1, x0:x1, :]

		else:
			ValueError(f"Unsupported ndims: {im.ndims=}")



class Model(BaseModel):

	class meta:

		mean = 0
		input_size = 1024
		feature_size = -1

		def prepare_func(x, size=None, *args, **kwargs):
			raise RuntimeError("Should not be called!")

	RGB2GRAY = np.array([0.299, 0.587, 0.114]).reshape(-1, 1, 1)

	def __init__(self, input_size, *args,
				 sigma: float = 5.0,
				 block_size_scale: float = 0.5,
				 dilate_iterations: int = 3,
				 kernel_size: int = 5,
				 **kwargs):

		super().__init__(input_size=input_size)


		self.sigma = sigma
		self.block_size_scale = block_size_scale
		self.dilate_iterations = dilate_iterations
		self.kernel_size = kernel_size

	def reinitialize_clf(self, n_classes, feat_size=None, initializer=None):
		pass

	def load(self, *args, **kwargs):
		pass

	def __call__(self, x):
		im = x + BBoxDataset.mean
		# RGB -> Grayscale
		im = (im * Model.RGB2GRAY).sum(axis=0)

		im0 = self.preprocess(im)
		im1 = self.threshold(im0)
		im2 = self.postprocess(im1)

		# images = [
		# fig, axs = plt.subplots(2,2, squeeze=False)
		# 	(im, "Original", plt.cm.gray),
		# 	(im0, "Pre-processed", plt.cm.gray),
		# 	(im1, "After threshold", plt.cm.jet),
		# 	(im2, "Post-processed", plt.cm.jet),
		# ]

		# for i, (_im, title, cmap) in enumerate(images):
		# 	ax = axs[np.unravel_index(i, axs.shape)]
		# 	ax.imshow(_im, cmap=cmap)
		# 	ax.set_title(title)
		# 	ax.axis("off")

		bboxes = self.detect(im2)

		return self.postprocess_boxes(im, bboxes)


	def preprocess(self, im: np.ndarray) -> np.ndarray:
		res = filters.gaussian(im, sigma=self.sigma, preserve_range=True)
		return res.astype(im.dtype)

	def threshold(self, im: np.ndarray, max_value = 255) -> np.ndarray:

		block_size = min(im.shape) * self.block_size_scale // 2 * 2 + 1

		thresh = filters.threshold_local(im,
			block_size=block_size,
			mode="constant",
		)

		bin_im = ((im > thresh) * max_value).astype(np.uint8)
		return max_value - bin_im


	def detect(self, im: np.ndarray) -> T.List[BBox]:

		contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)

		return [_contour2bbox(c, im.shape) for c in contours]


	def postprocess(self, im: np.ndarray) -> np.ndarray:
		kernel_size = self.kernel_size
		iterations = self.dilate_iterations
		kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

		im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
		im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

		if iterations >= 1:
			im = cv2.erode(im, kernel, iterations=iterations)
			im = cv2.dilate(im, kernel, iterations=iterations)

		return im

	def postprocess_boxes(self, im: np.ndarray, bboxes: T.List[BBox]):
		h, w = im.shape
		_im = im.astype(np.float64) / 255.
		integral, integral_sq = cv2.integral2(_im)
		del _im, im

		inds = cv2.dnn.NMSBoxes([[x0, y0, x1-x0, y1-y0] for (x0, y0, x1, y1) in bboxes],
							np.ones(len(bboxes), dtype=np.float32),
							score_threshold=0.99,
							nms_threshold=0.1,
						   )

		result = [[], [], []]

		for i in inds.squeeze():
			bbox = bboxes[i]

			mean, std, n = _im_mean_std(integral, integral_sq, bbox)
			area, ratio = bbox.area, bbox.ratio
			if not is_selected(mean, std, ratio, area):
				continue

			result[0].append([bbox.y0*h, bbox.x0*w, bbox.y1*h, bbox.x1*w]) # bbox
			result[1].append(0) # label
			result[2].append(1.0) # score

		return list(map(np.array, result))

def _contour2bbox(contour: np.ndarray, shape: T.Tuple[int, int], relative: bool = True) -> BBox:
	""" Gets the maximal extent of a contour and translates it to a bounding box. """
	x0, y0 = contour.min(axis=0)[0].astype(np.int32)
	x1, y1 = contour.max(axis=0)[0].astype(np.int32)

	if not relative:
		return BBox(x0, y0, x1, y1)

	h, w = shape
	return BBox(x0/w, y0/h, x1/w, y1/h)


def is_selected(mean: float, std: float, ratio: float, area: float) -> bool:
	# Caution, here are some magic numbers!
	return \
		std >= 5e-2 and \
		ratio >= .25 and \
		4e-4 <= area <= 4e-1


def _im_mean_std(integral: np.ndarray,
				 integral_sq: np.ndarray,
				 bbox: T.Optional[BBox] = None
				) -> T.Tuple[float, float, int]:

	h, w = integral.shape[0] - 1, integral.shape[1] - 1

	if bbox is None:
		arr_sum = integral[-1, -1]
		arr_sum_sq = integral_sq[-1, -1]
		N = h * w

	else:
		x0, y0, x1, y1 = bbox
		x0, x1 = int(x0 * w), int(x1 * w)
		y0, y1 = int(y0 * h), int(y1 * h)

		A, B, C, D = (y0,x0), (y1,x0), (y0,x1), (y1,x1)
		arr_sum = integral[D] + integral[A] - integral[B] - integral[C]
		arr_sum_sq = integral_sq[D] + integral_sq[A] - integral_sq[B] - integral_sq[C]

		N = (x1-x0) * (y1-y0)

	if N != 0:
		arr_mean = arr_sum / N
		arr_std  = np.sqrt((arr_sum_sq - (arr_sum**2) / N) / N)
	else:
		breakpoint()

	return arr_mean, arr_std, N
