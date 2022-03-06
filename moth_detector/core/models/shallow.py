import cv2
import numpy as np
import typing as T

from skimage import filters
from collections import namedtuple
from matplotlib import pyplot as plt

from blob_detector import utils
from blob_detector.core.pipeline import Pipeline
from blob_detector.core.bbox_proc import Splitter
from blob_detector.core.binarizers import BinarizerType

from moth_detector.core.models.base import BaseModel
from moth_detector.core.dataset.bbox_dataset import BBoxDataset


class Model(BaseModel):

	class meta:

		mean = 0
		input_size = 1024
		feature_size = -1

		def prepare_func(x, size=None, *args, **kwargs):
			raise RuntimeError("Should not be called!")

	RGB2GRAY = np.array([0.299, 0.587, 0.114]).reshape(-1, 1, 1)

	def __init__(self, input_size, *args,

				 # preprocessing
				 equalize: bool = False,
				 sigma: float = -5.0,

				 # binarization
				 thresholding: BinarizerType = BinarizerType.high_pass,
				 # binarization: Gauss-local binarization
				 block_size_scale: float = 0.2,
				 pad: bool = False,
				 # binarization: High-pass based
				 window_size: int = 30,
				 bin_sigma: float = 5.0,
				 # binarization: OTSU
				 use_cv2: bool = False,

				 remove_border: bool = True,

				 # postprocess
				 kernel_size: int = 5,
				 dilate_iterations: int = 3,

				 # bbox postprocess
				 enlarge: float = -0.01,
				 **kwargs):

		super().__init__(input_size=input_size)

		self._splitter = Splitter(preproc=Pipeline(), detector=Pipeline())

		self.img_proc = Pipeline()
		self.img_proc.add_operation(self._splitter.set_image)
		self.img_proc.preprocess(equalize=equalize, sigma=sigma)
		self.img_proc.binarize(type=thresholding,
			block_size_scale=block_size_scale,
			do_padding=pad,
			use_cv2=use_cv2,
			sigma=bin_sigma,
			window_size=window_size)
		if remove_border:
			self.img_proc.remove_border()
		self.img_proc.open_close(kernel_size=kernel_size, iterations=dilate_iterations)

		self.bbox_proc = Pipeline()
		self.bbox_proc.detect()
		self.bbox_proc.add_operation(self._splitter.split)
		self.bbox_proc.bbox_filter(enlarge=enlarge)

	def reinitialize_clf(self, n_classes, feat_size=None, initializer=None):
		pass

	def load(self, *args, **kwargs):
		pass

	def preprocess(self, x, **kwargs):
		im = x + BBoxDataset.mean
		# RGB -> Grayscale
		im = (im * Model.RGB2GRAY).sum(axis=0).astype(np.uint8)
		# h, w = im.shape

		return self.img_proc(im, **kwargs)

	def __call__(self, x):
		im0 = self.preprocess(x)
		bboxes, inds, _ = self.bbox_proc(im0)

		result = [[], [], []]

		for i in inds:
			(x0, y0), (x1, y1) = bboxes[i]

			result[0].append([y0, x0, y1, x1]) # bbox
			result[1].append(0) # label
			result[2].append(1.0) # score

		return list(map(np.array, result))
