import cv2
import numpy as np
import simplejson as json
import typing as T

from collections import namedtuple
from matplotlib import pyplot as plt
from skimage import filters

from idac.blobdet.blob_detector_factory import BlobDetectorFactory

from blob_detector import utils
from blob_detector.core.bbox_proc import Splitter
from blob_detector.core.binarizers import BinarizerType
from blob_detector.core.pipeline import Pipeline

from moth_detector.core.dataset.bbox_dataset import BBoxDataset
from moth_detector.core.models.base import BaseModel

class BaseShallowModel(BaseModel):

	class meta:

		mean = 0
		input_size = 1024
		feature_size = -1

		def prepare_func(x, size=None, *args, **kwargs):
			raise RuntimeError("Should not be called!")

	RGB2GRAY = np.array([0.299, 0.587, 0.114]).reshape(-1, 1, 1)

	def __init__(self, input_size, *args, **kwargs):
		super().__init__(input_size=input_size)

	def reinitialize_clf(self, n_classes, feat_size=None, initializer=None):
		pass

	def load(self, *args, **kwargs):
		pass

	def preprocess(self, x, **kwargs):

		im = x + BBoxDataset.mean
		# RGB -> Grayscale
		im = (im * BaseShallowModel.RGB2GRAY).sum(axis=0).astype(np.uint8)
		return im

	def convert_boxes(self, bboxes):

		result = [[], [], []]

		for (x0, y0), (x1, y1) in bboxes:

			result[0].append([y0, x0, y1, x1]) # bbox
			result[1].append(0) # label
			result[2].append(1.0) # score

		return list(map(np.array, result))


class Model(BaseShallowModel):

	def __init__(self, *args,

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
		super().__init__(*args, **kwargs)


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


	def preprocess(self, x, **kwargs):
		im = super().preprocess(x, **kwargs)
		# h, w = im.shape

		return self.img_proc(im, **kwargs)

	def __call__(self, x):
		im0 = self.preprocess(x)
		bboxes, inds, _ = self.bbox_proc(im0)

		return self.convert_boxes([bboxes[i] for i in inds])

class MCCModel(BaseShallowModel):

	config = None

	def __init__(self, *args,

		detector_type="adaptive",

		**kwargs):
		super().__init__(*args, **kwargs)

		assert self.config is not None, \
			"Setup of MCC Model has failed: config file was not set!"

		with open(self.config) as f:
			config = json.load(f)

		bg_img = (self.config.parent / config["blobdetector"]["backgroundpath"]).resolve()

		assert bg_img.exists(), \
			f"Background image was not found: {bg_img}!"

		config["blobdetector"]["backgroundpath"] = str(bg_img)
		self.bl_det = BlobDetectorFactory.get_blob_detector(detector_type, config)


	def preprocess(self, x, **kwargs):
		return (x + BBoxDataset.mean).astype(np.uint8).transpose(1,2,0)


	def __call__(self, x):
		im0 = self.preprocess(x)
		image_new, count, oois, id, binary = self.bl_det.findboxes(im0, 0)

		# ooi -> (x0, y0), (x1, y1)
		ooi2bbox = lambda ooi: [(ooi.x, ooi.y), (ooi.x + ooi.w, ooi.y + ooi.h)]

		return self.convert_boxes([ooi2bbox(ooi) for ooi in oois])
