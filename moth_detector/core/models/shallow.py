import numpy as np
import simplejson as json
import typing as T
import logging

try:
	from idac.blobdet.blob_detector_factory import BlobDetectorFactory
	MCC_AVAILABLE = True
except ImportError:
	MCC_AVAILABLE = False

from blob_detector import utils
from blob_detector.core.bbox import BBox
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
		super().__init__(*args, input_size=input_size, **kwargs)

	def reinitialize_clf(self, n_classes, feat_size=None, initializer=None):
		pass

	def load(self, *args, **kwargs):
		pass

	def preprocess(self, x, **kwargs):

		im = x + BBoxDataset.mean
		# RGB -> Grayscale
		im = (im * BaseShallowModel.RGB2GRAY).sum(axis=0).astype(np.uint8)
		return im

	def convert_boxes(self, bboxes: T.List[BBox], labels = None, scores = None):

		result = [[], [], []]

		if labels is None:
			labels = np.full(len(bboxes), 0, dtype=np.float32)

		if scores is None:
			scores = np.full(len(bboxes), -1, dtype=np.float32)

		for (bbox, label, score) in zip(bboxes, labels, scores):

			(x0, y0, x1, y1) = bbox
			result[0].append([y0, x0, y1, x1])
			result[1].append(label)
			result[2].append(score)

		return list(map(np.array, result))


class Model(BaseShallowModel):

	def __init__(self, *args,

				 min_size: int = 1080,

				 # preprocessing
				 equalize: bool = False,
				 sigma: float = 5.0,

				 # binarization
				 thresholding: BinarizerType = BinarizerType.gauss_local,

				 # binarization: base options
				 use_masked: bool = True,
				 use_cv2: bool = True,

				 # binarization: base local-based options
				 window_size: int = 31,
				 offset: float = 2.0,

				 # binarization: High-pass based
				 bin_sigma: float = 5.0,

				 remove_border: bool = True,

				 # postprocess
				 kernel_size: int = 5,
				 dilate_iterations: int = 2,

				 # bbox postprocess
				 enlarge: float = 0.01,
				 **kwargs):
		super().__init__(*args, **kwargs)

		BBox.MIN_AREA = 4e-4


		self.img_proc = Pipeline()
		self.img_proc.find_border()

		if min_size > 0:
			self.img_proc.rescale(min_size=min_size, min_scale=0.1)

		self.img_proc.preprocess(equalize=equalize, sigma=sigma)
		self.img_proc.binarize(type=thresholding,
			use_masked=use_masked,
			use_cv2=use_cv2,
			sigma=bin_sigma,
			window_size=window_size,
			offset=offset,
		)
		if remove_border:
			self.img_proc.remove_border()
		self.img_proc.open_close(
			kernel_size=kernel_size,
			iterations=dilate_iterations)

		self.bbox_proc = Pipeline()

		self.bbox_proc.detect(use_masked=True)

		_, splitter = self.bbox_proc.split_bboxes(
			preproc=Pipeline(), detector=Pipeline())

		_, bbox_filter = self.bbox_proc.bbox_filter(
			score_threshold=0.5,
			nms_threshold=0.3,
			enlarge=enlarge,
		)
		_, scorer = self.bbox_proc.score()

		self.img_proc.requires_input(splitter.set_image)
		self.img_proc.requires_input(bbox_filter.set_image)
		self.img_proc.requires_input(scorer.set_image)

	def __call__(self, x, show_intermediate: bool = False):
		im = self.preprocess(x)
		res = self.img_proc(im)

		if show_intermediate:
			utils.show_intermediate(res)

		det = self.bbox_proc(res)

		if show_intermediate:
			utils.show_intermediate(det)

		h, w, *_ = im.shape
		bboxes = [bbox * (w, h) for bbox in det.bboxes if bbox.is_valid]
		scores = [bbox.score for bbox in det.bboxes if bbox.is_valid]
		return self.convert_boxes(bboxes, scores=scores)

class MCCModel(BaseShallowModel):

	config = None

	def __init__(self, *args,

		detector_type="adaptive",

		**kwargs):
		global MCC_AVAILABLE
		super().__init__(*args, **kwargs)

		assert MCC_AVAILABLE, "MCC Code could not be found!"

		assert self.config is not None, \
			"Setup of MCC Model has failed: config file was not set!"

		with open(self.config) as f:
			config = json.load(f)

		bg_img = (self.config.parent / config["blobdetector"]["backgroundpath"]).resolve()

		assert bg_img.exists(), \
			f"Background image was not found: {bg_img}!"

		logging.info(f"Using Background image from \"{bg_img}\"")
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
