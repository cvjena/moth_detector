import numpy as np

from chainercv.links.model.faster_rcnn import FasterRCNNVGG16

from moth_detector.core.models.base import BaseModel

class Model(BaseModel, FasterRCNNVGG16):

	class meta:
		mean = np.array([122.7717, 115.9465, 102.9801]).reshape((-1, 1, 1))
		input_size = 300
		feature_size = 4096

		def prepare_func(x, size=None, *args, **kwargs):
			import pdb; pdb.set_trace()

