import numpy as np

# from chainercv.links.model.ssd import SSD300 as SSD
# from chainercv.links.model.ssd import VGG16Extractor300 as Extractor
from chainercv.links.model.ssd import SSD512 as SSD
from chainercv.links.model.ssd import VGG16Extractor512 as Extractor

from moth_detector.core.models.base import BaseModel

class Model(BaseModel, SSD):

	class meta:
		mean = np.array((123, 117, 104)).reshape((-1, 1, 1))
		input_size = Extractor.insize
		feature_size = 4096

		def prepare_func(x, size=None, *args, **kwargs):
			import pdb; pdb.set_trace()
