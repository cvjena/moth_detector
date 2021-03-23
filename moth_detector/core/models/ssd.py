import numpy as np

from chainercv.links.model.ssd import SSD300

from moth_detector.core.models.base import BaseModel

class Model(BaseModel, SSD300):

	class meta:
		mean = np.array((123, 117, 104)).reshape((-1, 1, 1))
		input_size = 300
		feature_size = 4096

		def prepare_func(x, size=None, *args, **kwargs):
			import pdb; pdb.set_trace()
