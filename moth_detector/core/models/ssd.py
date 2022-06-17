import os
import numpy as np
import logging

# kind of hacky, but should work
if os.environ.get("BIG") == "1":
	from chainercv.links.model.ssd import SSD512 as SSD
	from chainercv.links.model.ssd import VGG16Extractor512 as Extractor
else:
	from chainercv.links.model.ssd import SSD300 as SSD
	from chainercv.links.model.ssd import VGG16Extractor300 as Extractor

#print(f"Using {SSD.__name__} as SSD-variant")


from moth_detector.core.models.base import BaseModel

class Model(BaseModel, SSD):

	class meta:
		mean = np.array((123, 117, 104)).reshape((-1, 1, 1))
		input_size = Extractor.insize
		feature_size = 4096

		def prepare_func(x, size=None, *args, **kwargs):
			import pdb; pdb.set_trace()
