import abc
import logging

try:
	import chainermn
except Exception as e: #pragma: no cover
	_CHAINERMN_AVAILABLE = False #pragma: no cover
else:
	_CHAINERMN_AVAILABLE = True

from cvfinetune.finetuner import FinetunerFactory
from cvfinetune.finetuner import DefaultFinetuner
from cvfinetune.finetuner import MPIFinetuner

from moth_detector.core import model

class ssd_mixin(abc.ABC):

	# def init_evaluator(self, default_name="val"):
	# 	pass

	def init_model(self, opts):
		self.model = model.Model(
			input_size=opts.input_size,
			**self.model_kwargs)

	# def new_dataset(self, opts, *args, **kwargs):
	# 	ds = super(ssd_mixin, self).new_dataset(opts, *args, **kwargs)
	# 	ds.setup_coder(self.model.coder)
	# 	return ds


class SSD_DefaultFinetuner(ssd_mixin, DefaultFinetuner):
	pass

class SSD_MPIFinetuner(ssd_mixin, MPIFinetuner):
	pass

def get_finetuner(opts):

	opts.mpi = opts.mode == "train" and opts.mpi

	return FinetunerFactory.new(opts,
		default=SSD_DefaultFinetuner,
		mpi_tuner=SSD_MPIFinetuner)
