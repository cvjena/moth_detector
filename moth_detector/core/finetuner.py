import abc
import logging

try:
	import chainermn
except Exception as e: #pragma: no cover
	_CHAINERMN_AVAILABLE = False #pragma: no cover
else:
	_CHAINERMN_AVAILABLE = True

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

	if opts.mode == "train" and opts.mpi:
		assert _CHAINERMN_AVAILABLE, "Distributed training is not possible!"

		logging.info("===== MPI enabled. Creating NCCL communicator ! =====")
		comm = chainermn.create_communicator("pure_nccl")
		logging.info(f"===== Rank: {comm.rank}, IntraRank: {comm.intra_rank}, InterRank: {comm.inter_rank} =====")

		tuner_cls = SSD_MPIFinetuner
		ft_kwargs = dict(comm=comm)

	else:
		tuner_cls = SSD_DefaultFinetuner
		ft_kwargs = dict()
		comm = None

	return tuner_cls, ft_kwargs, comm
