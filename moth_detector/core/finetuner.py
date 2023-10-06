import abc

from cvfinetune import finetuner as ft

from chainercv.links.model.ssd import multibox_loss

from moth_detector.core import dataset
from moth_detector.core import detectors
from moth_detector.core import models
from moth_detector.core import training


def new_finetuner(opts):

	mpi = opts.mode == "train" and opts.mpi

	tuner_factory = ft.FinetunerFactory(
		mpi=mpi,
		default=DefaultFinetuner,
		mpi_tuner=MPIFinetuner)

	tuner = tuner_factory(
		opts=opts,
		experiment_name="Moth detector",
		manual_gc=True,
		**detectors.kwargs(opts),
		**training.updater_kwargs(opts),
		**dataset.kwargs(opts),
	)

	return tuner, tuner_factory.get("comm")

class _mixin(abc.ABC):

	def init_model(self):
		model_cls = models.get(self.model_type)

		self.model = model_cls(
			input_size=self.input_size,
			**self.model_kwargs
		)

	@property
	def loss_func(self):
		return multibox_loss

	def load_weights(self):
		if self.model_type in ["shallow", "mcc"]:
			return
		super().load_weights()


class DefaultFinetuner(_mixin, ft.DefaultFinetuner):
	pass

class MPIFinetuner(_mixin, ft.MPIFinetuner):
	pass

