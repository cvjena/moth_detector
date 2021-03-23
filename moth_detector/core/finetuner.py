import abc
import logging

try:
	import chainermn
except Exception as e: #pragma: no cover
	_CHAINERMN_AVAILABLE = False #pragma: no cover
else:
	_CHAINERMN_AVAILABLE = True

from cvfinetune.finetuner import DefaultFinetuner
from cvfinetune.finetuner import FinetunerFactory
from cvfinetune.finetuner import MPIFinetuner

from chainer.training.updaters import StandardUpdater
from chainer_addons.training import MiniBatchUpdater

from moth_detector.core import dataset
from moth_detector.core import detectors
from moth_detector.core import models

def get_updater(opts):
	if opts.mode == "train" and opts.update_size > opts.batch_size:
		cls = MiniBatchUpdater
		kwargs = dict(update_size=opts.update_size)

	else:
		cls = StandardUpdater
		kwargs = dict()

	return dict(updater_cls=cls, updater_kwargs=kwargs)

def get_model_kwargs(opts):

	return dict(
		n_fg_class=1,
		pretrained_model='imagenet'
	)

def get_detector(opts):
	return dict(
		classifier_cls=detectors.SSD_Detector,
		classifier_kwargs={},
	)

def new_finetuner(opts):

	opts.mpi = opts.mode == "train" and opts.mpi

	tuner_factory = FinetunerFactory.new(opts,
		default=SSD_DefaultFinetuner,
		mpi_tuner=SSD_MPIFinetuner)

	tuner = tuner_factory(

		opts=opts,
		model_kwargs=get_model_kwargs(opts),

		**get_detector(opts),
		**get_updater(opts),

		dataset_cls=dataset.BBoxDataset,
		dataset_kwargs_factory=dataset.BBoxDataset.kwargs,

	)

	return tuner, tuner_factory.get("comm")

class ssd_mixin(abc.ABC):

	# def init_evaluator(self, default_name="val"):
	# 	pass

	def init_model(self, opts):
		model_cls = models.SSD_Model

		self.model = model_cls(
			input_size=opts.input_size,
			**self.model_kwargs
		)

	# def new_dataset(self, opts, *args, **kwargs):
	# 	ds = super(ssd_mixin, self).new_dataset(opts, *args, **kwargs)
	# 	ds.setup_coder(self.model.coder)
	# 	return ds


class SSD_DefaultFinetuner(ssd_mixin, DefaultFinetuner):
	pass

class SSD_MPIFinetuner(ssd_mixin, MPIFinetuner):
	pass

