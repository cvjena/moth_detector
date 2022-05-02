import abc
import logging

try:
	import chainermn
except Exception as e: #pragma: no cover
	_CHAINERMN_AVAILABLE = False #pragma: no cover
else:
	_CHAINERMN_AVAILABLE = True

from cvfinetune import finetuner as ft

from chainer.training.updaters import StandardUpdater
from chainer_addons.training import MiniBatchUpdater
from chainercv.links.model.ssd import multibox_loss


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

def get_model_kwargs():

	return dict(
		n_fg_class=1,
		pretrained_model='imagenet'
	)

def get_detector(model_type: str):

	_detectors = {
		"chainercv.SSD300": detectors.SSD_Detector,
		"chainercv.FasterRCNNVGG16": detectors.FRCNN_Detector,
		"shallow": detectors.Shallow_Detector,
		"mcc": detectors.Shallow_Detector,
	}

	assert model_type in _detectors, \
		f"Detector type not found: {model_type}"

	cls = _detectors.get(model_type)

	return dict(
		classifier_cls=cls,
		classifier_kwargs={},

		model_kwargs=get_model_kwargs(),
	)

def get_model(model_type: str):

	_models = {
		"chainercv.SSD300": models.SSD_Model,
		"chainercv.FasterRCNNVGG16": models.FRCNN_Model,
		"shallow": models.Shallow_Model,
		"mcc": models.MCCModel,
	}

	assert model_type in _models, \
		f"Model type not found: {model_type}"

	return _models.get(model_type)

def new_finetuner(opts):

	mpi = opts.mode == "train" and opts.mpi

	tuner_factory = ft.FinetunerFactory.new(
		mpi=mpi,
		default=DefaultFinetuner,
		mpi_tuner=MPIFinetuner)

	tuner = tuner_factory(
		opts=opts,
		experiment_name="Moth detector",
		manual_gc=True,
		**get_detector(opts.model_type),
		**get_updater(opts),

		dataset_cls=dataset.BBoxDataset,
		dataset_kwargs_factory=dataset.BBoxDataset.kwargs(opts),

	)

	return tuner, tuner_factory.get("comm")

class _mixin(abc.ABC):


	def init_model(self):
		model_cls = get_model(self.model_type)

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

