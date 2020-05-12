import chainer
import logging

from chainer.training.updaters import StandardUpdater
from chainer_addons.training import MiniBatchUpdater

from cvdatasets.utils import pretty_print_dict

from moth_detector.core import dataset
from moth_detector.core import finetuner
from moth_detector.core import trainer
from moth_detector.core import model

class Pipeline(object):

	def finetuner_setup(self, opts):

		self.tuner_cls, self.ft_kwargs, self.comm = finetuner.get_finetuner(opts)
		logging.info(f"Using {self.tuner_cls.__name__} with arguments: {pretty_print_dict(self.ft_kwargs)}")

	def updater_setup(self, opts):

		if opts.mode == "train" and opts.update_size > opts.batch_size:
			self.updater_cls = MiniBatchUpdater
			self.updater_kwargs = dict(update_size=opts.update_size)

		else:
			self.updater_cls = StandardUpdater
			self.updater_kwargs = dict()

	def debug_setup(self, opts):
		chainer.set_debug(opts.debug)
		if opts.debug:
			logging.warning("DEBUG MODE ENABLED!")

	def __init__(self, opts):
		super(Pipeline, self).__init__()

		self.debug_setup(opts)
		self.finetuner_setup(opts)
		self.updater_setup(opts)
		self.opts = opts

		self.tuner = self.tuner_cls(
			opts=opts,
			classifier_cls=model.Detector,
			classifier_kwargs={},

			model_kwargs=dict(
				n_fg_class=1,
            	pretrained_model='imagenet'
			),

			dataset_cls=dataset.BBoxDataset,
			dataset_kwargs_factory=None,

			updater_cls=self.updater_cls,
			updater_kwargs=self.updater_kwargs,

			**self.ft_kwargs
		)



	def __call__(self, experiment_name, *args, **kwargs):

		if self.opts.mode != "train":
			raise NotImplementedError(f"this mode is not implemented: {self.opts.mode}")

		return self.tuner.run(
			opts=self.opts,
			trainer_cls=trainer.SSDTrainer,
			sacred_params=dict(
				name=experiment_name,
				comm=self.comm,
				no_observer=self.opts.no_sacred
			)
		)
