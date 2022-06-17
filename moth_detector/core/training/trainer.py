import numpy as np

from chainer.iterators import SerialIterator
from chainer.iterators import MultiprocessIterator

from chainer_addons.dataset.iterator import ProgressBarWrapper
from cvfinetune.training.trainer import base
from pathlib import Path
from tqdm import tqdm

from moth_detector.core.training.extensions import DetectionVisReport

def _new_iter(iterator):

	kwargs = dict(
		repeat=False,
		shuffle=False,
		dataset=iterator.dataset,
		batch_size=iterator.batch_size,
	)

	if isinstance(iterator, SerialIterator):
		cls = SerialIterator

	elif isinstance(iterator, MultiprocessIterator):
		cls = MultiprocessIterator
		kwargs.update(dict(
            n_processes=iterator.n_processes,
            n_prefetch=iterator.n_prefetch,
            shared_mem=iterator.shared_mem,
		))

	else:
		raise ValueError(f"Unsupported iterator class: {type(iterator).__name__}")

	return cls(**kwargs)

class DetectionTrainer(base.Trainer):

	def __init__(self, opts, intervals=base.default_intervals, *args, **kwargs):
		# intervals["eval"] = (2, "iteration")
		super(DetectionTrainer, self).__init__(opts=opts, intervals=intervals, *args, **kwargs)

		# n_batches = int(np.ceil(len(it.dataset) // it.batch_size))
		# self.init_vis_report(trigger=(5, "epoch"))

	def init_vis_report(self, trigger=(1, "epoch")):
		target = self.evaluator.get_target("main")
		val_it = _new_iter(self.evaluator.get_iterator("main"))
		train_it = _new_iter(self.updater.get_iterator("main"))

		detections_folder = Path(self.out) / "detections"
		detections_folder.mkdir(parents=True, exist_ok=True)

		self.train_detection_reporter = DetectionVisReport(
			iterator=ProgressBarWrapper(train_it, leave=False,
				desc="Qualitative Evaluation (train)"),
			target=target,
			filename="detections/train/iter{iteration:06d}/img{index:04d}.jpg"
		)

		self.detection_reporter = DetectionVisReport(
			iterator=ProgressBarWrapper(val_it, leave=False,
				desc="Qualitative Evaluation (val)"),
			target=target,
			filename="detections/val/iter{iteration:06d}/img{index:04d}.jpg"
		)

		self.extend(self.train_detection_reporter, trigger=trigger)
		self.extend(self.detection_reporter, trigger=trigger)


	def reportables(self, opts):
		print_values = [
			"elapsed_time",
			"epoch",

			"main/map@50", self.eval_name("main/map@50"),
			"main/map@75", self.eval_name("main/map@75"),
			"main/loss", self.eval_name("main/loss"),
			"main/loc_loss", self.eval_name("main/loc_loss"),
			"main/conf_loss", self.eval_name("main/conf_loss"),
		]

		plot_values = {
			"loss": [
				"main/loss", self.eval_name("main/loss"),
			],
			"loc_loss": [
				"main/loc_loss", self.eval_name("main/loc_loss"),
			],
			"conf_loss": [
				"main/conf_loss", self.eval_name("main/conf_loss"),
			],
		}
		return print_values, plot_values
