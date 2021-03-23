import numpy as np

from chainer_addons.dataset.iterator import ProgressBarWrapper
from cvfinetune.training.trainer import SacredTrainer
from cvfinetune.training.trainer.base import default_intervals
from pathlib import Path
from tqdm import tqdm

from moth_detector.core.training.extensions import DetectionVisReport

class DetectionTrainer(SacredTrainer):

	def __init__(self, opts, intervals=default_intervals, *args, **kwargs):
		# intervals["eval"] = (2, "iteration")
		super(DetectionTrainer, self).__init__(opts=opts, intervals=intervals, *args, **kwargs)

		it = self.evaluator.get_iterator("main")
		# n_batches = int(np.ceil(len(it.dataset) // it.batch_size))

		target = self.evaluator.get_target("main")

		detections_folder = Path(self.out) / "detections"
		detections_folder.mkdir(parents=True, exist_ok=True)

		self.detection_reporter = DetectionVisReport(
			iterator=ProgressBarWrapper(it, leave=False,
				desc="Qualitative Evaluation"),
			target=target,
			filename="detections/iter{iteration:06d}_img{index:04d}.jpg"
		)

		#self.extend(self.detection_reporter,
		#	trigger=(5, "epoch")
		#)

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
