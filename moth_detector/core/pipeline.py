import chainer
import logging
import matplotlib.pyplot as plt
import numpy as np

from chainer.backends.cuda import to_cpu
from chainer.dataset import convert
from chainer.training.updaters import StandardUpdater
from chainer_addons.training import MiniBatchUpdater
from chainercv.evaluations import eval_detection_coco
from chainercv.evaluations import eval_detection_voc
from chainercv.utils import apply_to_iterator
from chainercv.utils import bbox_iou
from chainercv.visualizations import vis_bbox

from matplotlib.patches import Rectangle
from skimage.transform import resize
from tabulate import tabulate
from tqdm import tqdm
from pathlib import Path
from contextlib import contextmanager

from cvdatasets.utils import new_iterator
from cvdatasets.utils import pretty_print_dict

from moth_detector.core import dataset
from moth_detector.core import finetuner
from moth_detector.core import model
from moth_detector.core.training import trainer

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

	def train(self, experiment_name):
		return self.tuner.run(
			opts=self.opts,
			trainer_cls=trainer.SSDTrainer,
			sacred_params=dict(
				name=experiment_name,
				comm=self.comm,
				no_observer=self.opts.no_sacred
			)
		)

	def detect(self):
		data = dict(
			train=self.tuner.train_data,
			test=self.tuner.val_data,
			val=self.tuner.val_data,
		)[self.opts.subset]

		device = convert._get_device(self.tuner.device)
		detector = self.tuner.clf

		if device.device.id >= 0:
			device.use()
			detector.to_gpu(device.device.id)

		detector.model.score_thresh = 0.5
		n_rows, n_cols = self.opts.rows, self.opts.cols

		idxs = np.arange(len(data))
		np.random.shuffle(idxs)

		bar = tqdm(idxs, desc="Processing batches")
		for _, n in enumerate(np.arange(len(data), step=n_cols*n_rows), 1):
			cur_idxs = idxs[n : n+n_cols*n_rows]

			fig, axs = plt.subplots(n_rows, n_cols, figsize=(16,9))
			fig.suptitle("GT boxes: $blue$ | Predicted boxes: $black$")


			for i, idx in enumerate(cur_idxs):
				img, gt = data.get_img_data(idx)
				multi_box = data.multi_box(idx, keys=["y0", "x0", "y1", "x1", "w", "h"])

				gt_boxes = np.array(multi_box, dtype=np.int32)[:, :-2]

				ax = axs[np.unravel_index(i, (n_rows, n_cols))]
				ax.axis("off")

				boxes, labels, scores = detector.model.predict([img])
				box, label, score = boxes[0], labels[0], scores[0]
				iou = bbox_iou(gt_boxes, box).max(axis=0)

				vis_bbox(img, box, label, score=iou,
					label_names=["IoU"],
					ax=ax,
					alpha=0.7,
					instance_colors=[(0,0,0)]
				)
				for y, x, y1, x1, w, h in multi_box:
					ax.add_patch(Rectangle(
						(x, y), w, h,
						fill=False,
						linewidth=3,
						alpha=0.5,
						edgecolor="blue"
					))

				bar.update()

			plt.tight_layout()
			if self.opts.vis_output:
				out_dir = Path(self.opts.vis_output)
				out_dir.mkdir(parents=True, exist_ok=True)
				plt.savefig(out_dir / f"det{_:03d}.jpg", dpi=300)
			else:
				plt.show()

			plt.close()


	def evaluate(self, converter=convert.concat_examples):
		data = dict(
			train=self.tuner.train_data,
			test=self.tuner.val_data,
			val=self.tuner.val_data,
		)[self.opts.subset]

		device = convert._get_device(self.tuner.device)
		detector = self.tuner.clf

		if device.device.id >= 0:
			device.use()
			detector.to_gpu(device.device.id)
		_converter = lambda batch: convert._call_converter(converter, batch, device=device)

		iterator, n_batches = new_iterator(data,
			n_jobs=self.opts.n_jobs,
			batch_size=self.opts.batch_size,
			shuffle=False,
			repeat=False)

		_it = iter(tqdm(iterator,
			total=n_batches, leave=False,
			desc=f"Evaluating"))

		in_values, out_values, rest_values = apply_to_iterator(
			detector.predict, _it, n_input=1)

		# delete unused iterators explicitly
		del in_values
		pred_bboxes, pred_labels, pred_scores = out_values
		gt_bboxes, gt_labels = rest_values

		# unpack the generators
		pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels = zip(*zip(
			pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels))

		result_coco = eval_detection_coco(
			pred_bboxes, pred_labels, pred_scores,
			gt_bboxes, gt_labels)

		rows = []
		for key in sorted(result_coco.keys()):
			if key in ["coco_eval", "existent_labels"]:
				continue
			value = result_coco[key]
			rows.append((key.replace("/", ", "), f"{float(value):.2%}"))

		print("COCO evaluation:")
		print(tabulate(rows, headers=("Metric", "Score"), tablefmt="fancy_grid"))

		threshs = np.arange(0.5, 0.96, 0.05)
		values = np.zeros_like(threshs)

		for i, thresh in enumerate(threshs):
			result = eval_detection_voc(
				pred_bboxes, pred_labels, pred_scores,
				gt_bboxes, gt_labels,
				iou_thresh=thresh)

			values[i] = result["map"]

		print("VOC evaluation:")
		rows = [(f"mAP@{int(thresh * 100):d}", f"{value:.2%}") for thresh, value in zip(threshs, values)]
		print(tabulate(rows, headers=("Metric", "Score"), tablefmt="fancy_grid"))

		fig, ax = plt.subplots()
		ax.plot(threshs, values)
		ax.set_xlabel("Thresholds")
		ax.set_ylabel("mAP@$X$")

		plt.show()
		plt.close()


	@contextmanager
	def eval_mode(self):
		with chainer.using_config("train", False), chainer.no_backprop_mode():
			yield

	def __call__(self, experiment_name, *args, **kwargs):

		if self.opts.mode == "train":
			return self.train(experiment_name)

		elif self.opts.mode == "detect":
			with self.eval_mode():
				try:
					return self.detect()
				except KeyboardInterrupt:
					pass

		elif self.opts.mode == "evaluate":
			with self.eval_mode():
				return self.evaluate()

		else:
			raise NotImplementedError(f"this mode is not implemented: {self.opts.mode}")
