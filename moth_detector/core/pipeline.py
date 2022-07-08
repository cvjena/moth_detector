import chainer
import logging
import matplotlib.pyplot as plt
import numpy as np
import time

from chainer.backends.cuda import to_cpu
from chainer.dataset import convert
from chainercv.utils import apply_to_iterator
from chainercv.utils import bbox_iou
from chainercv.visualizations import vis_bbox
from chainercv.transforms import resize_bbox

from matplotlib.patches import Rectangle
from skimage.transform import resize
from tabulate import tabulate
from functools import partial
from tqdm import tqdm
from functools import partial
from pathlib import Path
from contextlib import contextmanager

from cvdatasets.utils import new_iterator
from cvdatasets.utils import pretty_print_dict

from moth_detector.core import finetuner
from moth_detector.core.training import trainer
from moth_detector.core import evaluations as evals


def profile_data(dataset):
	logging.info("Profiling image data:")
	with dataset.enable_img_profiler():
		dataset[0]


class Pipeline(object):


	@contextmanager
	def eval_mode(self):
		with chainer.using_config("train", False), chainer.no_backprop_mode():
			yield

	def __call__(self, experiment_name, *args, **kwargs):

		self.rnd = np.random.RandomState(self.opts.seed)

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

	def __init__(self, opts):
		super(Pipeline, self).__init__()

		chainer.set_debug(opts.debug)
		if opts.debug:
			logging.warning("DEBUG MODE ENABLED!")

		self.tuner, self.comm = finetuner.new_finetuner(opts)
		self.opts = opts


	def train(self, experiment_name):
		profile_data(self.tuner.train_data)

		return self.tuner.run(
			opts=self.opts,
			trainer_cls=trainer.DetectionTrainer,
		)

	def _get_data_detector(self):

		data = dict(
			train=self.tuner.train_data,
			test=self.tuner.val_data,
			val=self.tuner.val_data,
		)[self.opts.subset]

		profile_data(data)

		detector = self.tuner.clf

		device = convert._get_device(self.tuner.device)
		if hasattr(device, "device") and device.device.id >= 0:
			device.use()
			detector.to_gpu(device.device.id)

		return data, detector

	def _predict(self, *args, preset=None):
		return self.tuner.clf.predict(*args, preset=preset)

	def detect(self, rnd=None):
		data, detector = self._get_data_detector()

		n_rows, n_cols = self.opts.rows, self.opts.cols

		n_samples = len(data)
		idxs = np.arange(n_samples)
		if self.opts.shuffle:
			rnd = getattr(self, "rnd", np.random.RandomState())
			rnd.shuffle(idxs)

		bar = tqdm(idxs, desc="Processing batches")

		# we only have a single image, so display some more info
		detail_view = n_rows == n_cols == 1

		if detail_view:
			grid = plt.GridSpec(1, 2)
		else:
			grid = plt.GridSpec(n_rows, n_cols)

		for n in np.arange(n_samples, step=n_cols*n_rows):
			cur_idxs = idxs[n : n+n_cols*n_rows]

			fig = plt.figure(figsize=(16,9))
			# fig, axs = plt.subplots(n_rows, n_cols, , squeeze=False)
			# [ax.axis("off") for ax in axs.ravel()]

			imgs, gt_bboxes, gt_labels = zip(*data[cur_idxs])
			inputs = imgs, gt_bboxes, gt_labels
			t0 = time.time()
			preds = pred_bboxes, pred_labels, pred_scores = self._predict(list(imgs))
			t1 = time.time()
			logging.info(f"Detection time: {t1-t0:.3f}s")
			imgs0 = [detector.model.preprocess(im, return_all=True)
				for im in imgs]

			for i, (img, gt_boxes, gt, boxes, label, score) in enumerate(zip(*inputs, *preds)):
			for i, (img, gt_boxes, gt_label, boxes, label, score) in enumerate(zip(*inputs, *preds)):
				if boxes.ndim != 2:
					boxes = np.array([[0,0,1,1]], dtype=gt_boxes.dtype)
					label = np.array([0], dtype=gt_label.dtype)
					score = np.array([0.01], dtype=np.float32)

				iou = bbox_iou(gt_boxes, boxes).max(axis=0)
				label_names = ["IoU"]

				if (iou == 0).all():
					iou = label_names = None

				orig = data.get_im_obj(cur_idxs[i]).im_array
				img = data.prepare_back(img)
				gt_boxes = resize_bbox(gt_boxes, img.shape, orig.shape)
				boxes = resize_bbox(boxes, img.shape, orig.shape)

				img2 = imgs0[i]
				if isinstance(img2, list):
					img2 = imgs0[i][0]

				row, col = np.unravel_index(i, (n_rows, n_cols))
				ax = plt.subplot(grid[row, col])
				ax.axis("off")
				ax.imshow(orig)

				vis_bbox(None, boxes, label,
					# score=iou,
					# label_names=label_names,
					ax=ax,
					alpha=0.7,
					instance_colors=[(0,0,255)]
				)
				for lab, (y0, x0, y1, x1) in zip(gt_label, gt_boxes):
					w, h = x1-x0, y1-y0
					ax.add_patch(Rectangle(
						(x0, y0), w, h,
						fill=False,
						linewidth=2,
						alpha=0.7,
						edgecolor="black" if lab != -1 else "gray"
					))


				if self.opts.voc_thresh:
					threshs = self.opts.voc_thresh
					values = [0 for _ in range(len(threshs))]
					# n_threshs = len(threshs)
					# _rows = int(np.ceil(np.sqrt(n_threshs)))
					# _cols = int(np.ceil(np.sqrt(n_threshs)))
					# _f, _axs = plt.subplots(_rows, _cols, squeeze=False)

					for i, thresh in enumerate(threshs):
						pred = evals.Records([boxes], [label], [score])
						gt = evals.Records([gt_boxes], [gt_label])
						precs, recs = evals.VOCEvaluations.prec_rec(pred, gt, iou_thresh=thresh)
						ap = evals.VOCEvaluations.avg_prec(precs, recs)
						values[i] = np.nanmean(ap)

						if not detail_view:
							continue

						_ax = plt.subplot(grid[0, 1])
						_ax.scatter(recs[0], precs[0], alpha=0.7, marker="x")
						_ax.plot(recs[0], precs[0],
							label=f"mAP@{thresh:.2f} | AP: {np.nanmean(ap):.2f}")

					if detail_view:
						_ax = plt.subplot(grid[0, 1])
						_ax.scatter([0,1], [0,1], c="white", alpha=0.01)
						_ax.legend()
						_ax.set_xlabel("Recall")
						_ax.set_ylabel("Precision")

					metrics = " | ".join([f"mAP@{thresh:.2f}: {value:.2%}" for thresh, value in zip(threshs, values)])
					ax.set_title(f"{len(boxes)}/{(gt_label!=-1).sum()} Boxes predicted: {metrics}")
				bar.update()

			plt.tight_layout()
			if self.opts.vis_output:
				out_dir = Path(self.opts.vis_output)
				out_dir.mkdir(parents=True, exist_ok=True)
				plt.savefig(out_dir / f"det{_:03d}.jpg", dpi=300)
				plt.savefig(out_dir / f"det{_:03d}.svg", dpi=300)
			else:
				plt.show()

			plt.close()


	def evaluate(self, converter=convert.concat_examples):
		data, model = self._get_data_detector()
		iterator, n_batches = new_iterator(data,
			n_jobs=self.opts.n_jobs,
			batch_size=self.opts.batch_size,
			shuffle=False,
			repeat=False)

		_it = iter(tqdm(iterator,
			total=n_batches, leave=False,
			desc=f"Evaluating"))

		in_values, out_values, rest_values = apply_to_iterator(self._predict, _it, n_input=1)

		# delete unused iterators explicitly
		del in_values

		# unpack the generators
		pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels = zip(*zip(
			*out_values, *rest_values))

		pred = evals.Records(pred_bboxes, pred_labels, pred_scores)
		gt = evals.Records(gt_bboxes, gt_labels)
		if "coco" in self.opts.eval_methods:

			results = evals.COCOEvaluations.evaluate(pred, gt)
			rows = list(sorted(results.items(), key=lambda item: item[0]))

			print("COCO evaluation:")
			print(tabulate(rows, headers=("Metric", "Score"), tablefmt="fancy_grid"))

		if "voc" in self.opts.eval_methods:
			threshs = np.array(self.opts.voc_thresh)
			values = np.zeros_like(threshs)

			for i, thresh in enumerate(threshs):
				values[i] = evals.VOCEvaluations.evaluate(pred, gt, iou_thresh=thresh)

			print("VOC evaluation:")
			rows = [(f"mAP@{int(thresh * 100):d}", f"{value:.2%}") for thresh, value in zip(threshs, values)]
			print(tabulate(rows, headers=("Metric", "Score"), tablefmt="fancy_grid"))
			print(*values, sep="\n")

			if self.opts.plot_voc:
				fig, ax = plt.subplots()
				ax.plot(threshs, values)
				ax.set_xlabel("Thresholds")
				ax.set_ylabel("mAP@$X$")

				plt.show()
				plt.close()

