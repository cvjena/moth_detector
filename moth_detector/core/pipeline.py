import chainer
import logging
import matplotlib.pyplot as plt
import numpy as np

from chainer.backends.cuda import to_cpu
from chainer.dataset import convert
from chainercv.evaluations import eval_detection_coco
from chainercv.evaluations import eval_detection_voc
from chainercv.utils import apply_to_iterator
from chainercv.utils import bbox_iou
from chainercv.visualizations import vis_bbox
from chainercv.transforms import resize_bbox

from matplotlib.patches import Rectangle
from skimage.transform import resize
from tabulate import tabulate
from functools import partial
from tqdm import tqdm
from pathlib import Path
from contextlib import contextmanager

from cvdatasets.utils import new_iterator
from cvdatasets.utils import pretty_print_dict

from moth_detector.core import finetuner
from moth_detector.core.training import trainer


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

	def detect(self, rnd=None):
		data = dict(
			train=self.tuner.train_data,
			test=self.tuner.val_data,
			val=self.tuner.val_data,
		)[self.opts.subset]

		profile_data(data)

		device = convert._get_device(self.tuner.device)
		detector = self.tuner.clf

		if device.device.id >= 0:
			device.use()
			detector.to_gpu(device.device.id)

		detector.model.score_thresh = 0.5
		n_rows, n_cols = self.opts.rows, self.opts.cols

		n_samples = len(data)
		idxs = np.arange(n_samples)
		if self.opts.shuffle:
			rnd = getattr(self, "rnd", np.random.RandomState())
			rnd.shuffle(idxs)

		bar = tqdm(idxs, desc="Processing batches")
		for _, n in enumerate(np.arange(n_samples, step=n_cols*n_rows), 1):
			cur_idxs = idxs[n : n+n_cols*n_rows]

			fig, axs = plt.subplots(n_rows, n_cols, figsize=(16,9), squeeze=False)
			[ax.axis("off") for ax in axs.ravel()]
			# fig.suptitle("GT boxes: $blue$ | Predicted boxes: $black$")

			imgs, gt_bboxes, gt_labels = zip(*data[cur_idxs])
			inputs = imgs, gt_bboxes, gt_labels
			preds = pred_bboxes, pred_labels, pred_scores = detector.predict(list(imgs), preset="visualize")

			imgs0 = [detector.model.preprocess(im, return_all=True)
				for im in imgs]

			for i, (img, gt_boxes, gt, boxes, label, score) in enumerate(zip(*inputs, *preds)):
				if boxes.ndim != 2:
					boxes = np.array([[0,0,1,1]], dtype=gt_boxes.dtype)
					label = np.array([0], dtype=gt.dtype)
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
				# img2 = imgs0[i][-1]
				row, col = np.unravel_index(i, (n_rows, n_cols))
				ax = axs[row, col]
				# ax2 = axs[row+n_rows, col]

				ax.imshow(orig)
				# ax2.imshow(img2, cmap=plt.cm.gray)

				vis_bbox(None, boxes, label, score=iou,
					label_names=label_names,
					ax=ax,
					alpha=0.7,
					instance_colors=[(0,0,255)]
				)
				for lab, (y0, x0, y1, x1) in zip(gt, gt_boxes):
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

					for i, thresh in enumerate(threshs):
						result = eval_detection_voc(
							[boxes], [label], [score], [gt_boxes], [gt],
							iou_thresh=thresh)

						values[i] = result["map"]

					title = " | ".join([f"mAP@{thresh}: {value:.2%}" for thresh, value in zip(threshs, values)])
					ax.set_title(title)
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
		data = dict(
			train=self.tuner.train_data,
			test=self.tuner.val_data,
			val=self.tuner.val_data,
		)[self.opts.subset]

		profile_data(data)

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
			partial(detector.predict, preset="evaluate"), _it, n_input=1)

		# delete unused iterators explicitly
		del in_values
		pred_bboxes, pred_labels, pred_scores = out_values
		gt_bboxes, gt_labels = rest_values

		# unpack the generators
		pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels = zip(*zip(
			pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels))

		if "coco" in self.opts.eval_methods:
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

		if "voc" in self.opts.eval_methods:
			threshs = np.array(self.opts.voc_thresh)
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
			print(*values, sep="\n")

			if self.opts.plot_voc:
				fig, ax = plt.subplots()
				ax.plot(threshs, values)
				ax.set_xlabel("Thresholds")
				ax.set_ylabel("mAP@$X$")

				plt.show()
				plt.close()

