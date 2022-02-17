import chainer
import copy
import numpy as np
import os
import warnings

try:
	import matplotlib  # NOQA
	_available = True

except (ImportError, TypeError):
	_available = False

from functools import partial
from pathlib import Path

from chainer.backends.cuda import to_cpu
from chainercv.extensions import DetectionVisReport as Original
from chainercv.utils import apply_to_iterator
from chainercv.utils import bbox_iou
from chainercv.visualizations.vis_bbox import vis_bbox



def visualize(model, it, output, fname_fmt, iteration=0, _im_mean=None, ax=None):
	# Dynamically import pyplot so that the backend of matplotlib
	# can be configured after importing chainercv.
	import matplotlib.pyplot as plt

	in_values, out_values, rest_values = apply_to_iterator(
		model, it, n_input=1)

	# get the iterators
	imgs, = in_values
	gt_bboxes, gt_labels = rest_values
	pred_boxes, pred_labels, pred_scores = out_values

	# unpack the generators
	for idx, content in enumerate(zip(imgs, gt_bboxes, gt_labels, pred_boxes, pred_labels, pred_scores)):
		img, gt_bbox, gt_label, pred_bbox, pred_label, pred_score = content

		best = np.argsort(-pred_score)[:50]
		pred_bbox, pred_label = pred_bbox[best], pred_label[best]

		iou = bbox_iou(gt_bbox, pred_bbox).max(axis=0)

		if _im_mean is not None:
			img += _im_mean

		fig, ax = plt.subplots(figsize=(16, 9))

		ax.set_title('GT: $black$, Prediction: $red$')
		ax.axis("off")
		vis_bbox(
			img, pred_bbox, pred_label, iou,
			label_names=["IoU"],
			alpha=0.7,
			ax=ax)

		for lab, (y0, x0, y1, x1) in zip(gt_label, gt_bbox):
			x, y, w, h = x0, y0, x1 - x0, y1 - y0
			if 0 in (w, h):
				continue

			rect = plt.Rectangle((x,y), w, h,
				fill=False,
				linewidth=3,
				edgecolor="black" if lab != -1 else "gray",
				alpha=0.7
			)
			ax.add_patch(rect)

		out_file = fname_fmt.format(index=idx, iteration=iteration)
		fpath = Path(output, out_file)
		fpath.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(fpath, bbox_inches='tight')
		plt.close()


class DetectionVisReport(Original):

	def __call__(self, trainer):
		if not _available:
			return

		if hasattr(self.iterator, 'reset'):
			self.iterator.reset()
			it = self.iterator
		else:
			it = copy.copy(self.iterator)

		predict = partial(self.target.predict, preset="visualize")

		with chainer.using_config("train", False):
			visualize(predict, it,
				_im_mean=getattr(it.dataset, "mean", 0),
				output=trainer.out,
				fname_fmt=self.filename,
				iteration=trainer.updater.iteration,
			)

