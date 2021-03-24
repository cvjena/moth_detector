import chainer
import numpy as np

from chainer import functions as F
from chainer.backends.cuda import to_cpu
from chainercv.evaluations import eval_detection_voc
from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain as FRCNN_Detector
from chainercv.links.model.faster_rcnn.faster_rcnn_train_chain import _fast_rcnn_loc_loss
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox

from moth_detector.core.detectors.base import BaseDetector
from moth_detector.utils import _unpack


class Detector(FRCNN_Detector, BaseDetector):
	__name__ = "Fast R-CNN Detector"

	def __init__(self, model, *args, loss_func, **kwargs):
		super(Detector, self).__init__(faster_rcnn=model, *args, **kwargs)

	@property
	def model(self):
		return self.faster_rcnn

	# updated version of FasterRCNNTrainChain.forward
	def forward(self, imgs, bboxes, labels, scales):
		"""Forward Faster R-CNN and calculate losses.

		Here are notations used.

		* :math:`N` is the batch size.
		* :math:`R` is the number of bounding boxes per image.

		Currently, only :math:`N=1` is supported.

		Args:
			imgs (~chainer.Variable): A variable with a batch of images.
			bboxes (~chainer.Variable): A batch of bounding boxes.
				Its shape is :math:`(N, R, 4)`.
			labels (~chainer.Variable): A batch of labels.
				Its shape is :math:`(N, R)`. The background is excluded from
				the definition, which means that the range of the value
				is :math:`[0, L - 1]`. :math:`L` is the number of foreground
				classes.
			scales (~chainer.Variable): Amount of scaling applied to
				each input image during preprocessing.

		Returns:
			chainer.Variable:
			Scalar loss variable.
			This is the sum of losses for Region Proposal Network and
			the head module.

		"""
		bboxes = _unpack(bboxes)
		labels = _unpack(labels)
		scales = _unpack(scales)
		scales = to_cpu(scales).tolist()

		# This is what we want to fix here
		# n = bboxes.shape[0]
		# if n != 1:
		# 	raise ValueError('Currently only batch size 1 is supported.')


		N, C, H, W = imgs.shape
		img_size = (H, W)

		features = self.faster_rcnn.extractor(imgs)
		_rpn_locs, _rpn_scores, rois, roi_indices, _anchor = \
			self.faster_rcnn.rpn(features, img_size, scales)

		pred_bboxes, pred_labels, pred_scores = \
			self._detection(features, rois, roi_indices)

		self.report_mAP(pred_bboxes, pred_labels, pred_scores, bboxes, labels)

		rpn_loc_loss = 0
		rpn_cls_loss = 0
		roi_loc_loss = 0
		roi_cls_loss = 0

		for bbox, label, rpn_score, rpn_loc in zip(bboxes, labels, _rpn_scores, _rpn_locs):

			sample_roi, gt_roi_loc, gt_roi_label = \
				self.proposal_target_creator(rois, bbox, label,
					self.loc_normalize_mean, self.loc_normalize_std)
			gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, _anchor, img_size)

			# Sample RoIs and forward
			sample_roi_index = self.xp.zeros((len(sample_roi),), dtype=np.int32)
			roi_cls_loc, roi_score = self.faster_rcnn.head(features, sample_roi, sample_roi_index)

			# RPN losses
			rpn_loc_loss += _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)

			# Losses for outputs of the head.
			n_sample = roi_cls_loc.shape[0]
			_roi_cls_loc = roi_cls_loc.reshape((n_sample, -1, 4))
			roi_loc = _roi_cls_loc[self.xp.arange(n_sample), gt_roi_label]
			roi_loc_loss += _fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label, self.roi_sigma)

			# rpn and roi cls losses
			rpn_cls_loss += F.softmax_cross_entropy(rpn_score, gt_rpn_label)
			roi_cls_loss += F.softmax_cross_entropy(roi_score, gt_roi_label)

		rpn_loc_loss /= N
		rpn_cls_loss /= N
		roi_loc_loss /= N
		roi_cls_loss /= N

		loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss

		self.report(
			loss=loss,
			loc_loss=rpn_loc_loss + roi_loc_loss,
			conf_loss=rpn_cls_loss + roi_cls_loss,
		)

		return loss

	def _detection(self, features, rois, roi_indices):
		rcnn = self.faster_rcnn

		loc_norms = [rcnn.loc_normalize_mean, rcnn.loc_normalize_std]
		mean, std = [np.tile(np.asarray(arr), rcnn.n_class) for arr in loc_norms]

		with chainer.using_config("train", False), chainer.no_backprop_mode():
			roi_cls_locs, roi_scores = rcnn.head(features, rois, roi_indices)

		roi_cls_locs = to_cpu(_unpack(roi_cls_locs))
		roi_scores = to_cpu(_unpack(roi_scores))
		roi_indices = to_cpu(_unpack(roi_indices))
		rois = to_cpu(_unpack(rois))

		bboxes, labels, scores = [], [], []
		for i in np.unique(roi_indices):
			mask = roi_indices == i

			roi_cls_loc = (roi_cls_locs[mask] * std + mean).astype(np.float32)
			roi_cls_loc = roi_cls_loc.reshape((-1, rcnn.n_class, 4))

			roi = np.broadcast_to(rois[mask, None], roi_cls_loc.shape)

			cls_bbox = loc2bbox(roi.reshape((-1, 4)), roi_cls_loc.reshape((-1, 4)))
			cls_bbox = cls_bbox.reshape((-1, rcnn.n_class * 4))

			# clip bounding box
			# cls_bbox[:, 0::2] = xp.clip(cls_bbox[:, 0::2], 0, size[0])
			# cls_bbox[:, 1::2] = xp.clip(cls_bbox[:, 1::2], 0, size[1])

			prob = F.softmax(roi_scores[mask]).array

			raw_cls_bbox = to_cpu(cls_bbox)
			raw_prob = to_cpu(prob)

			_bboxes, _labels, _scores = rcnn._suppress(raw_cls_bbox, raw_prob)
			bboxes.append(_bboxes)
			labels.append(_labels)
			scores.append(_scores)

		return bboxes, labels, scores
