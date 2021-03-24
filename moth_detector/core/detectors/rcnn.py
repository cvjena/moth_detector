import chainer
import numpy as np

from chainer.backends import cuda
from chainer import functions as F
from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain as FRCNN_Detector
from chainercv.links.model.faster_rcnn.faster_rcnn_train_chain import _fast_rcnn_loc_loss

from moth_detector.core.models.base import BaseModel
from moth_detector.utils import _unpack


class Detector(FRCNN_Detector, BaseModel):
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
		scales = cuda.to_cpu(scales).tolist()

		# This is what we want to fix here
		# n = bboxes.shape[0]
		# if n != 1:
		# 	raise ValueError('Currently only batch size 1 is supported.')


		N, C, H, W = imgs.shape
		img_size = (H, W)

		features = self.faster_rcnn.extractor(imgs)
		rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(
			features, img_size, scales)

		# bbox = bboxes[0]
		# label = labels[0]
		# rpn_score = rpn_scores[0]
		# rpn_loc = rpn_locs[0]
		# roi = rois
		rpn_loc_loss = 0
		rpn_cls_loss = 0
		roi_loc_loss = 0
		roi_cls_loss = 0

		for bbox, label, rpn_score, rpn_loc in zip(bboxes, labels, rpn_scores, rpn_locs):

			# Sample RoIs and forward
			sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
				rois, bbox, label,
				self.loc_normalize_mean, self.loc_normalize_std)
			sample_roi_index = self.xp.zeros((len(sample_roi),), dtype=np.int32)
			roi_cls_loc, roi_score = self.faster_rcnn.head(
				features, sample_roi, sample_roi_index)

			# RPN losses
			gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
				bbox, anchor, img_size)
			rpn_loc_loss += _fast_rcnn_loc_loss(
				rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
			rpn_cls_loss += F.softmax_cross_entropy(rpn_score, gt_rpn_label)

			# Losses for outputs of the head.
			n_sample = roi_cls_loc.shape[0]
			roi_cls_loc = roi_cls_loc.reshape((n_sample, -1, 4))
			roi_loc = roi_cls_loc[self.xp.arange(n_sample), gt_roi_label]
			roi_loc_loss += _fast_rcnn_loc_loss(
				roi_loc, gt_roi_loc, gt_roi_label, self.roi_sigma)
			roi_cls_loss += F.softmax_cross_entropy(roi_score, gt_roi_label)

		loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss

		chainer.report(dict(
				rpn_loc_loss=rpn_loc_loss,
				rpn_cls_loss=rpn_cls_loss,
				roi_loc_loss=roi_loc_loss,
				roi_cls_loss=roi_cls_loss,
				loss=loss), self)

		return loss
