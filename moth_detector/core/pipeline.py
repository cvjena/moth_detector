import chainer
import logging
import numpy as np
import matplotlib.pyplot as plt

from chainer.dataset import convert
from chainer.training.updaters import StandardUpdater
from chainer_addons.training import MiniBatchUpdater
from chainercv.visualizations import vis_bbox
from chainercv.utils import bbox_iou

from tqdm import tqdm
from skimage.transform import resize
from matplotlib.patches import Rectangle

from cvdatasets.utils import new_iterator
from cvdatasets.utils import pretty_print_dict

from moth_detector.core import dataset
from moth_detector.core import finetuner
from moth_detector.core import model
from moth_detector.core import trainer

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

	def detect(self, converter=convert.concat_examples):
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

		detector.model.use_preset("visualize")

		idxs = np.random.choice(len(data), 16, replace=False)

		fig, axs = plt.subplots(4, 4, figsize=(16,9))

		for i, idx in enumerate(tqdm(idxs)):
			img, gt = data.get_img_data(idx)
			gt_box = data.bounding_box(idx)
			x, y, w, h = gt_box
			x0, y0, x1, y1 = x, y, x+w, y+h

			ax = axs[np.unravel_index(i, (4,4))]
			ax.axis("off")

			boxes, labels, scores = detector.model.predict([img])
			box, label, score = boxes[0], labels[0], scores[0]
			iou = bbox_iou(np.array([[y0, x0, y1, x1]]), box)[0]

			vis_bbox(img, box, label, score=iou,
				label_names=["IoU"],
				ax=ax,
				alpha=0.5,
				instance_colors=[(0,0,0,1)]
			)
			ax.add_patch(Rectangle(
				(x, y), w, h,
				fill=False,
				linewidth=2,
				alpha=0.5,
				edgecolor="blue"
			))

		plt.show()
		plt.close()


	def __call__(self, experiment_name, *args, **kwargs):

		if self.opts.mode == "train":
			return self.train(experiment_name)

		elif self.opts.mode == "detect":
			with chainer.using_config("train", False), chainer.no_backprop_mode():
				return self.detect()

		else:
			raise NotImplementedError(f"this mode is not implemented: {self.opts.mode}")
