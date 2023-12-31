import numpy as np

from cvargparse import Arg
from cvargparse import BaseParser
from cvfinetune.parser import FineTuneParser

def add_training_args(subp, _common_parser):

	parser = subp.add_parser("train",
		help="Starts moth detector training",
		parents=[_common_parser])

	parser.add_args([
		Arg("--update_size", type=int, default=-1,
			help="if positive, MiniBatchUpdater is used." + \
			"It accumulates gradients over the training and " + \
			"updates the weights only if the update size is exceeded"),

		Arg("--mpi", action="store_true",
			help="Indicates that OpenMPI is used!"),

	], group_name="Training arguments")

	parser.add_args([
		Arg("--no_sacred", action="store_true",
			help="do save outputs to sacred"),

	], group_name="Sacred arguments")

def add_detection_args(subp, _common_parser):

	parser = subp.add_parser("detect",
		help="Use trained model to detect the objects / moths",
		parents=[_common_parser])

	parser.add_args([
		Arg("--subset", choices=["train", "val"], default="val",
			help="The subset to detect"),

		Arg("--rows", "-r", type=int, default=2,
			help="Number of rows in the figure "),
		Arg("--cols", "-c", type=int, default=3,
			help="Number of cols in the figure "),

		Arg("--vis_output",
			help="Save resulting visualizations under this folder"),
		Arg("--shuffle", action="store_true",
			help="shuffle images before visualization"),


	])
	parser.add_args([
		Arg("--voc_thresh", nargs="+", type=float,
			default=[0.33, 0.5, 0.75],
			help="Plot mAP values of the VOC evaluation"),

	], group_name="Thresholds")


def add_evaluation_args(subp, _common_parser):

	parser = subp.add_parser("evaluate",
		help="Evaluate a trained model",
		parents=[_common_parser])

	parser.add_args([
		Arg("--subset", choices=["train", "val"], default="val",
			help="The subset to evaluate"),

		Arg("--eval_methods", nargs="+",
			choices=["coco", "voc", "ap"],
			default=["ap"],
			help="Evaluation method"),

		Arg("--plot_voc", action="store_true",
			help="Plot mAP values of the VOC evaluation"),

	])
	parser.add_args([
		Arg("--voc_thresh", nargs="+", type=float,
			default=np.arange(0.5, 0.96, 0.05).tolist(),
			help="Plot mAP values of the VOC evaluation"),

	], group_name="Thresholds")


def parse_args(args=None, namespace=None):
	main_parser = BaseParser()

	subp = main_parser.add_subparsers(
		title="Execution modes",
		dest="mode",
		required=True
	)

	_common_parser = FineTuneParser(
		model_modules=["chainercv"],
		add_help=False,
		nologging=True)

	_common_parser.add_choices("model_type",
		"shallow", "mcc")

	_common_parser.add_args([

		Arg.int("--max_boxes", default=128,
			help="Maximum amount of boxes to consider. Used also for the box padding in the dataset."),

		Arg.float("--area_threshold", default=0,
			help="Minimum area (relative to the image area) of the objects that are considered relevant."),

		Arg.float("--score_threshold", default=0.1,
			help="score threshold for the bounding box estimation"),

		Arg.float("--nms_threshold", default=0.45,
			help="NMS threshold for the bounding box estimation"),

	], group_name="Thresholds")

	add_training_args(subp, _common_parser)
	add_detection_args(subp, _common_parser)
	add_evaluation_args(subp, _common_parser)

	return main_parser.parse_args(args=args, namespace=namespace)
