import numpy as np

from cvargparse import Arg
from cvargparse import BaseParser
from cvfinetune.parser import FineTuneParser
from cvfinetune.parser import add_dataset_args
from cvfinetune.parser import add_model_args

def parse_args(args=None, namespace=None):
	main_parser = BaseParser()

	subp = main_parser.add_subparsers(
		title="Execution modes",
		dest="mode",
		required=True
	)

	_common_parser = FineTuneParser(add_help=False, nologging=True)
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
			help="shuffle images before visualization")
	])

	parser = subp.add_parser("evaluate",
		help="Evaluate a trained model",
		parents=[_common_parser])

	parser.add_args([
		Arg("--subset", choices=["train", "val"], default="val",
			help="The subset to evaluate"),

		Arg("--score_threshold", type=float, default=0.5,
			help="score threshold for the bounding box estimation"),

		Arg("--nms_threshold", type=float, default=0.45,
			help="NMS threshold for the bounding box estimation"),

		Arg("--eval_methods", nargs="+", choices=["coco", "voc"], default=["coco", "voc"],
			help="Evaluation method"),

		Arg("--plot_voc", action="store_true",
			help="Plot mAP values of the VOC evaluation"),

		Arg("--voc_thresh", nargs="+", type=float,
			default=np.arange(0.5, 0.96, 0.05).tolist(),
			help="Plot mAP values of the VOC evaluation"),

	])

	return main_parser.parse_args(args=args, namespace=namespace)
