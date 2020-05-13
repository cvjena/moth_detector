
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
		help="Start training",
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
	])

	return main_parser.parse_args(args=args, namespace=namespace)
