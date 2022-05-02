#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import sys
from pathlib import Path

def mcc_setup():

	scanner_dir = Path(__file__).parent.parent
	mcc_root = scanner_dir / "mcc_detector"
	mcc_code = mcc_root / "code"

	assert mcc_code.exists(), \
		f"Could not find MCC implementation! (looked here: \"{mcc_code}\")"
	sys.path.append(str(mcc_code))

	try:
		import idac
	except ImportError:
		raise RuntimeError("Could not find MCC code! Please check your installation")

	from moth_detector.core.models import MCCModel
	MCCModel.config = mcc_root / "config" / "MCC_config.json"

	assert MCCModel.config.exists(), \
		f"Could not find MCC config file: {MCCModel.config}"


mcc_setup()

import chainer
import numpy as np

from moth_detector.core.pipeline import Pipeline
from moth_detector.utils import parser

def main(args):
	pipeline = Pipeline(args)
	return pipeline(experiment_name="Moth detector")


np.seterr(all="ignore")
chainer.config.cv_resize_backend = "cv2"
main(parser.parse_args())
