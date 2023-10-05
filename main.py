#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import numpy as np
import sys
import warnings

from pathlib import Path

from moth_detector.core.pipeline import Pipeline
from moth_detector.utils import parser

def mcc_setup():

	scanner_dir = Path(__file__).parent.parent
	mcc_root = scanner_dir / "mcc_detector"
	mcc_code = mcc_root / "code"

	if mcc_code.exists():
		sys.path.append(str(mcc_code))
		try:
			import idac  # noqa: F401
		except ImportError:
			raise RuntimeError("Could not find MCC code! Please check your installation")

		from moth_detector.core.models import MCCModel
		MCCModel.config = mcc_root / "config" / "MCC_config.json"

		assert MCCModel.config.exists(), \
			f"Could not find MCC config file: {MCCModel.config}"
	else:
		warnings.warn(
			f"Could not find MCC implementation! (looked here: \"{mcc_code}\")\n"\
			"Please clone it from https://github.com/kimbjerge/MCC-trap\n"\
			f"git clone git@github.com:kimbjerge/MCC-trap.git {mcc_root}"
		)





def main(args):
	mcc_setup()
	pipeline = Pipeline(args)
	return pipeline(experiment_name="Moth detector")


np.seterr(all="ignore")
chainer.config.cv_resize_backend = "cv2"
main(parser.parse_args())
