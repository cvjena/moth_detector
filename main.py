#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

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
