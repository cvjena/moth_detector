from chainercv import evaluations as Eval

class _Static:
	def __new__(cls):
		raise TypeError("Static class only!")


class VOCEvaluations(_Static):

	@classmethod
	def prec_rec(cls, *args, **kwargs):
		return Eval.calc_detection_voc_prec_rec(*args, **kwargs)

	@classmethod
	def avg_prec(cls, prec, rec, *args, **kwargs):
		return Eval.calc_detection_voc_ap(prec, rec, *args, **kwargs)

	@classmethod
	def evaluate(cls, *args, **kwargs):
		result = Eval.eval_detection_voc(*args, **kwargs)
		return result["map"]

class COCOEvaluations(_Static):

	@classmethod
	def evaluate(cls, *args, **kwargs):
		result = Eval.eval_detection_coco(*args, **kwargs)

		return {key.replace("/", ", "): float(value) for key, value in result.items()
			if key not in ("coco_eval", "existent_labels")}
