
def _unpack(arr):
	""" in case of chainer.Variable, return the actual array
		otherwise return itself
	"""
	return getattr(arr, "array", arr)
