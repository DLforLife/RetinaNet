"""
Definitions for the metrics used in our network. Each metric is to be implemented as a class. Example: IOU
"""

class Metrics():
	def __init__(self):
		self.fp = None
		self.fn = None
		self.tp = None

	def precision(self, (y_out_classes, y_classes), (y_out_boxes, y_boxes)):
		return self.tp/(self.tp + self.fp)

	def recall(self, (y_out_classes, y_classes), (y_out_boxes, y_boxes)):
		return self.tp/(self.tp + self.fn)