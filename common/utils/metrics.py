"""
Definitions for the metrics used in our network. Each metric is to be implemented as a class. Example: IOU
"""

class Metrics():
	def __init__(self, labels, iou_threshold, conf_threshold):
		self.fp = None
		self.fn = None
		self.tp = None
		self.precision = None
		self.recall = None
		self.labels = labels
		self.iou_threshold = iou_threshold
		self.conf_threshold = conf_threshold

	def update_metrics(self, y_boxes, pred_boxes):
		detected_boxes = []
		matched_predictions = []
		tp = 0
		fp = 0
		fn = 0
		for label in self.labels:
			y_boxes_label = y_boxes[label]
			pred_boxes_label = pred_boxes[label]
			for y_box in y_boxes_label:
				for pred_box in pred_boxes_label:
					if(iou(y_box, pred_box)>self.iou_threshold):
						tp += 1
						detected_boxes.append(y_box)
						matched_predictions.append(pred_box)
		fp = len(pred_boxes) - len(matched_predictions)
		fn = len(y_boxes) - len(detected_boxes)
		self.tp += tp
		self.fp += fp
		self.fn += fn
		self.precision = self.tp/(self.tp + self.fp)
		self.recall = self.tp/(self.tp + self.fn)

