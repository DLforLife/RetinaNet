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

	@staticmethod
	def iou(y_box, pred_box):
		intersection_x1 = np.max(y_box[0], pred_box[0])
		intersection_y1 = np.max(y_box[1], pred_box[1])
		intersection_x2 = np.min(y_box[2], pred_box[2])
		intersection_y2 = np.min(y_box[3], pred_box[3])
		intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
		union_area = (y_box[0] - y_box[2]) * (y_box[1] - y_box[3]) + \
		             (pred_box[0] - pred_box[2]) * (pred_box[1] - pred_box[3]) - intersection_area
		return intersection_area / union_area

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
					if (self.iou(y_box, pred_box) > self.iou_threshold and
							not y_box in detected_boxes and
							not pred_box in matched_predictions):
						tp += 1
						detected_boxes.append(y_box)
						matched_predictions.append(pred_box)

		fp += len(pred_boxes) - len(matched_predictions)
		fn += len(y_boxes) - len(detected_boxes)
		self.tp += tp
		self.fp += fp
		self.fn += fn
		self.precision = self.tp / (self.tp + self.fp)
		self.recall = self.tp / (self.tp + self.fn)

	def reset(self):
		self.fp = 0
		self.fn = 0
		self.tp = 0
		self.precision = 0
		self.recall = 0
