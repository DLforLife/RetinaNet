class dataset_loader:
	def __init__(self,config):
		pass
	def next_batch(self,is_training):
		'''

		:param is_training: bool
		:return:
		feed_dict = {self.model.x: x_batch,
					 self.model.y_classes: y_classes_batch,
					 self.model.y_boxes: y_boxes_batch,
					 self.model.is_training: is_training
					 }
		'''

		feed_dict = dict()

		return feed_dict

	def reset(self):
		pass