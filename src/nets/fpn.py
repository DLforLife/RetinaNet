"""
An implementation for the feature pyramid network class.
"""
import tensorflow as tf
from bunch import Bunch

from layers import *


class FPN:
	def __init__(self, config):
		self.config = config
		#########################################
		self.config.img_size = self.config.img_width * self.config.img_height
		self.number_of_anchors = self.config.number_of_anchors
		self.number_of_classes = self.config.number_of_classes
		self.confidence_threshold = self.config.confidence_threshold
		self.iou_threshold = self.config.iou_threshold
		self.learning_rate = self.config.learning_rate
		self.anchor_scales = self.config.anchor_scales
		self.anchor_ratios = self.config.anchor_ratios
		self.anchor_sizes = self.config.anchor_sizes
		self.anchor_strides = self.config.anchor_strides
		#########################################
		self.x = None
		self.y_classes = None
		self.y_boxes = None
		self.is_training = None
		self.batch_dim = None
		#########################################
		self.y_out_classes = None
		self.y_out_boxes = None
		#########################################
		self.loss = None
		self.loss_indicator = None
		self.train_op = None
		self.metrics = Metrics(self.labels, self.confidence_threshold, self.iou_threshold)
		self.precision = None
		self.recall = None
		#########################################
		self.global_epoch_tensor = None
		self.global_epoch_input = None
		self.global_epoch_assign_op = None
		self.global_step_tensor = None
		self.global_step_input = None
		self.global_step_assign_op = None
		#########################################
		self.best_measure_tensor = None
		self.best_measure_input = None
		self.best_measure_assign_op = None
		#########################################

	def build(self):
		self.init_helper_variables()
		self.init_input()
		self.init_resnet()
		self.init_network()
		self.init_output()

	def init_helper_variables(self):
		"""
		Create a global step variable to be a reference to the number of iterations
		Create cur epoch tensor to totally save the process of the training
		Save the best iou on validation
		:return:
		"""
		with tf.variable_scope('global_epoch'):
			self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
			self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
			self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)
		with tf.variable_scope('global_step'):
			self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
			self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
			self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)
		with tf.variable_scope('best_measure'):
			self.best_measure_tensor = tf.Variable(0.0, trainable=False, name='best_measure')
			self.best_measure_input = tf.placeholder('float32', None, name='best_measure_input')
			self.best_measure_assign_op = self.best_measure_tensor.assign(self.best_measure_input)

	def init_input(self):
		with tf.name_scope('input'):
			self.x = tf.placeholder(tf.float32, [self.config.batch_size,
												 self.config.img_height,
												 self.config.img_width,
												 self.config.num_channels])
			self.y_classes = tf.placeholder(tf.float32, [self.config.batch_size,
														 self.config.img_height,
														 self.config.img_width,
														 self.config.number_of_anchors,
														 self.config.number_of_classes])
			self.y_boxes = tf.placeholder(tf.float32, [self.config.batch_size,
													   self.config.img_height,
													   self.config.img_width,
													   self.config.number_of_anchors, 4])
			self.is_training = tf.placeholder(tf.bool)

	def init_resnet(self):
		with tf.variable_scope('stage_1'):
			x =conv('conv_7x7', self.x, 256, kernel_size=(7, 7), stride=(2, 2))
			self.stage1_out = tf.nn.max_pool(x, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

		with tf.variable_scope('stage_2'):
			x =residual_block('res_block_1', self.stage1_out, filters=[64, 64, 256], is_training=self.is_training)
			x =residual_block('res_block_2', x, filters=[64, 64, 256], is_training=self.is_training)
			self.stage2_out =residual_block('res_block_3', x, filters=[64, 64, 256], is_training=self.is_training)

		with tf.variable_scope('stage_3'):
			x =residual_block('res_block_1', self.stage2_out, filters=[128, 128, 512], strides=2, is_training=self.is_training)
			x =residual_block('res_block_2', x, filters=[128, 128, 512], is_training=self.is_training)
			x =residual_block('res_block_3', x, filters=[128, 128, 512], is_training=self.is_training)
			self.stage3_out =residual_block('res_block_4', x, filters=[128, 128, 512], is_training=self.is_training)

		with tf.variable_scope('stage_4'):
			x =residual_block('res_block_1', self.stage3_out, filters=[256, 256, 1024], strides=2, is_training=self.is_training)
			x =residual_block('res_block_2', x, filters=[256, 256, 1024], is_training=self.is_training)
			x =residual_block('res_block_3', x, filters=[256, 256, 1024], is_training=self.is_training)
			x =residual_block('res_block_5', x, filters=[256, 256, 1024], is_training=self.is_training)
			x =residual_block('res_block_6', x, filters=[256, 256, 1024], is_training=self.is_training)
			self.stage4_out =residual_block('res_block_7', x, filters=[256, 256, 1024], is_training=self.is_training)

		with tf.variable_scope('stage_5'):
			x =residual_block('res_block_1', self.stage4_out, filters=[512, 512, 2048], strides=2, is_training=self.is_training)
			x =residual_block('res_block_2', x, filters=[512, 512, 2048], is_training=self.is_training)
			self.stage5_out =residual_block('res_block_3', x, filters=[512, 512, 2048], is_training=self.is_training)

	def init_network(self):
		with tf.variable_scope('top_down_pathway'):
			with tf.variable_scope('p5'):
				merge0 =conv('conv1x1', self.stage5_out, num_filters=256, kernel_size=(1, 1), stride=(1, 1))
				self.merge1 =merge_block('merge', self.stage4_out, merge0)
			print('merge1 shape:', self.merge1.get_shape())
			with tf.variable_scope('p4'):
				self.merge2 =merge_block('merge', self.stage3_out, self.merge1)
			print('merge2 shape:', self.merge2.get_shape())
			with tf.variable_scope('p3'):
				self.merge3 =merge_block('merge', self.stage2_out, self.merge2)
			print('merge3 shape:', self.merge3.get_shape())
			with tf.variable_scope('p6'):
				self.p6 =conv('conv', self.stage5_out, 256, (3, 3), stride=(2, 2))
			with tf.variable_scope('p7'):
				self.p7 =relu('relu', self.p6)
				self.p7 =conv('conv', self.p7, 256, (3, 3), stride=(2, 2))

			self.class_subnet_out1 = self._class_subnet(self.merge1)
			self.class_subnet_out2 = self._class_subnet(self.merge2)
			self.class_subnet_out3 = self._class_subnet(self.merge3)
			self.class_subnet_out4 = self._class_subnet(self.p6)
			self.class_subnet_out5 = self._class_subnet(self.p7)

			self.box_subnet_out1 = self._box_subnet(self.merge1)
			self.box_subnet_out2 = self._box_subnet(self.merge2)
			self.box_subnet_out3 = self._box_subnet(self.merge3)
			self.box_subnet_out4 = self._box_subnet(self.p6)
			self.box_subnet_out5 = self._box_subnet(self.p7)

	def init_output(self):
		with tf.name_scope('output_classes'):
			self.y_out_classes = self.class_subnet_out1 + self.class_subnet_out2 + self.class_subnet_out3 + \
								 self.class_subnet_out4 + self.class_subnet_out5

		with tf.name_scope('output_boxes'):
			self.y_out_boxes = self.box_subnet_out1 + self.box_subnet_out2 + self.box_subnet_out3 + \
							   self.box_subnet_out4 + self.box_subnet_out5

		with tf.name_scope('output_decoded'):
			self.y_out_decoded = decode_netout([self.class_subnet_out1, self.class_subnet_out2,
			                                    self.class_subnet_out3, self.class_subnet_out4,
			                                    self.class_subnet_out5],
			                                   [self.box_subnet_out1, self.box_subnet_out2,
			                                    self.box_subnet_out3, self.box_subnet_out4,
			                                    self.box_subnet_out5])

		with tf.name_scope('loss'):
			self.loss = focal_loss(self.y_out_classes, self.y_classes) + \
						mean_squared_error(self.y_out_boxes, self.y_boxes)

		with tf.name_scope('metrics'):
			self.metrics.update_metrics(y_boxes, self.y_out_decoded)
			self.precision = metrics.precision
			self.recall = metrics.recall

		with tf.name_scope('train_op'):
			self.train_op = tf.train.GradientDescentOptimizer(self.flags.learning_rate).minimize(self.loss)

	def _base_subnet(self, input):
		with tf.variable_scope('conv_1_x'):
			conv1 = tf.layers.conv2d(input, 256, [3, 3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv1')
			conv1 = relu('relu1', conv1)
		with tf.variable_scope('conv_2_x'):
			conv2 = tf.layers.conv2d(conv1, 256, [3, 3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv2')
			conv2 = relu('relu2', conv2)
		with tf.variable_scope('conv_3_x'):
			conv3 = tf.layers.conv2d(conv2, 256, [3, 3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv3')
			conv3 = relu('relu3', conv3)
		with tf.variable_scope('conv_4_x'):
			conv4 = tf.layers.conv2d(conv3, 256, [3, 3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv4')
			conv4 = relu('relu4', conv4)
			return conv4

	def _class_subnet(self, input):
		with tf.variable_scope('class_subnet'):
			base_subnet_out = self._base_subnet(input)
			with tf.variable_scope('conv_5_x'):
				out = tf.layers.conv2d(base_subnet_out, self.number_of_classes * self.number_of_anchors, [3, 3], padding='SAME',
									   reuse=tf.AUTO_REUSE, name='conv5')
				return out

	def _box_subnet(self, input):
		with tf.variable_scope('box_subnet'):
			base_subnet_out = self._base_subnet(input)
			with tf.variable_scope('conv_5_x'):
				out = tf.layers.conv2d(base_subnet_out, self.number_of_anchors * 4, [3, 3], padding='SAME', reuse=tf.AUTO_REUSE,
									   name='conv5')
				return out


if __name__ == '__main__':
	config = {"learning_rate": 1.0e-3, "momentum": 0.99, "weight_decay": 0.00005, "log_interval": 2000, "batch_size": 16,
			  "number_of_classes": 9, "max_epoch": 10, "exp_dir": "Coco_exp_1", "model": "", "resume_training": 0,
			  "checkpoint": "best_model.ckpt", "x_train": "data.npy", "y_train": "labels.npy", "img_width": 512,
			  "img_height": 512, "num_channels": 3, "number_of_anchors": 9, "confidence_threshold": 0.05}
	config = Bunch(config)
	fpn = FPN(config)
	fpn.build()
