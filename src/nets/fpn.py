"""
An implementation for the feature pyramid network class.
"""
import tensorflow as tf

class FPN:
    def __init__(self, config):
        self.config = config
        #########################################
        self.config.img_size = self.config.img_width * self.config.img_height
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
        self.train_accuracy = None
        self.segmented_summary = None
        self.merged_summaries = None
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
        self.init_network()
        self.init_output()

    def init_input(self):
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [self.config.batch_size,
                                                 self.config.img_height,
                                                 self.config.img_width,
                                                 self.config.num_channels])
            self.y_classes = tf.placeholder(tf.int64, [self.config.batch_size,
                                                       self.config.img_height,
                                                       self.config.img_width,
                                                       self.config.number_of_anchors,
                                                       self.config.number_of_classes])
            self.y_boxes = tf.placeholder(tf.int64, [self.config.batch_size,
                                                     self.config.img_height,
                                                     self.config.img_width,
                                                     self.config.number_of_anchors, 4])
            self.is_training = tf.placeholder(tf.bool)

    def init_network(self):
        with tf.variable_scope('buttom_up_pathway'):
            with tf.variable_scope('conv_1_x'):
                self.conv1 = self._conv('conv1', self.x, num_filters=3)
                self.conv1 = self._relu('relu1', self.conv1)
            with tf.variable_scope('conv_2_x'):
                self.conv2 = self._conv('conv2', self.conv1, num_filters=16)
                self.conv2 = self._relu('relu2', self.conv2)
            with tf.variable_scope('conv_3_x'):
                self.conv3 = self._conv('conv3', self.conv2, num_filters=16)
                self.conv3 = self._relu('relu3', self.conv3)
        with tf.variable_scope('top_down_pathway'):
            with tf.variable_scope('merge_1'):
                self.merge1 = self._merge_block(self.conv2, self.conv3)
            self.class_subnet1 = self._class_subnet(self.merge1)
            with tf.variable_scope('merge2'):
                self.merge2 = self._merge_block(self.conv1, self.merge1)
            self.class_subnet2 = self._class_subnet(self.merge2)


    def init_output(self):
        with tf.name_scope('output_classes'):
            self.y_out_classes = self.class_subnet1 + self.class_subnet2
        with tf.name_scope('output_boxes'):
            self.y_out_boxes =

    def _merge_block(self, name, upstream, downstream):
        upsampled = tf.image.resize_images(downstream, upstream.get_shape(), method=ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)
        conv1x1 =  self._conv(name, upstream, num_filters=downstream.shape[-1], kernel_size=(1,1))
        merged = tf.add(upsampled, conv1x1, name)
        return merged

    def _class_subnet(self, input):
        with tf.variable_scope('class_subnet'):
            with tf.variable_scope('conv_1_x'):
                conv1 = tf.layers.conv2d(input, 256, [3, 3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv1')
                conv1 = self._relu('relu1', conv1)
            with tf.variable_scope('conv_2_x'):
                conv2 = tf.layers.conv2d(conv1, 256, [3, 3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv2')
                conv2 = self._relu('relu2', conv1)
            with tf.variable_scope('conv_3_x'):
                conv3 = tf.layers.conv2d(conv2, 256, [3, 3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv3')
                conv3 = self._relu('relu3', conv1)
            with tf.variable_scope('conv_4_x'):
                conv4 = tf.layers.conv2d(conv3, 256, [3, 3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv4')
                conv4 = self._relu('relu4', conv1)
            with tf.variable_scope('conv_5_x'):
                conv5 = tf.layers.conv2d(conv4, self.y_classes * self.y_boxes, [3, 3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv5')
                conv5 = self._sigmoid('sigmoid5', conv5)
                return conv5

    def _box_subnet(self, input):
        raise NotImplementedError("box subnet not implemented")

    @staticmethod
    def _conv(name, x, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
              initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0):

        with tf.variable_scope(name):
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]
            w = _variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            variable_summaries(w)
            conv = tf.nn.conv2d(x, w, stride, padding)
            return conv

    @staticmethod
    def _sigmoid(name, x):
        with tf.variable_scope(name):
            return tf.nn.sigmoid(x)
    @staticmethod
    def _relu(name, x):
        with tf.variable_scope(name):
            return tf.nn.relu(x)


