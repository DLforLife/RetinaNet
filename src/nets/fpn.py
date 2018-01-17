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
            x = self._conv('conv_7x7', self.x, 64, kernel_size=(7,7), stride=(2,2))
            self.res_stage1_out = tf.nn.max_pool(x, [3,3], strides=[2,2], padding='VALID')

        with tf.variable_scope('stage_2'):
            x = self._residual_block('res_block_1', self.res_stage1_out, filters=[64, 64, 256], three_stage=True)
            x = self._residual_block('res_block_2', x, filters=[64, 64, 256], three_stage=True)
            self.res_stage2_out = self._residual_block('res_block_3', x, filters=[64, 64, 256], three_stage=True)

        with tf.variable_scope('stage_3'):
            x = self._residual_block('res_block_1', self.res_stage2_out, filters=[128, 128, 512], three_stage=True)
            x = self._residual_block('res_block_2', x, filters=[128, 128, 512], three_stage=True)
            x = self._residual_block('res_block_3', x, filters=[128, 128, 512], three_stage=True)
            self.res_stage3_out = self._residual_block('res_block_4', x, filters=[128, 128, 512], three_stage=True)

        with tf.variable_scope('stage_4'):
            x = self._residual_block('res_block_1', self.res_stage3_out, filters=[256, 256, 1024], three_stage=True)
            x = self._residual_block('res_block_2', x, filters=[256, 256, 1024], three_stage=True)
            x = self._residual_block('res_block_3', x, filters=[256, 256, 1024], three_stage=True)
            x = self._residual_block('res_block_5', x, filters=[256, 256, 1024], three_stage=True)
            x = self._residual_block('res_block_6', x, filters=[256, 256, 1024], three_stage=True)
            self.res_stage4_out = self._residual_block('res_block_7', x, filters=[256, 256, 1024], three_stage=True)

        with tf.variable_scope('stage_5'):
            x = self._residual_block('res_block_1', self.res_stage4_out, filters=[512, 512, 2048], three_stage=True)
            x = self._residual_block('res_block_2', x, filters=[512, 512, 2048], three_stage=True)
            self.res_stage5_out = self._residual_block('res_block_3', x, filters=[512, 512, 2048], three_stage=True)

    def init_network(self):
        with tf.variable_scope('top_down_pathway'):
            with tf.variable_scope('p5'):
                self.merge1 = self._merge_block(self.res_stage4_out, self.res_stage5_out)
            self.class_subnet1 = self._class_subnet(self.merge1)
            with tf.variable_scope('p4'):
                self.merge2 = self._merge_block(self.res_stage3_out, self.merge1)
            self.class_subnet2 = self._class_subnet(self.merge2)
            with tf.variable_scope('p3'):
                self.merge3 = self._merge_block(self.res_stage2_out, self.merge2)
            self.class_subnet3 = self._class_subnet(self.merge3)
            with tf.variable_scope('p6'):
                self.p6 = self._conv('conv', self.res_stage5_out, 256, (3,3), stride=(2,2))
            self.class_subnet2 = self._class_subnet(self.p6)
            with tf.variable_scope('p7'):
                self.p7 = self._relu('relu', self.p6)
                self.p7 = self._conv('conv', self.res_stage5_out, 256, (3,3), stride=(2,2))
            self.class_subnet2 = self._class_subnet(self.p7)


    def init_output(self):
        with tf.name_scope('output_classes'):
            self.y_out_classes = self.class_subnet1 + self.class_subnet2
        with tf.name_scope('output_boxes'):
            self.y_out_boxes = self.box_subnet1 + self.box_subnet2

    def _class_subnet(self, input):
        with tf.variable_scope('class_subnet'):
            with tf.variable_scope('conv_1_x'):
                conv1 = tf.layers.conv2d(input, 256, [3, 3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv1')
                conv1 = self._relu('relu1', conv1)
            with tf.variable_scope('conv_2_x'):
                conv2 = tf.layers.conv2d(conv1, 256, [3, 3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv2')
                conv2 = self._relu('relu2', conv2)
            with tf.variable_scope('conv_3_x'):
                conv3 = tf.layers.conv2d(conv2, 256, [3, 3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv3')
                conv3 = self._relu('relu3', conv3)
            with tf.variable_scope('conv_4_x'):
                conv4 = tf.layers.conv2d(conv3, 256, [3, 3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv4')
                conv4 = self._relu('relu4', conv4)
            with tf.variable_scope('conv_5_x'):
                conv5 = tf.layers.conv2d(conv4, self.y_classes * self.y_boxes, [3, 3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv5')
                conv5 = self._sigmoid('sigmoid5', conv5)
                return conv5

    def _box_subnet(self, input):
        raise NotImplementedError("box subnet not implemented")

    def _merge_block(self, name, upstream, downstream):
        upsampled = tf.image.resize_images(downstream, upstream.get_shape(), method=ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)
        conv1x1 =  self._conv(name, upstream, num_filters=downstream.shape[-1], kernel_size=(1,1))
        merged = tf.add(upsampled, conv1x1, name)
        return merged

    def _residual_block(self, name, x, filters, pool_first=False, strides=1, three_stage=False):
        print('Building residual unit: %s' % name)
        with tf.variable_scope(name):
            # get input channels
            in_channel = x.shape.as_list()[-1]

            # Shortcut connection
            shortcut = tf.identity(x)
            filters1, filters2, filters3 = filters
            if pool_first:
                if in_channel == filters:
                    if strides == 1:
                        shortcut = tf.identity(x)
                    else:
                        shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
                else:
                    shortcut = self._conv('shortcut_conv', x,
                                          num_filters=filters, kernel_size=(1, 1), stride=(strides, strides))

            # Residual
            if(three_stage):
                x = self._conv('conv_1', x,
                               num_filters=filters1, kernel_size=(1, 1), stride=(strides, strides))
                x = self._bn('bn_1', x)
                x = self._relu('relu_1', x)
                x = self._conv('conv_2', x,
                               num_filters=filters2, kernel_size=(3, 3))
                x = self._bn('bn_2', x)
                x = self._relu('relu_2', x)
                x = self._conv('conv_3', x,
                               num_filters=filters3, kernel_size=(1, 1), stride=(strides, strides))
                x = self._bn('bn_3', x)
                # Merge
                x = x + shortcut
                x = self._relu('relu_3', x)

            else:
                x = self._conv('conv_1', x,
                               num_filters=filters1, kernel_size=(3, 3), stride=(strides, strides))
                x = self._bn('bn_1', x)
                x = self._relu('relu_1', x)
                x = self._conv('conv_2', x,
                               num_filters=filters2, kernel_size=(3, 3))
                x = self._bn('bn_2', x)

                # Merge
                x = x + shortcut
                x = self._relu('relu_2', x)

            print('residual-unit-%s-shape: ' % name + str(x.shape.as_list()))

            return x

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


