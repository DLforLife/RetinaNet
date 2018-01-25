"""
Defintions for all the layers / sub-modules that will be used in our models
"""
import tensorflow as tf

def merge_block(name, upstream, downstream):
    upsampled = tf.image.resize_images(downstream, upstream.get_shape()[1:3],
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)
    conv1x1 = conv(name, upstream, num_filters=downstream.shape[-1], kernel_size=(1, 1))
    merged = tf.add(upsampled, conv1x1, name)
    return merged

def residual_block(name, x, filters, strides=1, three_stage=True, is_training=True):
    print('Building residual unit: %s' % name)
    with tf.variable_scope(name):
        # get input channels
        in_channel = x.shape.as_list()[-1]
        # Shortcut connection
        if (not in_channel == filters[-1]):
            shortcut = conv('shortcut_conv', x,
                                  num_filters=filters[-1], kernel_size=(1, 1), stride=(strides, strides))
        elif (strides == 1):
            shortcut = tf.identity(x)
        else:
            shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')

        # Residual
        if (three_stage):
            filters1, filters2, filters3 = filters

            x = conv('conv_1', x,
                           num_filters=filters1, kernel_size=(1, 1), stride=(strides, strides))
            x = bn('bn_1', x, is_training)
            x = relu('relu_1', x)
            x = conv('conv_2', x,
                           num_filters=filters2, kernel_size=(3, 3))
            x = bn('bn_2', x, is_training)
            x = relu('relu_2', x)
            x = conv('conv_3', x,
                           num_filters=filters3, kernel_size=(1, 1), stride=(1, 1))
            x = bn('bn_3', x, is_training)
            # Merge
            x = x + shortcut
            x = relu('relu_3', x)

        else:
            filters1, filters2 = filters

            x = conv('conv_1', x,
                           num_filters=filters1, kernel_size=(3, 3), stride=(strides, strides))
            x = bn('bn_1', x, is_training)
            x = relu('relu_1', x)
            x = conv('conv_2', x,
                           num_filters=filters2, kernel_size=(3, 3))
            x = bn('bn_2', x, is_training)

            # Merge
            x = x + shortcut
            x = relu('relu_2', x)

        print('residual-unit-%s-shape: ' % name + str(x.shape.as_list()))

        return x

def bn(name, x, is_training):
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(x, training=is_training)

def conv(name, x, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
          initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]
        w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)
        # variable_summaries(w)
        conv = tf.nn.conv2d(x, w, stride, padding)
        return conv

def softmax(name, x, dim):
	with tf.variable_scope(name):
		return tf.nn.softmax(x, dim, name)

def sigmoid(name, x):
    with tf.variable_scope(name):
        return tf.nn.sigmoid(x)

def relu(name, x):
    with tf.variable_scope(name):
        return tf.nn.relu(x)

def variable_with_weight_decay(kernel_shape, initializer, wd, trainable=True):
    w = tf.get_variable('w_conv', kernel_shape, tf.float32, initializer=initializer, trainable=trainable)
    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(
            tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    #variable_summaries(w)
    return w