"""
Defintions for all the layers / sub-modules that will be used in our models
"""
import tensorflow as tf

def _variable_with_weight_decay(kernel_shape, initializer, wd, trainable=True):
    w = tf.get_variable('w_conv', kernel_shape, tf.float32, initializer=initializer, trainable=trainable)
    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(
            tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    #variable_summaries(w)
    return w