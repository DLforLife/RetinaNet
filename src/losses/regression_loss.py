"""
Implementation for smooth l1 loss for box regression
Same one as the paper and Fast-RCNN too
Fast R-CNN: https://arxiv.org/pdf/1504.08083.pdf
"""
def smooth_l1_loss(logits, targets, delta=1.0, sigma=1.0):
    """
    Computes the standard smooth_l1_loss
    Args:
    - logits(tensor): predicted anchors of shape [num_anchors, 4]
    - targets(tensor): ground truth anchors of shape [num_anchors, 4]
    - delta(scalar): the value where the function changes from quadratic to linear
    Returns:
    - loss(float):
    """
    diff = targets - logits
    abs_diff = tf.abs(diff)
    smooth_l1_loss = tf.where(tf.less(abs_diff, delta), 0.5 * tf.square(abs_diff), abs_diff - 0.5)
    loss = tf.reduce_sum(smooth_l1_loss, 2)
    loss = tf.reduce_sum(loss, 1)
    return loss

def huber_loss(logits, targets, weights=1.0, delta=1.0):
    """
    Computes the huber loss, the same as the smooth L1 loss
    Args:
    - logits(tensor): predicted anchors of shape [num_anchors, 4]
    - targets(tensor): ground truth anchors of shape [num_anchors, 4]
    - weights(scalar or tensor): a weighting coefficient for the loss. Can be scalar, or have the shape of a batch size or the same shape as logits
    - delta(scalar): the value where the function changes from quadratic to linear
    Returns:
    - loss(float):
    """
    loss = tf.losses.huber_loss(labels=targets, predictions=logits, weights=weights, delta=delta)
    return loss
