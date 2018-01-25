"""
Implementation of the multi-task loss
"""
import tensorflow as tf
from class_loss import *
from regression_loss import *

def multi_task_loss(pred_classes, pred_boxes, gt_labels, gt_anchors, num_classes, weights=None):
    """
    Calculating the loss for both class predictions and box regression, using s
    """
    print ("Multi task loss...")

    sess = tf.Session()
    focal_loss = sess.run(FocalLoss(pred=pred_classes, labels=gt_labels, num_classes=num_classes).calculate())
    box_loss = sess.run(huber_loss(pred=pred_boxes, targets=gt_anchors))
    print ("Focal loss:", focal_loss)
    print ("Bbox loss:", box_loss)
    total_loss = sess.run(tf.add(focal_loss, box_loss))
    sess.close()
    return total_loss

"""
For testing the loss functions
"""
if __name__ == '__main__':
    logits = tf.convert_to_tensor([[0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2]])
    num_classes = 4
    labels = tf.constant([1, 2])
    bbox = tf.ones_like(logits)
    loss = multi_task_loss(pred_classes=logits, pred_boxes=logits, gt_labels=labels, gt_anchors=bbox,num_classes=num_classes)
    print ("Loss", loss)
