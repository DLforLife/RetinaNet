"""
Implementation for focal loss
"""
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

class FocalLoss:
    def __init__(self, pred, labels, num_classes, gamma = 2.0, alpha = 0.25, weighted=True):
        """
        Initialzing the parameter for focal loss function
        Args:
        - pred (tensor): class predictions of shape [Batch_size, num_classes]
        - onehot_labels (tensor): ground truth one hot lables of shape [Batch_size, num_classes]
        - gamma (float): focusing factor
        - alpha (float): balancing factor
        - weighted (boolean): to specify if Alpha is constant or not
        Returns:
        - loss
        """
        self.pred = pred
        self.num_classes = num_classes
        self.onehot_labels = tf.one_hot(labels, self.num_classes)
        self.gamma = gamma
        self.alpha = alpha
        self.weighted = weighted
        self.epsilon = 1e-7

    def calculate(self):
        """
        Computing the focal loss
        Args:
        Returns:
        """
        self.pred = tf.nn.softmax(self.pred)
        pred_t = self.transform_to_t(self.pred, self.onehot_labels)

        if self.weighted:
            alpha_t = tf.scalar_mul(self.alpha, tf.ones_like(self.onehot_labels, dtype=tf.float32))
            alpha_t = self.transform_to_t(alpha_t, self.onehot_labels)
        else:
            alpha_t = self.alpha

        loss = -1 * alpha_t * tf.pow(pred_t, self.gamma) * tf.log(pred_t + self.epsilon)
        loss = tf.reduce_mean(loss,axis=1)
        loss = tf.reduce_mean(loss)
        return loss

    def transform_to_t(self, x, onehot_labels):
        """
        Transforming values based on the onehot_labels
        Args:
        - x (tensor):
        - onehot_labels (tensor):
        Returns:
        - xt (tensor): a transformed x
        """
        xt = tf.where(tf.equal(onehot_labels, 1), x, 1. - x)
        return xt
