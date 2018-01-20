"""
Implementation for 2D cross entropy
"""
import tensorflow as tf
class CrossEntropy:
    def __init__(self, logits, labels):
        """
        Initialzing the parameters for 2D Cross entropy
        Args:
        - logits(Tensor): class predictions of shape [Batch_size, num_classes]
        - one_hot_labels (Tensor): ground truth one hot lables of shape [Batch_size, num_classes]
        Returns:
        - loss
        """
        self.logits = logits
        self.labels = labels

    def calculate(self):
        """
        Computing the focal loss
        Args:
        Returns:
        """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        loss = tf.reduce_mean(loss)
        return loss
