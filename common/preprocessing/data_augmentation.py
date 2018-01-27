"""
Performs Data augmentation on the input data
Section 4.1: optimization
"We use horizontal image flipping as the only form of data augmentation unless otherwise noted."
"""
import tensorflow as tf
import numpy as np

class data_augmentation:
    def __init__(self, images, boxes):
        """
        Initializing data augmentation class
        Args:
        - image(Tensor): batch image of shape (BxWxHxC)
        - box(Tensor): batch box of shape (BxNx4)
        Returns:
        """
        self.images = images
        self.boxes = boxes

    def random_horizontal_flip(self, image, boxes, flip_prob=0.5):
        """
        Performs random horizontal flipping to the image and/or box
        Args:
        - image: image data of shape (WxHxC)
        - boxes: ground truth boxes in image (Nx4)
        - flip_prob (float): the probability of flipping the image
        Returns:
        - flipped_image
        - flipped_boxes
        """
        with tf.name_scope('RandomHorizontalFlip', values=[image, boxes]):
            img_w = tf.get_shape(image)[0]
            # random variable defining whether to do flip or not
            do_a_flip_random = tf.random_uniform([], seed=seed)
            # flip only if there are bounding boxes in the image
            do_a_flip_random = tf.logical_and(tf.greater(tf.size(boxes), 0), tf.greater(do_a_flip_random, flip_prob))
            # flip image
            flipped_image = tf.cond(do_a_flip_random, lambda: self.flip_image_horizontally(image), lambda: image)
            # flip box
            flipped_boxes = tf.cond(do_a_flip_random, lambda: self.flip_box_horizontally(boxes,img_w), lambda: boxes, img_w)
            return flipped_image, flipped_boxes

     def flip_image_horizontally(self, image):
        """
        flip image
        """
        flipped_image = tf.image.flip_left_right(image)
        return flipped_image

    def flip_box_horizontally(self, boxes):
        """
        flip box
        """
        x, y, w, h = tf.split(value=boxes, num_or_size_splits=4, axis=1)
        flipped_x = tf.subract(img_w - x)
        flipped_box = tf.concat([flipped_x, y, w, h], 1)
        return flipped_box

    def random_vertical_flip(self, flip_prob=0.5):
        """
        Performs random vertical flipping to the image and/or box
        Args:
        - image: image data of shape (WxHxC)
        - boxes: ground truth boxes in image (Nx4)
        - flip_prob (float): the probability of flipping the image
        Returns:
        - flipped_image
        - flipped_boxes
        """
        with tf.name_scope('RandomVerticalFlip', values=[image, boxes]):
            img_h = tf.get_shape(image)[1]
            # random variable defining whether to do flip or not
            do_a_flip_random = tf.random_uniform([], seed=seed)
            # flip only if there are bounding boxes in the image
            do_a_flip_random = tf.logical_and(tf.greater(tf.size(boxes), 0), tf.greater(do_a_flip_random, flip_prob))
            # flip image
            flipped_image = tf.cond(do_a_flip_random, lambda: self.flip_image_vertically(image), lambda: image)
            # flip box
            flipped_boxes = tf.cond(do_a_flip_random, lambda: self.flip_box_vertically(boxes,img_h), lambda: boxes, img_h)
            return flipped_image, flipped_boxes

     def flip_image_vertically(self,image):
        """
        flip image
        """
        flipped_image = tf.image.flip_up_down(image)
        return flipped_image

    def flip_box_vertically(self, boxes, img_h):
        """
        flip box
        """
        x, y, w, h = tf.split(value=boxes, num_or_size_splits=4, axis=1)
        flipped_y = tf.subract(img_wh - y)
        flipped_box = tf.concat([x, flipped_y, w, h], 1)
        return flipped_box
