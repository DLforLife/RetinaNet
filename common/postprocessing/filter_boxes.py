"""
Performs anchor filtering based on their IOU
- non-maximum suppression with a threshold of 0.5
"""
import numpy as np
import tensorflow as tf
def decode_highest_scores(detector_confidence=0.05):
    """
    To improve speed, we only decode box predictions from at most
    1k top-scoring predictions per FPN level, after thresholding detector confidence at 0.05
    section 4.1: inference
    """

def tf_non_max_suppression(pred_boxes, pred_scores, max_output_size=256, iou_threshold=0.5, name="nms"):
    """
    Perfoms non max suppresion based on Tensorflow's implementation
    Args:
    - pred_boxes (tensor): predicted boxes
    - pred_scores (tensor): predicted scores
    - max_output_size (int): max number of boxes to choose
    - iou_threshold (float):
    Returns:
    - final_detection_list: filtered detected box list
    """
    if not 0 <= iou_threshold <= 1.0:
        raise ValueError('iou_thresh must be between 0 and 1')
    final_detected_indices=tf.image.non_max_suppression(boxes=pred_boxes,\
                                                        scores=pred_scores,\
                                                        max_output_size=max_output_size,\
                                                        iou_threshold=iou_threshold,\
                                                        name=name)

    return tf.gather(pred_boxes, final_detected_indices)

def non_max_suppression(pred_boxes, pred_scores, threshold=0.5):
    """
    Perfoms non max suppresion to choose the max score detections based on IOU
    Args:
    - box_detections (numpy array):
    - class_scores (numpy array):
    - threshold (float):
    Returns:
    - final_detection_list (numpy array):
    """
    print ("Calculating non-max suppression")
    pred_scores = np.expand_dims(pred_scores, -1)
    all_detections = np.concatenate([pred_boxes, pred_scores], axis=1)
    all_detections = all_detections[all_detections[:,4].argsort()[::-1]]

    final_detection_list = []
    for bbox in all_detections:
        for final_bbox in final_detection_list:
            iou = Intersection_over_union(final_bbox, bbox)
            if iou > threshold:
                break
        else:
            final_detection_list.append(bbox)
    final_detection_list = np.array(final_detection_list)
    return final_detection_list[:,0:4]

def Intersection_over_union(box_one, box_two):
    """
    Calculates intersection over union between 2 boxes
    Args:
    - box_one (list): (xc,yc,w,h,score)
    - box_two (list): (xc,yc,w,h)
    Returns:
    - iou (float): iou value
    """
    left_one = box_one[0] - (box_one[2]/2.0)
    top_one = box_one[1] + (box_one[3]/2.0)

    right_one = box_one[0] + (box_one[2]/2.0)
    bottom_one = box_one[1] - (box_one[3]/2.0)

    left_two = box_two[0] - (box_two[2]/2.0)
    top_two = box_two[1] + (box_two[3]/2.0)

    right_two = box_two[0] + (box_two[2]/2.0)
    bottom_two = box_two[1] - (box_two[3]/2.0)

    # Revisit this part
    intersected_w = max(0, min(right_one,right_two) - max(left_one,left_two))
    intersected_h = max(0, min(top_one,top_two) - max(bottom_one,bottom_two))

    intersected_area = intersected_h * intersected_w
    area_one = box_one[2] * box_one[3]
    area_two = box_two[2] * box_two[3]

    union_area = area_one + area_two - intersected_area
    iou = intersected_area/float(union_area)
    return iou
"""
For testing..
"""
if __name__ == "__main__":
    box_pred = tf.convert_to_tensor([(11.0, 11.0, 24.0, 24.0), (10.0, 11.0, 20.0, 20.0), (11.0, 9.0, 24.0, 24.0), (40.0, 42.0, 20.0, 20.0)])
    scores = tf.convert_to_tensor([0.75, 0.8, 0.7, 0.6])
    print (scores.get_shape())
    print (box_pred.get_shape())
    final_detections = tf.Session().run(tf_non_max_suppression(box_pred, scores))
    print ('Final Detections: ' + ', '.join(map(str, final_detections)))
