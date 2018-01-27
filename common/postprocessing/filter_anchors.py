"""
Performs anchor filtering based on their IOU
- non-maximum suppression with a threshold of 0.5
boxes : nx4, scores: nx1
"""
import numpy as np
def Intersection_over_union(box_one, box_two):
    """
    Calculates intersection over union between 2 boxes
    Args:
    - box_one (list): (xc,yc,w,h,score)
    - box_two (list): (xc,yc,w,h,score)
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

def non_max_suppression(box_detections, class_scores, threshold=0.5):
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
    all_detections = np.concatenate([box_detections, class_scores], axis=1)
    all_detections = all_detections[all_detections[:,4].argsort()[::-1]]

    final_detection_list = []
    for bbox in all_detections:
        for final_bbox in final_detection_list:
            iou = Intersection_over_union(final_bbox, bbox)
            if iou > threshold:
                break
        else:
            final_detection_list.append(bbox)

    return final_detection_list

if __name__ == "__main__":
    detections = np.array([(11, 11, 24, 24), (10, 11, 20, 20), (11, 9, 24, 24), (40, 42, 20, 20)])
    scores = np.array([0.75, 0.8, 0.7, 0.6])
    scores = np.expand_dims(scores, -1)
    final_detections = non_max_suppression(detections,scores)
    print ('Final Detections: ' + ', '.join(map(str, final_detections)))
