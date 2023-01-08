import os
import json
import cv2

import numpy as np 

from constants import *

def load_label_info():
    f = open(os.path.join(ANNO_DIR, 'labels.json'))
    labels = json.load(f)
    label_dict = {}
    categories = []
    for idx, label in enumerate(labels):
        class_id = idx + 1
        label_dict[label['id']] = class_id    # Mapping labelId to idx as class idx
        categories.append({
          'id': class_id,
          'name': label['label'],
        })
    return label_dict, categories

def selectLongestContour(contours):
    if (len(contours) == 1):
        return np.squeeze(contours)
    longest_contour = contours[0]
    for contour in contours:
        if len(longest_contour) < len(contour):
            longest_contour = contour
    return np.squeeze(longest_contour)

def fillMaskWithBoundary(im):
    try:
        im = np.asarray(im, dtype='uint8')[:,:,0]
        ret, thresh = cv2.threshold(im, 128, 255, 0)
        filled = np.zeros_like(im)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # contours = contours[0] if len(contours) != 1 else contours
        contour = selectLongestContour(contours)
        assert len(contour) > 10
        bbox = cv2.boundingRect(contour)
        
        cv2.drawContours(filled, [contour], 0, 255, -1)
        return filled, bbox, contour.tolist()
    except:
        print(len(contours))
        print(len(contour))
        print(contours)