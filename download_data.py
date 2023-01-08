from io import BytesIO
import json
import cv2
import numpy as np
import os
import requests
from tqdm import tqdm
from PIL import Image

from constants import *
from utils import *


def download_and_save(url, save_path, refetch = False, is_save = True):
    if (os.path.exists(save_path) and not refetch):
        return Image.open(save_path).convert('RGB')
    r = requests.get(url, allow_redirects=True)
    img = Image.open(BytesIO(r.content)).convert('RGB')
    if (is_save):
        img.save(save_path)
    return img


if __name__ == "__main__":
    
    coco_anno = {
        'images': [],
        'categories': [],
        'annotations': [],
    }

    # Load label info 
    label_dict, categories = load_label_info()

    list_annos = os.listdir(ANNO_DIR)

    images = []
    annotations = []

    anno_idx = 0

    for img_idx, anno_path in enumerate(tqdm(list_annos)):
        
        # Ignore label info file
        if (anno_path == 'labels.json'):
            continue

        fname = os.path.join(ANNO_DIR, anno_path)
        with open(fname, 'r') as f:
            image_info = json.load(f)
        
        # # Ignore undone data
        # if (image_info['annotateStatus'] != 1):
        #     continue

        # # Skip image w/o annotation
        # if (len(image_info['annotations']) == 0):
        #     continue
        
        # Get image
        os.makedirs(IMAGE_DIR, exist_ok=True)
        image_path = os.path.join(
            IMAGE_DIR, image_info['name'])
        download_and_save(image_info['image']['original']['URL'], image_path)

        images.append({
            "id": img_idx,
            "file_name": image_info['name'],
            "height": image_info['height'],
            "width": image_info['width']
        })


        image_mask_dir = os.path.join(
            MASK_DIR, image_info['name'].split('.')[0])
    

        # Create mapping from annotation obj to label idx
        annoObjDict = {}
        for annotationObj in image_info['annotationObjects']:
            try:
                annoObjDict[annotationObj['id']] = label_dict[annotationObj['labelId']]
            except:
                print(anno_path)
                print(annotationObj['id'])
                print(annotationObj['labelId'])

        # Download mask
        for annotation in image_info['annotations']:
            if annotation['type'] != 'MASK':
                continue
            try:

                ext = '.' + annotation['maskData']['mask']['URL'].split('.')[-1]
                label_idx = str(annoObjDict[annotation['annotationObjectId']])
                
                os.makedirs(os.path.join(image_mask_dir, label_idx), exist_ok=True)
                mask_path = os.path.join(
                    image_mask_dir, label_idx, annotation['id'] + ext)
                mask = download_and_save(annotation['maskData']['mask']['URL'], mask_path, refetch=True, is_save=False)
                
                mask = mask.resize((image_info['width'], image_info['height']), Image.Resampling.NEAREST)
                filled, bbox, polygons = fillMaskWithBoundary(mask)
                # cv2.imwrite(mask_path, filled)

                annotations.append({
                    'id': anno_idx,
                    'category_id': int(label_idx),
                    'image_id': img_idx,
                    "segmentation": polygons,
                    'bbox': list(bbox)
                })
                anno_idx += 1
                
            except Exception as e:
                print(e)
                print(image_info['name'])
                pass
    coco_anno = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }

    with open(f'{DATA_DIR}/annotation.json', 'w') as fw:
        json.dump(coco_anno, fw)