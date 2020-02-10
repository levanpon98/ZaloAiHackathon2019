from augmentor.data_aug.data_aug import *
from augmentor.data_aug.bbox_util import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import argparse
from datetime import date
import pyximport

pyximport.install()


def get_img_id(img_path):
    # Parse image path to image id using for evaluation
    img_name = os.path.basename(img_path)
    return int(img_name.split('.')[0][4:])


def get_annotations(annotations, img_id):
    for a in annotations:
        if a['image_id'] == img_id:
            return a


def main():
    with open('/home/levanpon/Desktop/Data/robotic/train/annotations/train.json') as json_file:
        data = json.load(json_file)
        all_annotations_id = {'annotations': [], 'images': [], 'bbox': []}
        data_augmentation = data.copy()

        for a in data['annotations']:
            all_annotations_id['annotations'].append(a['image_id'])
            all_annotations_id['bbox'].append(a['bbox'])

        for a in data['images']:
            if a['id'] in all_annotations_id['annotations']:
                all_annotations_id['images'].append(a['file_name'])

        train_file = '/home/levanpon/Desktop/Data/robotic/train/'
        i = 0
        for a in data['images']:
            print(a)
            # if a['id'] in all_annotations_id['annotations']:
            #     temp = a
            #     img = cv2.imread(train_file + all_annotations_id['images'][i])[:, :, ::-1]
            #
            #     bboxes = all_annotations_id['bbox'][i]
            #     bboxes = np.expand_dims(bboxes, axis=0)
            #     bboxes = bboxes.astype(np.float32)
            #     img_id = get_img_id(all_annotations_id['images'][i])
            #     annotation = get_annotations(data['annotations'], img_id)
            #
            #     ##RandomHorizontalFlip
            #     img_, bboxes_ = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
            #     if len(bboxes_) > 0:
            #         # cv2.imwrite(train_file + str(img_id) + '_HorizontalFlip.jpg', img_)
            #         temp['file_name'] = str(img_id) + '_HorizontalFlip.jpg'
            #         temp['id'] = max_id
            #         max_id = max_id + 1
            #         annotation['bbox'] = list(bboxes_[0])
            #         data_augmentation['images'].append(temp)
            #         data_augmentation['annotations'].append(annotation)
            #
            #     # ##RandomScale
            #     img_, bboxes_ = RandomScale(0.3, diff=True)(img.copy(), bboxes.copy())
            #     if len(bboxes_) > 0:
            #         # cv2.imwrite(train_file + str(img_id) + '_RandomScale.jpg', img_)
            #         temp['file_name'] = str(img_id) + '_RandomScale.jpg'
            #         temp['id'] = max_id
            #         max_id = max_id + 1
            #
            #         annotation['bbox'] = list(bboxes_[0])
            #         data_augmentation['images'].append(temp)
            #         data_augmentation['annotations'].append(annotation)
            #
            #     img_, bboxes_ = RandomTranslate(0.3, diff=True)(img.copy(), bboxes.copy())
            #     if len(bboxes_) > 0:
            #         # cv2.imwrite(train_file + str(img_id) + '_RandomTranslate.jpg', img_)
            #         temp['file_name'] = str(img_id) + '_RandomTranslate.jpg'
            #         temp['id'] = max_id
            #         max_id = max_id + 1
            #         annotation['bbox'] = list(bboxes_[0])
            #         data_augmentation['images'].append(temp)
            #         data_augmentation['annotations'].append(annotation)
            #     #
            #     img_, bboxes_ = RandomRotate(20)(img.copy(), bboxes.copy())
            #     if len(bboxes_) > 0:
            #         # cv2.imwrite(train_file + str(img_id) + '_RandomRotate.jpg', img_)
            #         temp['file_name'] = str(img_id) + '_RandomRotate.jpg'
            #         temp['id'] = max_id
            #         max_id = max_id + 1
            #
            #         annotation['bbox'] = list(bboxes_[0])
            #         data_augmentation['images'].append(temp)
            #         data_augmentation['annotations'].append(annotation)
            #
            #     img_, bboxes_ = RandomShear(0.2)(img.copy(), bboxes.copy())
            #     if len(bboxes_) > 0:
            #         # cv2.imwrite(train_file + str(img_id) + '_RandomShear.jpg', img_)
            #         temp['file_name'] = str(img_id) + '_RandomShear.jpg'
            #         temp['id'] = max_id
            #         max_id = max_id + 1
            #
            #         annotation['bbox'] = list(bboxes_[0])
            #         data_augmentation['images'].append(temp)
            #         data_augmentation['annotations'].append(annotation)
            #
            #     img_, bboxes_ = RandomHSV(100, 100, 100)(img.copy(), bboxes.copy())
            #     if len(bboxes_) > 0:
            #         # cv2.imwrite(train_file + str(img_id) + '_RandomHSV.jpg', img_)
            #         temp['file_name'] = str(img_id) + '_RandomHSV.jpg'
            #         temp['id'] = max_id
            #         max_id = max_id + 1
            #
            #         annotation['bbox'] = list(bboxes_[0])
            #         data_augmentation['images'].append(temp)
            #         data_augmentation['annotations'].append(annotation)
            #     i += 1

        # data = data_augmentation
        #
        # with open('train.json', 'w') as outfile:
        #     json.dump(str(data), outfile)


if __name__ == '__main__':
    main()
