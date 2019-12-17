import argparse
from datetime import date
import os
import sys

from utils.compute_overlap import compute_overlap
from utils.visualization import draw_detections, draw_annotations
from utils.anchors import anchors_for_shape, anchor_targets_bbox

from model import efficientdet
import cv2
import os
import numpy as np
import time

import os
import glob
import json


def preprocess_image(image, image_size):
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size
    image = cv2.resize(image, (resized_width, resized_height))
    new_image = np.ones((image_size, image_size, 3), dtype=np.float32) * 128.
    offset_h = (image_size - resized_height) // 2
    offset_w = (image_size - resized_width) // 2
    new_image[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = image.astype(np.float32)
    new_image /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    new_image[..., 0] -= mean[0]
    new_image[..., 1] -= mean[1]
    new_image[..., 2] -= mean[2]
    new_image[..., 0] /= std[0]
    new_image[..., 1] /= std[1]
    new_image[..., 2] /= std[2]
    return new_image, scale, offset_h, offset_w


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_box_score(image_path, prediction_model, anchors, image_size, args=None):
    image = load_image(image_path)
    h, w = image.shape[:2]
    src_image = image[:, :, ::-1].copy()

    image, scale, offset_h, offset_w = preprocess_image(image, image_size)
    inputs = np.expand_dims(image, axis=0)

    # run network
    start = time.time()
    boxes, scores, labels = prediction_model.predict_on_batch([np.expand_dims(image, axis=0),
                                                               np.expand_dims(anchors, axis=0)])

    boxes[0, :, [0, 2]] = boxes[0, :, [0, 2]] - offset_w
    boxes[0, :, [1, 3]] = boxes[0, :, [1, 3]] - offset_h
    boxes /= scale
    boxes[0, :, 0] = np.clip(boxes[0, :, 0], 0, w - 1)
    boxes[0, :, 1] = np.clip(boxes[0, :, 1], 0, h - 1)
    boxes[0, :, 2] = np.clip(boxes[0, :, 2], 0, w - 1)
    boxes[0, :, 3] = np.clip(boxes[0, :, 3], 0, h - 1)

    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > args.score_threshold)[0]

    # select those scores
    scores = scores[0][indices]

    # find the order with which to sort the scores
    scores_sort = np.argsort(-scores)[:args.max_detections]

    # select detections
    # (n, 4)
    image_boxes = boxes[0, indices[scores_sort], :]
    # (n, )
    image_scores = scores[scores_sort]
    # (n, )
    image_labels = labels[0, indices[scores_sort]]
    # (n, 6)
    detections = np.concatenate(
        [image_boxes, np.expand_dims(image_scores, axis=1)], axis=1)

    return detections


def get_img_id(img_path):
    # Parse image path to image id using for evaluation
    img_name = os.path.basename(img_path)
    return int(img_name.split('.')[0])


def dummy_box_predict(img_path, prediction_model, anchors, image_size, args=None):
    prediction = []
    detections = get_box_score(img_path, prediction_model, anchors, image_size, args)
    for detection in detections:
        if len(detection) > 0:
            prediction.append(
                [float(detection[0]),
                 float(detection[1]),
                 float(detection[2]) - float(detection[0]),
                 float(detection[3]) - float(detection[1]),
                 float(detection[4])])
    return prediction


def predict(image_paths, prediction_model, anchors, image_size, args=None):
    detection_results = {}
    for img_path in image_paths:
        # Get image id
        img_id = get_img_id(img_path)

        # Load image and predict
        prediction = dummy_box_predict(img_path, prediction_model, anchors, image_size, args)

        # Save prediction with image id
        detection_results[img_id] = prediction

    return detection_results


def parse_prediction(detection_results, args=None):
    detections = []
    for img_id, prediction in detection_results.items():
        for bbox in prediction:
            x, y, w, h, score = bbox
            detections.append({
                "image_id": img_id,
                "category_id": args.ninedash_category_id,
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(score)
            })
    with open(args.target_path, 'w') as f:
        json.dump(detections, f)


def parse_args(args):
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--data-path', help='Data for prediction', type=str, required=True)
    parser.add_argument('--target-path', help='Target path', type=str, default='result.json')
    parser.add_argument('--split', help='Target path', type=str, default='val')
    parser.add_argument('--max-detections', help='Max detection', default=10)
    parser.add_argument('--ninedash-category-id', help='Ninedash category ID', default=1)
    parser.add_argument('--model-path', help='Model path of the network', type=str, required=True)
    parser.add_argument('--score-threshold', help='Minimum score threshold', type=float, default=0.3)
    parser.add_argument('--phi', help='Hyper parameter phi', default=0, type=int, choices=(0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--weighted-bifpn', help='Use weighted BiFPN', action='store_true')
    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument('--num-classes', help='Number of classes', default=1, type=int)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')

    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    image_paths = glob.glob('{}/*.jpg'.format(os.path.join(args.data_path, 'images', args.split)))

    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[args.phi]

    model, prediction_model = efficientdet(phi=args.phi,
                                           weighted_bifpn=args.weighted_bifpn,
                                           num_classes=args.num_classes,
                                           score_threshold=args.score_threshold)

    prediction_model.load_weights(args.model_path, by_name=True)

    anchors = anchors_for_shape((image_size, image_size))

    print('RUNNING PREDICTION...')
    detection_results = predict(image_paths, prediction_model, anchors, image_size, args)
    print('RUNNING PREDICTION... DONE')

    print('PARSING PREDICTION...')
    parse_prediction(detection_results, args)
    print('PARSING PREDICTION... DONE')


if __name__ == '__main__':
    main()
