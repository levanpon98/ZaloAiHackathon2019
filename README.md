# Nine Dash Detection
This is an implementation of [EfficientDet](https://arxiv.org/pdf/1911.09070.pdf) for object detection on Keras and Tensorflow. The project is based on [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)
and the [qubvel/efficientnet](https://github.com/qubvel/efficientnet). 
The pretrained EfficientNet weights files are downloaded from [Callidior/keras-applications/releases](https://github.com/Callidior/keras-applications/releases)

Thanks for their hard work.
This project is released under the Apache License. Please take their licenses into consideration too when use this project.

## Train
* STEP1: `python3 train.py --snapshot imagenet --phi {0, 1, 2, 3, 4, 5, 6} --gpu 0 --random-transform --freeze-backbone --batch-size 32 --steps 1000 coco path/to/dataset` to start training. The init lr is 1e-3.
* STEP2: `python3 train.py --snapshot xxx.h5 --phi {0, 1, 2, 3, 4, 5, 6} --gpu 0 --random-transform --compute-val-loss --freeze-bn --batch-size 4 --steps 10000 coco path/to/dataset` to start training when val mAP can not increase during STEP1. The init lr is 1e-4 and decays to 1e-5 when val mAP keeps dropping down.
* Optional arguments:
    - --snapshot SNAPSHOT   Resume training from a snapshot.
    - --freeze-backbone     Freeze training of backbone layers.
    - --freeze-bn           Freeze training of BatchNormalization layers.
    - --weighted-bifpn      Use weighted BiFPN
    - --batch-size BATCH_SIZE Size of the batches.
    - --phi {0,1,2,3,4,5,6}  Hyper parameter phi
    - --gpu GPU             Id of the GPU to use (as reported by nvidia-smi).
    - --num_gpus NUM_GPUS   Number of GPUs to use for parallel processing.
    - --multi-gpu-force     Extra flag needed to enable (experimental) multi-gpu support.
    - --epochs EPOCHS       Number of epochs to train.
    - --steps STEPS         Number of steps per epoch.
    - --snapshot-path SNAPSHOT_PATH Path to store snapshots of models during training
    - --tensorboard-dir TENSORBOARD_DIR  Log directory for Tensorboard output
    - --no-snapshots        Disable saving snapshots.
    - --no-evaluation       Disable per epoch evaluation.
    - --random-transform    Randomly transform image and annotations.
    - --compute-val-loss    Compute validation loss during training
    - --multiprocessing     Use multiprocessing in fit_generator.
    - --workers WORKERS     Number of generator workers.
    - --max-queue-size MAX_QUEUE_SIZE     Queue length for multiprocessing workers in  fit_generator.

## Test

- `predict.py --data-path /pat/to/dataset --score-threshold 0.5 --model-path path/to/our/model` to start testing, the result will store in `result.json`
- Optional arguments:
    - --data-path DATA_PATH  Data for prediction 
    - --target-path TARGET_PATH    Target path
    - --split SPLIT         Target path
    - --max-detections MAX_DETECTIONS    Max detection
    - --ninedash-category-id NINEDASH_CATEGORY_ID    Ninedash category ID
    - --model-path MODEL_PATH Model path of the network
    - --score-threshold SCORE_THRESHOLD    Minimum score threshold
    - --phi {0,1,2,3,4,5,6} Hyper parameter phi
    - --weighted-bifpn      Use weighted BiFPN
    - --batch-size BATCH_SIZE   Size of the batches.
    - --num-classes NUM_CLASSES Number of classes
    - --gpu GPU             Id of the GPU to use (as reported by nvidia-smi).