#!/bin/bash

# Where the training checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/fishes/inception_v4_bbox_regression

# Where the dataset is saved to.
DATASET_DIR=/home/ubuntu/workspace/data_reg

echo "Running evaluation on training data" 

#python eval_image_regressor.py \
#  --checkpoint_path=${TRAIN_DIR} \
#  --eval_dir=${TRAIN_DIR} \
#  --dataset_name=fishes_bboxes \
#  --dataset_split_name=train \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=inception_v4_regression

echo "Running evaluation on validation data"

python eval_image_regressor.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=fishes_bboxes \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4_regression \
  --max_num_batches=1
#  --batch_size=308

