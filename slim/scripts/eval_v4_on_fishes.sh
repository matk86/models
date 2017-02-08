#!/bin/bash

# Where the training checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/fishes/inception_v4

# Where the dataset is saved to.
DATASET_DIR=/home/ubuntu/workspace/data

echo "Running evaluation on training data" 

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=fishes \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4 \
  --max_num_batches=1

echo "Running evaluation on validation data"

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=fishes \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4 \
  --max_num_batches=1

