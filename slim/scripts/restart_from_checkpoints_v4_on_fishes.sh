#!/bin/bash

# Where the training checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/fishes/inception_v4

# Where the dataset is saved to.
DATASET_DIR=/home/ubuntu/workspace/data

touch fish_log

for steps in $(seq 6000 1000 10000)
do

echo "Steps: "$steps | tee -a fish_log

# Fine-tune all the new layers
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=fishes \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4 \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=${steps} \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=200 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 | tee -a fish_log

echo "Running evaluation on training data" | tee -a fish_log

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=fishes \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4 | tee -a fish_log

echo "Running evaluation on validation data" | tee -a fish_log

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=fishes \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4 | tee -a fish_log

done
