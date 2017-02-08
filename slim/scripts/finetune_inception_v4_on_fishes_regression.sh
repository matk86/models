#!/bin/bash

# Usage:
# cd slim
# ./slim/scripts/finetune_inceptionv4_on_fishes.sh

# Where the pre-trained InceptionV4 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=~/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/fishes/inception_v4_bbox_regression

# Where the dataset is saved to.
DATASET_DIR=~/workspace/data_bbox

# Fine-tune only the new layers for 5000 steps.
python train_image_bbox_regressor.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=fishes_bboxes \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4_regression \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt \
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
  --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
  --max_number_of_steps=5000 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=250 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

echo "Running evaluation on training data" 

python eval_image_regressor.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=fishes_bboxes \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4_regression \
  --max_num_batches=1 

echo "Running evaluation on validation data"

python eval_image_regressor.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=fishes_bboxes \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4_regression \
  --max_num_batches=1
