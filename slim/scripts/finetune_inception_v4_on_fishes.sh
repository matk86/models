#!/bin/bash

# Usage:
# cd slim
# ./scripts/finetune_inception_v4_on_fishes.sh [checkpoint dir] [log dir] [data dir] [type reg or class] [total number of samples] [number of validation samples] [number of steps] [batch size]
# ./scripts/finetune_inception_v4_on_fishes.sh ~/checkpoints /tmp/fishes/inception_v4_with_bbox ~/workspace/data_bbox reg 3423 354 5000 32 0.01

if [[ $# != 9 ]]
then
    echo "Usage:"
    echo "cd slim"
    echo "./scripts/finetune_inception_v4_on_fishes.sh [checkpoint dir] [log dir] [data dir] [type reg or class] [total number of samples] [number of validation samples] [number of steps] [batch size] [learning rate]"
    exit
fi


# Where the pre-trained InceptionV4 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=$1 #~/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=$2 #/tmp/fishes/inception_v4_with_bbox

# Where the dataset is saved to.
DATASET_DIR=$3 #~/workspace/data_bbox

TYPE=$4

NUM_SAMPLES=$5

NUM_VALIDATION_SAMPLES=$6

NUM_STEPS=$7 #5000

BATCH_SIZE=$8 #32

LEARN_RATE=$9

((NUM_TRAIN_SAMPLES=$NUM_SAMPLES-$NUM_VALIDATION_SAMPLES))

dataset_name="fishes"
model_name="inception_v4"
train_script="train_image_classifier.py"
eval_script="eval_image_classifier.py"
# for regression
if [[ "${TYPE}" == "reg" ]]
then
    dataset_name="fishes_bboxes"
    model_name="inception_v4_regression"
    train_script="train_image_bbox_regressor.py"
    eval_script="eval_image_regressor.py"
fi


# Fine-tune only the new layers.
python ${train_script} \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt \
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
  --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
  --max_number_of_steps=${NUM_STEPS} \
  --batch_size=${BATCH_SIZE} \
  --learning_rate=${LEARN_RATE} \
  --learning_rate_decay_type=polynomial \
  --num_epochs_per_decay=10 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.000001 \
  --num_train_samples=$NUM_TRAIN_SAMPLES \
  --num_validation_samples=$NUM_VALIDATION_SAMPLES  


# Run evaluation on training data
python ${eval_script} \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --max_num_batches=1 \
  --num_train_samples=$NUM_TRAIN_SAMPLES \
  --num_validation_samples=$NUM_VALIDATION_SAMPLES  


# Run evaluation on validation data
python ${eval_script} \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --max_num_batches=1 \
  --num_train_samples=$NUM_TRAIN_SAMPLES \
  --num_validation_samples=$NUM_VALIDATION_SAMPLES  
