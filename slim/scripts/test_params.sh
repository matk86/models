#!/bin/bash

# Usage:
# cd slim
# ./slim/scripts/finetune_inceptionv4_on_fishes.sh

# Where the pre-trained InceptionV4 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/home/ubuntu/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/fishes/inception_v4

# Where the dataset is saved to.
DATASET_DIR=/home/ubuntu/workspace/data

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
  tar -xvf inception_v4_2016_09_09.tar.gz
  mv inception_v4.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt
  rm inception_v4_2016_09_09.tar.gz
fi


for i in `seq 0.0005 0.0005 0.002`
do

# Fine-tune only the new layers for 5000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/$i \
  --dataset_name=fishes \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt \
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
  --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
  --max_number_of_steps=1000 \
  --batch_size=32 \
  --learning_rate=$i \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

done

# # Run evaluation on training data
# python eval_image_classifier.py \
#   --checkpoint_path=${TRAIN_DIR} \
#   --eval_dir=${TRAIN_DIR} \
#   --dataset_name=fishes \
#   --dataset_split_name=train \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=inception_v4


# # Run evaluation on validation data
# python eval_image_classifier.py \
#   --checkpoint_path=${TRAIN_DIR} \
#   --eval_dir=${TRAIN_DIR} \
#   --dataset_name=fishes \
#   --dataset_split_name=validation \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=inception_v4

# # Fine-tune all the new layers for 1000 steps.
# python train_image_classifier.py \
#   --train_dir=${TRAIN_DIR}/all \
#   --dataset_name=fishes \
#   --dataset_split_name=train \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=inception_v4 \
#   --checkpoint_path=${TRAIN_DIR} \
#   --max_number_of_steps=1000 \
#   --batch_size=32 \
#   --learning_rate=0.0001 \
#   --learning_rate_decay_type=fixed \
#   --save_interval_secs=60 \
#   --save_summaries_secs=60 \
#   --log_every_n_steps=100 \
#   --optimizer=rmsprop \
#   --weight_decay=0.00004


# # Run evaluation on training data
# python eval_image_classifier.py \
#   --checkpoint_path=${TRAIN_DIR} \
#   --eval_dir=${TRAIN_DIR} \
#   --dataset_name=fishes \
#   --dataset_split_name=train \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=inception_v4


# # Run evaluation on validation data
# python eval_image_classifier.py \
#   --checkpoint_path=${TRAIN_DIR} \
#   --eval_dir=${TRAIN_DIR} \
#   --dataset_name=fishes \
#   --dataset_split_name=validation \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=inception_v4
