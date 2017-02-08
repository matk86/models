#!/bin/bash

CWD="$(pwd)"

if [[ $# != 5 ]]
then
    echo "run from slim dir"
    echo "usage: ./scripts/tester.sh [raw test data dir] [path to regression checkpoint dir] [path to classification checkpoint dir] [total num samples] [batch size]"
    exit
fi

TESTING_DATA_DIR=$1

# Pre-trained InceptionV4 checkpoints
REG_CHECKPOINT_DIR=$2
CLASS_CHECKPOINT_DIR=$3

NUM_SAMPLES=$4

BATCH_SIZE=$5

LABELS_FILE="$CWD/datasets/process_fishes/labels.txt"

METADATA_FILE="$CWD/datasets/process_fishes/fishes_metadata.txt"

echo $METADATA_FILE

# tfrecords path
DATASET_DIR_1="/tmp/fishes_test/data_1"
DATASET_DIR_2="/tmp/fishes_test/data_2"

mkdir -p $DATASET_DIR_1
mkdir -p $DATASET_DIR_2

RAW_TEST_DATA_DIR="/tmp/fishes_test/raw_data"
mkdir -p $RAW_TEST_DATA_DIR
rm -rf $RAW_TEST_DATA_DIR/*

# copy original test data to temp folder
cp $TESTING_DATA_DIR/* $RAW_TEST_DATA_DIR

# Where the logs will be saved
EVAL_DIR="/tmp/fishes_1/eval"


((NUM_BATCHES = NUM_SAMPLES/BATCH_SIZE))

RAW_DATA_BATCH="/tmp/fishes_test/raw_data_batch"
mkdir -p $RAW_DATA_BATCH

COMPLETE_BOUNDING_BOX_FILE="$CWD/fishes_bbox_complete.csv"
COMPLETE_TEST_OUT_FILE="$CWD/fishes_logits_complete.csv"

rm -f $COMPLETE_TEST_OUT_FILE
rm -f $COMPLETE_BOUNDING_BOX_FILE

for i in $(seq 1 $NUM_BATCHES)
do

# bbox file name
BOUNDING_BOX_FILE="$CWD/fishes_bbox_${i}.csv"
# output csv file with the logits
TEST_OUT_FILE="$CWD/fishes_logits_${i}.csv"
    
rm -rf $RAW_DATA_BATCH/*
rm -rf $DATASET_DIR_1/*
rm -rf $DATASET_DIR_2/*

echo "moving ${BATCH_SIZE} test sample to ${RAW_DATA_BATCH}"
cd $RAW_TEST_DATA_DIR
j=1
for f in *
do 
mv $f $RAW_DATA_BATCH/
((j = j + 1))
#echo $j
if (( $j > $BATCH_SIZE ))
then 
break
fi
done

cd $CWD
    
BUILD_SCRIPT="$CWD/datasets/process_fishes/build_fishes_data.py"


echo "converting the raw test data to tfrecords"

"${BUILD_SCRIPT}" \
  --metadata_file=${METADATA_FILE} \
  --test_directory=${RAW_DATA_BATCH} \
  --test_shards=1  \
  --output_directory=${DATASET_DIR_1} \
  --labels_file=${LABELS_FILE} \
  --num_threads=1 


if [[ $? != 0 ]]
then
echo "Failed: converting raw data"
exit
fi

echo "Running bounding box regression"


python eval_image_regressor.py \
  --checkpoint_path=${REG_CHECKPOINT_DIR} \
  --eval_dir=${EVAL_DIR}/reg \
  --dataset_name=fishes_bboxes \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR_1} \
  --model_name=inception_v4_regression \
  --bbox_output_fname="${BOUNDING_BOX_FILE}" \
  --batch_size=$BATCH_SIZE \
  --num_test_samples=$BATCH_SIZE \
  --max_num_batches=1 \


if [[ $? != 0 ]]
then
echo "Failed: regression net"
exit
fi


echo "Converting raw test data + predicted bbox data to tfrecords"


"${BUILD_SCRIPT}" \
  --metadata_file=${METADATA_FILE} \
  --test_directory=${RAW_DATA_BATCH} \
  --test_shards=1  \
  --output_directory=${DATASET_DIR_2} \
  --labels_file=${LABELS_FILE} \
  --num_threads=1 \
  --bounding_box_file=${BOUNDING_BOX_FILE} 


if [[ $? != 0 ]]
then
echo "Failed: converting raw data(with bbox)"
exit
fi

  
echo "Running image classification"

# Run evaluation on validation data
python eval_image_classifier.py \
  --checkpoint_path=${CLASS_CHECKPOINT_DIR} \
  --eval_dir=${EVAL_DIR}/class \
  --dataset_name=fishes \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR_2} \
  --model_name=inception_v4 \
  --test_output_fname="${TEST_OUT_FILE}" \
  --batch_size=$BATCH_SIZE \
  --num_test_samples=$BATCH_SIZE \
  --max_num_batches=1

cat $BOUNDING_BOX_FILE >> $COMPLETE_BOUNDING_BOX_FILE
cat $TEST_OUT_FILE >> $COMPLETE_TEST_OUT_FILE

done
