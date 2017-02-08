#!/bin/bash

# The final output of this script are sharded TFRecord files containing
# serialized Example protocol buffers. See build_imagenet_data.py for
# details of how the Example protocol buffers contain the ImageNet data.
#
# The final output of this script appears as such:
#
#   data_dir/train-00000-of-01024
#   data_dir/train-00001-of-01024
#    ...
#   data_dir/train-00127-of-01024
#
# and
#
#   data_dir/validation-00000-of-00128
#   data_dir/validation-00001-of-00128
#   ...
#   data_dir/validation-00127-of-00128
set -e

if [[ $# <  5 ]]
then
  echo "usage ./preprocess_fishes.sh [raw training data dir] [processed data dir] [labels file] [bboxes file] [percent] [raw testing data dir]"
  exit
fi

RAW_DATA_DIR="$1"
DATA_DIR="$2"
LABELS_FILE="$3"
BOUNDING_BOX_FILE="$4"
PERCENT="$5"
TESTING_RAW_DATA_DIR="NULL"

if [[ $# ==  6 ]]
then
    TESTING_RAW_DATA_DIR="$6"
fi

echo "Inputs: $RAW_DATA_DIR $DATA_DIR $LABELS_FILE $BOUNDING_BOX_FILE"

#----------------------------------
# create directories
#----------------------------------
mkdir -p "${DATA_DIR}"
mkdir -p "${RAW_DATA_DIR}"
WORK_DIR="$(pwd)"

TRAIN_DIRECTORY="$DATA_DIR/train/"
VALIDATION_DIRECTORY="$DATA_DIR/validation/"

#----------------------------------------------
# move data to validation dir and training dir
#----------------------------------------------
echo "Splitting raw data to training and validation"
PREPROCESS_VAL_SCRIPT="$WORK_DIR/generate_train_validation_sets.py"

"${PREPROCESS_VAL_SCRIPT}" "${RAW_DATA_DIR}" "${DATA_DIR}" "${PERCENT}"

#----------------------------------
# Build the TFRecords
#----------------------------------
BUILD_SCRIPT="${WORK_DIR}/build_fishes_data.py"
OUTPUT_DIRECTORY="${DATA_DIR}"
# metadata --> image/class/text
METADATA_FILE="${WORK_DIR}/fishes_metadata.txt"

# training
if [[ "$TESTING_RAW_DATA_DIR" == "NULL" ]]
then
    
"${BUILD_SCRIPT}" \
  --train_directory="${TRAIN_DIRECTORY}" \
  --validation_directory="${VALIDATION_DIRECTORY}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}" \
  --bounding_box_file="${BOUNDING_BOX_FILE}" \
  --train_shards=3 \
  --validation_shards=1 \
  --num_threads=1 \
  --metadata_file="${METADATA_FILE}"
# testing
else
  "${BUILD_SCRIPT}" \
  --test_directory="${TESTING_RAW_DATA_DIR}" \
  --test_shards=1	
  --train_directory="${TRAIN_DIRECTORY}" \
  --validation_directory="${VALIDATION_DIRECTORY}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}" \
  --bounding_box_file="${BOUNDING_BOX_FILE}" \
  --train_shards=3 \
  --validation_shards=1 \
  --num_threads=1 \
  --metadata_file="${METADATA_FILE}"
fi
