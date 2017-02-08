# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import json
import numpy as np
import csv

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory


slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/fishes/inception_v4',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/fishes/inception_v4', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'fishes', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'validation', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'num_train_samples', 3400,
    'Number of training samples')

tf.app.flags.DEFINE_integer(
    'num_validation_samples', 320,
    'Number of validation samples')

tf.app.flags.DEFINE_integer(
    'num_test_samples', 200,
    'Number of test samples')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v4', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_string(
    'bbox_output_fname', None, 'Bbox output filename(csv file containing filenames and corresponding bboxes)')


FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir,
        splits_to_sizes={'train': FLAGS.num_train_samples,
                         'validation': FLAGS.num_validation_samples,
                         'test': FLAGS.num_test_samples})

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=dataset.num_classes,
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=3 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, bbox, filename] = provider.get(['image', 'bbox', 'filename'])

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, bboxes, filenames = tf.train.batch(
        [image, bbox, filename],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=2 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    predictions, _ = network_fn(images)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_mean_squared_error(predictions, bboxes)
    })

    # Print the summaries to screen.
    for name, value in names_to_values.iteritems():
      summary_name = 'eval/%s' % name
      op = tf.scalar_summary(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Num samples %s' % dataset.num_samples)
    tf.logging.info('Num batches %s' % num_batches)
    tf.logging.info('Batch size %s' % FLAGS.batch_size)
    tf.logging.info('Evaluating %s' % checkpoint_path)

    # no evaluation if testing
    if FLAGS.bbox_output_fname is not None:
      num_evals = 0
      eval_op = []
      final_op=[predictions, bboxes, filenames]
    else:
      num_evals=num_batches
      eval_op=names_to_updates.values()
      final_op=[]      
    
    preds, bbxs, fnames = slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_evals,
        eval_op=eval_op,
        final_op=final_op,
        variables_to_restore=variables_to_restore)

    if FLAGS.bbox_output_fname is not None:
      d = {}
      with open(FLAGS.bbox_output_fname, 'w') as f:
        writer = csv.writer(f)      
        for i in range(len(preds)):
          #b = np.asarray(bbxs[i])
          p = np.asarray(preds[i])
          fname = str(fnames[i])
          #d[fname] = [p.tolist(), b.tolist()]
          writer.writerow([fname]+p.tolist())          
      #json.dump(d, open("regression_output.json", "w"))
      #print(np.mean((preds-bbxs)**2))


if __name__ == '__main__':
  tf.app.run()
