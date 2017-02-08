#!/usr/bin/env python
"""Converts ImageNet data to TFRecords file format with Example protos.

The raw ImageNet data set is expected to reside in JPEG files located in the
following directory structure.

  data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
  data_dir/n01440764/ILSVRC2012_val_00000543.JPEG
  ...

where 'n01440764' is the unique synset label associated with
these images.

The training data set consists of 1000 sub-directories (i.e. labels)
each containing 1200 JPEG images for a total of 1.2M JPEG images.

The evaluation data set consists of 1000 sub-directories (i.e. labels)
each containing 50 JPEG images for a total of 50K JPEG images.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-00127-of-01024

and

  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128

Each validation TFRecord file contains ~390 records. Each training TFREcord
file contains ~1250 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always'JPEG'

  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [1, 1000] where 0 is not used.
  image/class/synset: string specifying the unique ID of the label,
    e.g. 'n01440764'
  image/class/text: string specifying the human-readable version of the label
    e.g. 'red fox, Vulpes vulpes'

  image/object/bbox/xmin: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/xmax: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/ymin: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/ymax: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/label: integer specifying the index in a classification
    layer. The label ranges from [1, 1000] where 0 is not used. Note this is
    always identical to the image label.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.

Running this script using 16 threads may take around ~2.5 hours on a HP Z420.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf


tf.app.flags.DEFINE_string('train_directory', '/tmp/',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/tmp/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('test_directory', None,
                           'test data directory')
tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 1024,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 128,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('test_shards', 1,
                            'Number of shards in testing TFRecord files.')


tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')


# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   n01440764
#   n01443537
#   n01484850
# where each line corresponds to a label expressed as a synset. We map
# each synset contained in the file to an integer (based on the alphabetical
# ordering). See below for details.
tf.app.flags.DEFINE_string('labels_file',
                           'imagenet_lsvrc_2015_synsets.txt',
                           'Labels file')

# This file containing mapping from synset to human-readable label.
# Assumes each line of the file looks like:
#
#   n02119247    black fox
#   n02119359    silver fox
#   n02119477    red fox, Vulpes fulva
#
# where each line corresponds to a unique mapping. Note that each line is
# formatted as <synset>\t<human readable label>.
tf.app.flags.DEFINE_string('metadata_file',
                           'fishes_metadata.txt',
                           'metadata file')

# This file is the output of process_bounding_box.py
# Assumes each line of the file looks like:
#
#   n00007846_64193.JPEG,0.0060,0.2620,0.7545,0.9940
#
# where each line corresponds to one bounding box annotation associated
# with an image. Each line can be parsed as:
#
#   <JPEG file name>, <xmin>, <ymin>, <xmax>, <ymax>
#
# Note that there might exist mulitple bounding box annotations associated
# with an image file.
tf.app.flags.DEFINE_string('bounding_box_file',
                           None,
                           'Bounding box file')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, synset, human, bbox,
                        height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
    bbox: list of bounding boxes; each box is a list of integers
      specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong to
      the same label as the image label.
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  bbox = bbox if bbox is not None else []
  
  xmin = []
  ymin = []
  xmax = []
  ymax = []
  bbx = []

  for b in bbox:
    assert len(b) == 4
    # pylint: disable=expression-not-assigned
    [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
    # pylint: enable=expression-not-assigned
  
  # atmost one bbox per file ==> regression problem
  if len(bbox) == 1:
    assert len(bbox[0]) == 4
    # bbox[0] --> [xmin, ymin, xmax, ymax]
    # bbx(bbox_list) --> [ymin, xmin, ymax, xmax]
    bbx = [bbox[0][1], bbox[0][0], bbox[0][3], bbox[0][2]]
    #for bi in bbox[0]:
    #  bbx.append(bi)
 # images containing no fish
  elif (len(bbox) == 0) or ("NoF" in filename):
    #print("no bbox for:{}".format(filename))
    xmin = [0.0]
    ymin = [0.0]
    xmax = [0.0]
    ymax = [0.0]
    bbx = [0.0, 0.0, 0.0, 0.0]
  # more than one bounding boxes ==> not a regression problem, so bbox_list not used, set to some value
  else:
    bbx = [0.0, 0.0, 0.0, 0.0]

  colorspace = 'RGB'
  channels = 3
  image_format = 'jpg'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/synset': _bytes_feature(synset),
      'image/class/text': _bytes_feature(human),
      'image/object/bbox/xmin': _float_feature(xmin),
      'image/object/bbox/xmax': _float_feature(xmax),
      'image/object/bbox/ymin': _float_feature(ymin),
      'image/object/bbox/ymax': _float_feature(ymax),
      'image/object/bbox/label': _int64_feature([label] * len(xmin)),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/encoded': _bytes_feature(image_buffer),
      'image/bbox_list': tf.train.Feature(float_list=tf.train.FloatList(value=bbx))
  })
      )
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _is_png(filename):
  """Determine if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
  return '.png' in filename


def _is_cmyk(filename):
  """Determine if file contains a CMYK JPEG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a JPEG encoded with CMYK color space.
  """
  return False


def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'r') as f:
    image_data = f.read()

  # Clean the dirty data.
  if _is_png(filename):
    # 1 image is a PNG.
    print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)
  elif _is_cmyk(filename):
    # 22 JPEG images are in CMYK colorspace.
    print('Converting CMYK to RGB for %s' % filename)
    image_data = coder.cmyk_to_rgb(image_data)

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               synsets, labels, humans, bboxes, num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: list of integer; each integer identifies the ground truth
    humans: list of strings; each string is a human-readable label
    bboxes: list of bounding boxes for each image. Note that each entry in this
      list might contain from 0+ entries corresponding to the number of bounding
      box annotations for the image.
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      synset = synsets[i]
      human = humans[i]
      bbox = bboxes[i] if bboxes else []

      image_buffer, height, width = _process_image(filename, coder)

      example = _convert_to_example(filename, image_buffer, label,
                                    synset, human, bbox,
                                    height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, synsets, labels, humans,
                         bboxes, num_shards):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: list of integer; each integer identifies the ground truth
    humans: list of strings; each string is a human-readable label
    bboxes: list of bounding boxes for each image. Note that each entry in this
      list might contain from 0+ entries corresponding to the number of bounding
      box annotations for the image.
    num_shards: integer number of shards for this data set.
  """
  assert len(filenames) == len(synsets)
  assert len(filenames) == len(labels)
  assert len(filenames) == len(humans)
  if bboxes is not None:
    assert len(filenames) == len(bboxes)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames,
            synsets, labels, humans, bboxes, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(data_dir, labels_file):
  """Build a list of all images files and labels in the data set.

  Args:
    data_dir: string, path to the root directory of images.

      Assumes that the ImageNet data set resides in JPEG files located in
      the following directory structure.

        data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
        data_dir/n01440764/ILSVRC2012_val_00000543.JPEG

      where 'n01440764' is the unique synset label associated with these images.

    labels_file: string, path to the labels file.

      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        n01440764
        n01443537
        n01484850
      where each line corresponds to a label expressed as a synset. We map
      each synset contained in the file to an integer (based on the alphabetical
      ordering) starting with the integer 1 corresponding to the synset
      contained in the first line.

      The reason we start the integer labels at 1 is to reserve label 0 as an
      unused background class.

  Returns:
    filenames: list of strings; each string is a path to an image file.
    synsets: list of strings; each string is a unique WordNet ID.
    labels: list of integer; each integer identifies the ground truth.
  """
  print('Determining list of input files and labels from %s.' % data_dir)
  challenge_synsets = [l.strip() for l in
                       tf.gfile.FastGFile(labels_file, 'r').readlines()]

  labels = []
  filenames = []
  synsets = []

  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of JPEG files and labels.
  if FLAGS.test_directory is not None:
    jpeg_file_path = '%s/*.jpg' % (data_dir,)
    matching_files = tf.gfile.Glob(jpeg_file_path)
    labels.extend([label_index] * len(matching_files))
    synsets.extend([challenge_synsets[0]] * len(matching_files))
    filenames.extend(matching_files)
  else:
    for synset in challenge_synsets:
      jpeg_file_path = '%s/%s/*.jpg' % (data_dir, synset)
      matching_files = tf.gfile.Glob(jpeg_file_path)
      labels.extend([label_index] * len(matching_files))
      synsets.extend([synset] * len(matching_files))
      filenames.extend(matching_files)

      if not label_index % 100:
        print('Finished finding files in %d of %d classes.' % (
          label_index, len(challenge_synsets)))
      label_index += 1

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  synsets = [synsets[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(challenge_synsets), data_dir))
  return filenames, synsets, labels


def _find_human_readable_labels(synsets, synset_to_human):
  """Build a list of human-readable labels.

  Args:
    synsets: list of strings; each string is a unique WordNet ID.
    synset_to_human: dict of synset to human labels, e.g.,
      'n02119022' --> 'red fox, Vulpes vulpes'

  Returns:
    List of human-readable strings corresponding to each synset.
  """
  humans = []
  for s in synsets:
    assert s in synset_to_human, ('Failed to find: %s' % s)
    humans.append(synset_to_human[s])
  return humans


def _find_image_bounding_boxes(filenames, image_to_bboxes):
  """Find the bounding boxes for a given image file.

  Args:
    filenames: list of strings; each string is a path to an image file.
    image_to_bboxes: dictionary mapping image file names to a list of
      bounding boxes. This list contains 0+ bounding boxes.
  Returns:
    List of bounding boxes for each image. Note that each entry in this
    list might contain from 0+ entries corresponding to the number of bounding
    box annotations for the image.
  """
  num_image_bbox = 0
  bboxes = []
  for f in filenames:
    basename = os.path.basename(f)
    if basename in image_to_bboxes:
      bboxes.append(image_to_bboxes[basename])
      num_image_bbox += 1
    else:
      bboxes.append([])
  print('Found %d images with bboxes out of %d images' % (
      num_image_bbox, len(filenames)))
  return bboxes


def _process_dataset(name, directory, num_shards, synset_to_human,
                     image_to_bboxes):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    synset_to_human: dict of synset to human labels, e.g.,
      'n02119022' --> 'red fox, Vulpes vulpes'
    image_to_bboxes: dictionary mapping image file names to a list of
      bounding boxes. This list contains 0+ bounding boxes.
  """
  filenames, synsets, labels = _find_image_files(directory, FLAGS.labels_file)
  humans = _find_human_readable_labels(synsets, synset_to_human)
  bboxes = None
  if image_to_bboxes:
    bboxes = _find_image_bounding_boxes(filenames, image_to_bboxes)
  _process_image_files(name, filenames, synsets, labels,
                       humans, bboxes, num_shards)


def _build_synset_lookup(metadata_file):
  """Build lookup for synset to human-readable label.

  Args:
    imagenet_metadata_file: string, path to file containing mapping from
      synset to human-readable label.

      Assumes each line of the file looks like:

        n02119247    black fox
        n02119359    silver fox
        n02119477    red fox, Vulpes fulva

      where each line corresponds to a unique mapping. Note that each line is
      formatted as <synset>\t<human readable label>.

  Returns:
    Dictionary of synset to human labels, such as:
      'n02119022' --> 'red fox, Vulpes vulpes'
  """
  lines = tf.gfile.FastGFile(metadata_file, 'r').readlines()
  synset_to_human = {}
  for l in lines:
    if l:
      parts = l.strip().split()
      assert len(parts) > 1
      synset = parts[0]
      human = " ".join(parts[1:])
      synset_to_human[synset] = human
  return synset_to_human


def _build_bounding_box_lookup(bounding_box_file):
  """Build a lookup from image file to bounding boxes.

  Args:
    bounding_box_file: string, path to file with bounding boxes annotations.

      Assumes each line of the file looks like:

        n00007846_64193.JPEG,0.0060,0.2620,0.7545,0.9940

      where each line corresponds to one bounding box annotation associated
      with an image. Each line can be parsed as:

        <JPEG file name>, <xmin>, <ymin>, <xmax>, <ymax>

      Note that there might exist mulitple bounding box annotations associated
      with an image file. This file is the output of process_bounding_boxes.py.

  Returns:
    Dictionary mapping image file names to a list of bounding boxes. This list
    contains 0+ bounding boxes.
  """
  lines = tf.gfile.FastGFile(bounding_box_file, 'r').readlines()
  images_to_bboxes = {}
  num_bbox = 0
  num_image = 0
  for l in lines:
    if l:
      parts = l.split(',')
      assert len(parts) == 5, ('Failed to parse: %s' % l)
      filename = parts[0]
      xmin = float(parts[1])
      ymin = float(parts[2])
      xmax = float(parts[3])
      ymax = float(parts[4])
      box = [xmin, ymin, xmax, ymax]

      if filename not in images_to_bboxes:
        images_to_bboxes[filename] = []
        num_image += 1
      images_to_bboxes[filename].append(box)
      num_bbox += 1

  print('Successfully read %d bounding boxes '
        'across %d images.' % (num_bbox, num_image))
  return images_to_bboxes


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)

  # Build a map from synset to human-readable label.
  synset_to_human = _build_synset_lookup(FLAGS.metadata_file)

  # read in bboxes is provided
  if FLAGS.bounding_box_file is None:
    image_to_bboxes = None
  else:
    image_to_bboxes = _build_bounding_box_lookup(FLAGS.bounding_box_file)

  # Run it!
  if FLAGS.test_directory is not None:
      _process_dataset('test', FLAGS.test_directory,
                       FLAGS.test_shards, synset_to_human, image_to_bboxes)
  else:
    _process_dataset('validation', FLAGS.validation_directory,
                   FLAGS.validation_shards, synset_to_human, image_to_bboxes)
    _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards,
                   synset_to_human, image_to_bboxes)


if __name__ == '__main__':
  tf.app.run()
