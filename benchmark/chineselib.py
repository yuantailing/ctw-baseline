# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import random
import settings
import tensorflow as tf

from pythonapi import common_tools
from six.moves import cPickle


class TrainSet:
    def load_data(self, FLAGS):
        self.FLAGS = FLAGS
        self.num_classes = settings.NUM_CHAR_CATES + 1
        self.num_samples = 0
        self.labels = [[] for i in range(self.num_classes)]
        with open(FLAGS.dataset_dir, 'rb') as f:
            all = cPickle.load(f)
        for image, label in all:
            if label is None:
                self.labels[settings.NUM_CHAR_CATES].append(image)
            else:
                self.labels[label].append(image)
            self.num_samples += 1
        for i in range(self.num_classes):
            assert 0 < len(self.labels[i])

    def set_holders(self, image_holder, label_holder):
        self.image_holder = image_holder
        self.label_holder = label_holder

    def get_feed_dict(self):
        a = [random.randrange(self.num_classes) for i in range(self.FLAGS.batch_size)]
        labels_feed = []
        for c in a:
            v = [0.] * self.num_classes
            v[c] = 1.
            labels_feed.append(v)
        images_feed = [self.labels[a[i]][random.randrange(0, len(self.labels[a[i]]))] for i in range(self.FLAGS.batch_size)]

        def preprocess_image(i):
            img = images_feed[i]
            rows, cols, ch = img.shape
            output_size = self.FLAGS.train_image_size
            r = lambda: (random.random() - 0.5) * 0.2 * output_size
            pts1 = np.float32([[0, 0], [cols, rows], [0, rows]])
            pts2 = np.float32([[r(), r()], [output_size + r(), output_size + r()], [r(), output_size + r()]])
            M = cv2.getAffineTransform(pts1, pts2)
            noize = np.random.normal(0, random.random() * (0.05 * 255), size=img.shape)
            img = np.array(img, dtype=np.float32) + noize
            img = cv2.warpAffine(img, M, (output_size, output_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            if output_size >= 128:
                img = cv2.GaussianBlur(img, (5, 5), random.random() * 0.5)
            images_feed[i] = img
        common_tools.multithreaded(preprocess_image, range(len(images_feed)), num_thread=8)
        return {self.image_holder: np.array(images_feed, dtype=np.float),
                self.label_holder: np.array(labels_feed, dtype=np.float)}

    @staticmethod
    def get_tf_preprocess_image(is_training):
        def tf_preprocess_image(image, output_height, output_width, is_training):
            image = tf.to_float(image)
            image = tf.image.resize_image_with_crop_or_pad(
                image, output_width, output_height)
            image = tf.subtract(image, 128.0)
            image = tf.div(image, 128.0)
            return image
        def foo(image, output_height, output_width, **kwargs):
            return tf_preprocess_image(
                image, output_height, output_width, is_training=is_training, **kwargs)
        return foo

trainset = TrainSet()


import os
import sys
import time
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.training.python.training import training
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import supervisor
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util


def train_step(sess, train_op, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.
  Args:
    sess: The current session.
    train_op: An `Operation` that evaluates the gradients and returns the
      total loss.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments.
  Returns:
    The total loss and a boolean indicating whether or not to stop training.
  Raises:
    ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
  """
  start_time = time.time()

  trace_run_options = None
  run_metadata = None

  if 'should_trace' in train_step_kwargs:
    if 'logdir' not in train_step_kwargs:
      raise ValueError('logdir must be present in train_step_kwargs when '
                       'should_trace is present')
    if sess.run(train_step_kwargs['should_trace']):
      trace_run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()

  total_loss, np_global_step = sess.run([train_op, global_step],
                                        feed_dict=trainset.get_feed_dict(),
                                        options=trace_run_options,
                                        run_metadata=run_metadata)
  time_elapsed = time.time() - start_time

  if run_metadata is not None:
    tl = timeline.Timeline(run_metadata.step_stats)
    trace = tl.generate_chrome_trace_format()
    trace_filename = os.path.join(train_step_kwargs['logdir'],
                                  'tf_trace-%d.json' % np_global_step)
    logging.info('Writing trace to %s', trace_filename)
    file_io.write_string_to_file(trace_filename, trace)
    if 'summary_writer' in train_step_kwargs:
      train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                           'run_metadata-%d' %
                                                           np_global_step)

  if 'should_log' in train_step_kwargs:
    if sess.run(train_step_kwargs['should_log']):
      logging.info('global step %d: loss = %.4f (%.2f sec/step)',
                   np_global_step, total_loss, time_elapsed)

  # TODO(nsilberman): figure out why we can't put this into sess.run. The
  # issue right now is that the stop check depends on the global step. The
  # increment of global step often happens via the train op, which used
  # created using optimizer.apply_gradients.
  #
  # Since running `train_op` causes the global step to be incremented, one
  # would expected that using a control dependency would allow the
  # should_stop check to be run in the same session.run call:
  #
  #   with ops.control_dependencies([train_op]):
  #     should_stop_op = ...
  #
  # However, this actually seems not to work on certain platforms.
  if 'should_stop' in train_step_kwargs:
    should_stop = sess.run(train_step_kwargs['should_stop'])
  else:
    should_stop = False

  return total_loss, should_stop
