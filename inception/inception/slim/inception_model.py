# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Inception-v3 expressed in TensorFlow-Slim.

  Usage:

  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the batch_norm moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      # Force all Variables to reside on the CPU.
      with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
        logits, endpoints = slim.inception.inception_v3(
            images,
            dropout_keep_prob=0.8,
            num_classes=num_classes,
            is_training=for_training,
            restore_logits=restore_logits,
            scope=scope)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from inception.slim import ops
from inception.slim import scopes


def inception_v3(inputs,
                 dropout_keep_prob=0.8,
                 num_classes=1001,
                 is_training=True,
                 restore_logits=True,
                 scope=''):
  """Latest Inception from http://arxiv.org/abs/1512.00567.

    "Rethinking the Inception Architecture for Computer Vision"

    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
    Zbigniew Wojna

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    dropout_keep_prob: dropout keep_prob.
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: Optional scope for op_scope.

  Returns:
    a list containing 'logits', 'aux_logits' Tensors.
  """
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}
  with tf.op_scope([inputs], scope, 'baxNet'):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                            stride=1, padding='VALID'):
        # 256 x 256 x 3
        end_points['conv0'] = ops.conv2d(inputs, 8, [5, 5], stride=1,
                                         scope='conv0', padding='SAME')
        
        end_points['batch_norm1'] = ops.batch_norm(end_points['conv0'], scope='batch_norm1')

        # 256 x 256 x 32
        end_points['conv1'] = ops.conv2d(end_points['batch_norm1'], 16, [3, 3],
                                         scope='conv1', padding='SAME')

        end_points['batch_norm2'] = ops.batch_norm(end_points['conv1'], scope='batch_norm2')

        # 128 x 128 x 64
        end_points['conv2'] = ops.conv2d(end_points['batch_norm2'], 16, [3, 3],
                                         scope='conv2', padding='SAME')
        
        end_points['batch_norm3'] = ops.batch_norm(end_points['conv2'], scope='batch_norm3')

        in_net = end_points['batch_norm3']
        print('IN_NET SHAPE')
        print(in_net.get_shape())
        curr_filters = 16
        base_layer_num = [16,12,8,4]
        for i in xrange(1,5):
          for j in xrange(1,base_layer_num[i-1] + i):
            with tf.variable_scope('res%d_%d' % (i,j)):
              if (j < (base_layer_num[i-1] + i - 1)):
                curr_padding = 'SAME'
                curr_stride = 1
              else:
                curr_filters = 2*curr_filters
                curr_padding = 'SAME'
                curr_stride = 2

              conv1_1 = ops.conv2d(in_net, curr_filters, [3, 3], padding=curr_padding, stride=curr_stride, scope='conv1_1')
              batch_norm1_1 = ops.batch_norm(conv1_1, scope='batch_norm1_1')
              conv1_2 = ops.conv2d(batch_norm1_1, curr_filters, [3, 3], padding='SAME', scope='conv1_2')
              if (j < (base_layer_num[i-1] + i - 1)):
                combined = in_net + conv1_2
              else:
                combined = ops.conv2d(in_net, curr_filters, [1, 1], padding='SAME', stride=2, scope='combined')
                combined = combined + conv1_2
                print('DOWN SAMPLE')
                print(in_net.get_shape())
                print(combined.get_shape())
              batch_norm1_2 = ops.batch_norm(combined, scope='batch_norm1_2')
              in_net = batch_norm1_2
              end_points['res%d_%d' %(i,j)] = in_net

#        for i in xrange(1,int(np.log2(in_net.get_shape()[1])) + 1):
#        print('SHAPPEEEE')
        print(in_net.get_shape())
        for i in xrange(1,3):
          with tf.variable_scope('res_final%d' % i):
            conv1_1 = ops.conv2d(in_net, curr_filters, [3, 3], padding='SAME', stride=2, scope='conv1_1')
            batch_norm1_1 = ops.batch_norm(conv1_1, scope='batch_norm1_1')
            conv1_2 = ops.conv2d(batch_norm1_1, curr_filters, [3, 3], padding='SAME', scope='conv1_2')
            combined = ops.conv2d(in_net, curr_filters, [1, 1], padding='SAME', stride=2, scope='combined')
            combined = combined + conv1_2
            batch_norm1_2 = ops.batch_norm(combined, scope='batch_norm1_2')
            in_net = batch_norm1_2
            end_points['res_final%d' % i] = in_net

        with tf.variable_scope('logits'):
          shape = in_net.get_shape()
          print('FINAL SHAPE')
          print(shape)
          if (shape[1] > 1):
            in_net = ops.avg_pool(in_net, shape[1:3], padding='VALID', scope='avg_pool')
          in_net = ops.flatten(in_net, scope='flatten')
          logits = ops.fc(in_net, num_classes, activation=None, scope='logits',
                          restore=restore_logits)
          end_points['logits'] = logits
          end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
          
      return logits, end_points


def inception_v3_parameters(weight_decay=0.00004, stddev=0.1,
                            batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
  """Yields the scope with the default parameters for inception_v3.

  Args:
    weight_decay: the weight decay for weights variables.
    stddev: standard deviation of the truncated guassian weight distribution.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.

  Yields:
    a arg_scope with the parameters needed for inception_v3.
  """
  # Set weight_decay for weights in Conv and FC layers.
  with scopes.arg_scope([ops.conv2d, ops.fc],
                        weight_decay=weight_decay):
    # Set stddev, activation and parameters for batch_norm.
    with scopes.arg_scope([ops.conv2d],
                          stddev=stddev,
                          activation=tf.nn.relu,
                          batch_norm_params={
                              'decay': batch_norm_decay,
                              'epsilon': batch_norm_epsilon}) as arg_scope:
      yield arg_scope
