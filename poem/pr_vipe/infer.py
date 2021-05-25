# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs pose embedding model inference.

Currently, we support loading model inputs from a CSV file. The CSV file is
expected to have:

1. The first row as header, including the following values:

image/width,
image/height,
image/object/part/NOSE_TIP/center/x,
image/object/part/NOSE_TIP/center/y,
image/object/part/NOSE_TIP/score,
image/object/part/LEFT_SHOULDER/center/x,
image/object/part/LEFT_SHOULDER/center/y,
image/object/part/LEFT_SHOULDER/score,
image/object/part/RIGHT_SHOULDER/center/x,
image/object/part/RIGHT_SHOULDER/center/y,
image/object/part/RIGHT_SHOULDER/score,
image/object/part/LEFT_ELBOW/center/x,
image/object/part/LEFT_ELBOW/center/y,
image/object/part/LEFT_ELBOW/score,
image/object/part/RIGHT_ELBOW/center/x,
image/object/part/RIGHT_ELBOW/center/y,
image/object/part/RIGHT_ELBOW/score,
image/object/part/LEFT_WRIST/center/x,
image/object/part/LEFT_WRIST/center/y,
image/object/part/LEFT_WRIST/score,
image/object/part/RIGHT_WRIST/center/x,
image/object/part/RIGHT_WRIST/center/y,
image/object/part/RIGHT_WRIST/score,
image/object/part/LEFT_HIP/center/x,
image/object/part/LEFT_HIP/center/y,
image/object/part/LEFT_HIP/score,
image/object/part/RIGHT_HIP/center/x,
image/object/part/RIGHT_HIP/center/y,
image/object/part/RIGHT_HIP/score,
image/object/part/LEFT_KNEE/center/x,
image/object/part/LEFT_KNEE/center/y,
image/object/part/LEFT_KNEE/score,
image/object/part/RIGHT_KNEE/center/x,
image/object/part/RIGHT_KNEE/center/y,
image/object/part/RIGHT_KNEE/score,
image/object/part/LEFT_ANKLE/center/x,
image/object/part/LEFT_ANKLE/center/y,
image/object/part/LEFT_ANKLE/score,
image/object/part/RIGHT_ANKLE/center/x,
image/object/part/RIGHT_ANKLE/center/y,
image/object/part/RIGHT_ANKLE/score

2. The following rows are CSVs according to the header, one sample per row.

Note: The input 2D keypoint coordinate values are required to be normalized by
image sizes to within [0, 1].

The outputs will be written to `output_dir` in the format of CSV, with file
base names being the corresponding tensor keys, such as
`unnormalized_embeddings.csv`, `eller_havaya.csv`, etc.

In an output CSV file, each row corresponds to an input sample (the same row in
the input CSV file).

"""

import os

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from poem.core import common
from poem.core import input_generator
from poem.core import keypoint_profiles
from poem.core import keypoint_utils
from poem.core import models
from poem.core import pipeline_utils
from poem.pr_vipe.flags import Flags

tf.disable_v2_behavior()

FLAGS = Flags

def read_inputs(keypoint_profile_2d, width, height, inputs):

  """Reads model inputs."""
  keypoints_2d_col_names, keypoint_scores_2d_col_names = [], []
  for keypoint_name in keypoint_profile_2d.keypoint_names:
    keypoints_2d_col_names.append(
        (common.TFE_KEY_PREFIX_KEYPOINT_2D + keypoint_name +
         common.TFE_KEY_SUFFIX_KEYPOINT_2D[0]))
    keypoints_2d_col_names.append(
        (common.TFE_KEY_PREFIX_KEYPOINT_2D + keypoint_name +
         common.TFE_KEY_SUFFIX_KEYPOINT_2D[1]))
    keypoint_scores_2d_col_names.append(
        (common.TFE_KEY_PREFIX_KEYPOINT_2D + keypoint_name +
         common.TFE_KEY_SUFFIX_KEYPOINT_SCORE))

  inputs = np.array(inputs)
  keypoints_2d_col_names_y = [keypoints_2d_col_names[idx] for idx in range(0, len(keypoints_2d_col_names), 2)]
  keypoints_2d_col_names_x = [keypoints_2d_col_names[idx] for idx in range(1, len(keypoints_2d_col_names), 2)]
  d = {}
  for idx, col_name in enumerate(keypoints_2d_col_names_y):
      d[col_name] = inputs[:, idx, 0].reshape(-1).tolist()

  for idx, col_name in enumerate(keypoints_2d_col_names_x):
      d[col_name] = inputs[:, idx, 1].reshape(-1).tolist()

  for idx, col_name in enumerate(keypoint_scores_2d_col_names):
      d[col_name] = inputs[:, idx, 2].reshape(-1).tolist()
  d[common.TFE_KEY_IMAGE_HEIGHT] = height
  d[common.TFE_KEY_IMAGE_WIDTH] = width

  data = pd.DataFrame(d)

  image_sizes = tf.constant(
      data[[common.TFE_KEY_IMAGE_HEIGHT,
              common.TFE_KEY_IMAGE_WIDTH]].to_numpy(dtype=np.float32))
  keypoints_2d = tf.constant( data[keypoints_2d_col_names].to_numpy(dtype=np.float32))
  keypoint_scores_2d = tf.constant(
      data[keypoint_scores_2d_col_names].to_numpy(dtype=np.float32))

  keypoints_2d = tf.reshape(
      keypoints_2d,
      [-1, keypoint_profile_2d.keypoint_num, keypoint_profile_2d.keypoint_dim])
  keypoints_2d = keypoint_utils.denormalize_points_by_image_size(
      keypoints_2d, image_sizes=image_sizes)

  if FLAGS.min_input_keypoint_score_2d < 0.0:
    keypoint_masks_2d = tf.ones_like(keypoint_scores_2d, dtype=tf.float32)
  else:
    keypoint_masks_2d = tf.cast(
        tf.math.greater_equal(keypoint_scores_2d,
                              FLAGS.min_input_keypoint_score_2d),
        dtype=tf.float32)

  return keypoints_2d, keypoint_masks_2d


def infer(width, height, frames):
  """Runs inference."""
  keypoint_profile_2d = (
      keypoint_profiles.create_keypoint_profile_or_die(
          FLAGS.input_keypoint_profile_name_2d))

  g = tf.Graph()
  with g.as_default():
    keypoints_2d, keypoint_masks_2d = read_inputs(keypoint_profile_2d, width, height, frames)

    model_inputs, _ = input_generator.create_model_input(
        keypoints_2d,
        keypoint_masks_2d=keypoint_masks_2d,
        keypoints_3d=None,
        model_input_keypoint_type=common.MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT,
        keypoint_profile_2d=keypoint_profile_2d,
        # Fix seed for determinism.
        seed=1)

    embedder_fn = models.get_embedder(
        base_model_type=FLAGS.base_model_type,
        embedding_type=FLAGS.embedding_type,
        num_embedding_components=FLAGS.num_embedding_components,
        embedding_size=FLAGS.embedding_size,
        num_embedding_samples=FLAGS.num_embedding_samples,
        is_training=False,
        num_fc_blocks=FLAGS.num_fc_blocks,
        num_fcs_per_block=FLAGS.num_fcs_per_block,
        num_hidden_nodes=FLAGS.num_hidden_nodes,
        num_bottleneck_nodes=FLAGS.num_bottleneck_nodes,
        weight_max_norm=FLAGS.weight_max_norm)

    outputs, _ = embedder_fn(model_inputs)

    if FLAGS.use_moving_average:
      variables_to_restore = (
          pipeline_utils.get_moving_average_variables_to_restore())
      saver = tf.train.Saver(variables_to_restore)
    else:
      saver = tf.train.Saver()

    scaffold = tf.train.Scaffold(
        init_op=tf.global_variables_initializer(), saver=saver)
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold,
        master=FLAGS.master,
        checkpoint_filename_with_path=FLAGS.checkpoint_path)

    with tf.train.MonitoredSession(
        session_creator=session_creator, hooks=None) as sess:
      outputs_result = sess.run(outputs)

    # TODO: should we use mean embeddings instead
    output = outputs_result[common.KEY_EMBEDDING_MEANS]
    out = output.reshape([output.shape[0], -1]).tolist()
    return out


