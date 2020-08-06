# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

"""Core model definition of YAMNet."""

import csv

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

import features as features_lib
import params
from tensorflow.keras.applications import MobileNetV2


def yamnet():
    """Define the core YAMNet mode in Keras."""
    waveform = layers.Input(batch_shape=(None, params.PATCH_FRAMES, params.PATCH_BANDS))
    net_inp = layers.Reshape(
        (params.PATCH_FRAMES, params.PATCH_BANDS, 1),
        input_shape=(params.PATCH_FRAMES, params.PATCH_BANDS))(waveform)
    model_input =  Model(inputs=waveform, outputs=net_inp)
    net = MobileNetV2(input_tensor=model_input.output,
                      alpha=1.0,
                      include_top=False,
                      weights=None)
    net = layers.GlobalAveragePooling2D()(net._layers[-1].output)
    logits = layers.Dense(units=params.NUM_CLASSES, use_bias=True)(net)
    predictions = layers.Activation(
        name=params.EXAMPLE_PREDICTIONS_LAYER_NAME,
        activation=params.CLASSIFIER_ACTIVATION)(logits)

    model = Model(name='yamnet', 
                  inputs=waveform, outputs=predictions)
    return model

def yamnet_predict(features):
  """Define the core YAMNet mode in Keras."""
  net_inp = layers.Reshape(
      (params.PATCH_FRAMES, params.PATCH_BANDS, 1),
      input_shape=(params.PATCH_FRAMES, params.PATCH_BANDS))(features)
  #model_input =  Model(inputs=features, outputs=net_inp)
  net = MobileNetV2(input_tensor=net_inp,
                    alpha=1.0,
                    include_top=False,
                    weights=None)
  net = layers.GlobalAveragePooling2D()(net._layers[-1].output)
  logits = layers.Dense(units=params.NUM_CLASSES, use_bias=True)(net)
  predictions = layers.Activation(
      name=params.EXAMPLE_PREDICTIONS_LAYER_NAME,
      activation=params.CLASSIFIER_ACTIVATION)(logits)
  return predictions

def yamnet_frames_model(feature_params):
  """Defines the YAMNet waveform-to-class-scores model.

  Args:
    feature_params: An object with parameter fields to control the feature
    calculation.

  Returns:
    A model accepting (1, num_samples) waveform input and emitting a
    (num_patches, num_classes) matrix of class scores per time frame as
    well as a (num_spectrogram_frames, num_mel_bins) spectrogram feature
    matrix.
  """
  waveform = layers.Input(batch_shape=(1, None))
  # Store the intermediate spectrogram features to use in visualization.
  spectrogram = features_lib.waveform_to_log_mel_spectrogram(
    tf.squeeze(waveform, axis=0), feature_params)
  patches = features_lib.spectrogram_to_patches(spectrogram, feature_params)
  predictions = yamnet_predict(patches)
  frames_model = Model(name='yamnet_frames', 
                       inputs=waveform, outputs=[predictions, spectrogram])
  return frames_model

def class_names(class_map_csv):
  """Read the class name definition file and return a list of strings."""
  with open(class_map_csv) as csv_file:
    reader = csv.reader(csv_file)
    next(reader)   # Skip header
    return np.array([display_name for (_, _, display_name) in reader])