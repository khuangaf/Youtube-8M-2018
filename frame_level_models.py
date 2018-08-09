# Copyright 2017 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils
import os

import tensorflow.contrib.slim as slim
from tensorflow import flags

import scipy.io as sio
import numpy as np

FLAGS = flags.FLAGS


flags.DEFINE_bool("gating_remove_diag", False,
                  "Remove diag for self gating")
flags.DEFINE_bool("lightvlad", False,
                  "Light or full NetVLAD")
flags.DEFINE_bool("vlagd", False,
                  "vlagd of vlad")



flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 16384,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 2048,
                    "Number of units in the DBoF hidden layer.")
flags.DEFINE_bool("dbof_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("dbof_var_features", 0,
                     "Variance features on top of Dbof cluster layer.")

flags.DEFINE_string("dbof_activation", "relu", 'dbof activation')

flags.DEFINE_bool("softdbof_maxpool", False, 'add max pool to soft dbof')

flags.DEFINE_integer("netvlad_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")
flags.DEFINE_bool("netvlad_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("netvlad_dimred", -1,
                   "NetVLAD output dimension reduction")
flags.DEFINE_integer("gatednetvlad_dimred", 1024,
                   "GatedNetVLAD output dimension reduction")
# flags.DEFINE_integer(
#     "moe_num_mixtures", 2,
#     "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

flags.DEFINE_bool("gating", False,
                   "Gating for NetVLAD")
flags.DEFINE_integer("hidden_size", 1024,
                     "size of hidden layer for BasicStatModel.")


flags.DEFINE_integer("netvlad_hidden_size", 1024,
                     "Number of units in the NetVLAD hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_video", 1024,
                     "Number of units in the NetVLAD video hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_audio", 64,
                     "Number of units in the NetVLAD audio hidden layer.")



flags.DEFINE_bool("netvlad_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")

flags.DEFINE_integer("fv_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")

flags.DEFINE_integer("fv_hidden_size", 2048,
                     "Number of units in the NetVLAD hidden layer.")
flags.DEFINE_bool("fv_relu", True,
                     "ReLU after the NetFV hidden layer.")


flags.DEFINE_bool("fv_couple_weights", True,
                     "Coupling cluster weights or not")
 
flags.DEFINE_float("fv_coupling_factor", 0.01,
                     "Coupling factor")


flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
flags.DEFINE_integer("lstm_cells_video", 1024, "Number of LSTM cells (video).")
flags.DEFINE_integer("lstm_cells_audio", 128, "Number of LSTM cells (audio).")



flags.DEFINE_integer("gru_cells", 1024, "Number of GRU cells.")
flags.DEFINE_integer("gru_cells_video", 1024, "Number of GRU cells (video).")
flags.DEFINE_integer("gru_cells_audio", 128, "Number of GRU cells (audio).")
flags.DEFINE_integer("gru_layers", 2, "Number of GRU layers.")
flags.DEFINE_bool("lstm_random_sequence", False,
                     "Random sequence input for lstm.")
flags.DEFINE_bool("gru_random_sequence", False,
                     "Random sequence input for gru.")
flags.DEFINE_bool("gru_backward", False, "BW reading for GRU")
flags.DEFINE_bool("lstm_backward", False, "BW reading for LSTM")

flags.DEFINE_integer("deep_chain_layers", 3, "Adding FC dimred after pooling")
flags.DEFINE_integer("deep_chain_relu_cells", 128, "Adding FC dimred after pooling")

flags.DEFINE_bool("fc_dimred", True, "Adding FC dimred after pooling")

class LightVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])
       
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad


class NetVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
            [1,self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        a = tf.multiply(a_sum,cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.subtract(vlad,a)
        

        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad


class NetVLAGD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        gate_weights = tf.get_variable("gate_weights",
            [1, self.cluster_size,self.feature_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        gate_weights = tf.sigmoid(gate_weights)

        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])

        vlagd = tf.matmul(activation,reshaped_input)
        vlagd = tf.multiply(vlagd,gate_weights)

        vlagd = tf.transpose(vlagd,perm=[0,2,1])
        
        vlagd = tf.nn.l2_normalize(vlagd,1)

        vlagd = tf.reshape(vlagd,[-1,self.cluster_size*self.feature_size])
        vlagd = tf.nn.l2_normalize(vlagd,1)

        return vlagd




class GatedDBoF():
    def __init__(self, feature_size,max_frames,cluster_size, max_pool, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.max_pool = max_pool

    def forward(self, reshaped_input):

        feature_size = self.feature_size
        cluster_size = self.cluster_size
        add_batch_norm = self.add_batch_norm
        max_frames = self.max_frames
        is_training = self.is_training
        max_pool = self.max_pool

        cluster_weights = tf.get_variable("cluster_weights",
          [feature_size, cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
        
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])

        activation_sum = tf.reduce_sum(activation,1)
        
        activation_max = tf.reduce_max(activation,1)
        activation_max = tf.nn.l2_normalize(activation_max,1)


        dim_red = tf.get_variable("dim_red",
          [cluster_size, feature_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
 
        cluster_weights_2 = tf.get_variable("cluster_weights_2",
          [feature_size, cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
        
        tf.summary.histogram("cluster_weights_2", cluster_weights_2)
        
        activation = tf.matmul(activation_max, dim_red)
        activation = tf.matmul(activation, cluster_weights_2)
        
        if add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="cluster_bn_2")
        else:
          cluster_biases = tf.get_variable("cluster_biases_2",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          tf.summary.histogram("cluster_biases_2", cluster_biases)
          activation += cluster_biases

        activation = tf.sigmoid(activation)

        activation = tf.multiply(activation,activation_sum)
        activation = tf.nn.l2_normalize(activation,1)

        return activation



class SoftDBoF():
    def __init__(self, feature_size,max_frames,cluster_size, max_pool, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.max_pool = max_pool

    def forward(self, reshaped_input):

        feature_size = self.feature_size
        cluster_size = self.cluster_size
        add_batch_norm = self.add_batch_norm
        max_frames = self.max_frames
        is_training = self.is_training
        max_pool = self.max_pool

        cluster_weights = tf.get_variable("cluster_weights",
          [feature_size, cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
        
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])

        activation_sum = tf.reduce_sum(activation,1)
        activation_sum = tf.nn.l2_normalize(activation_sum,1)

        if max_pool:
            activation_max = tf.reduce_max(activation,1)
            activation_max = tf.nn.l2_normalize(activation_max,1)
            activation = tf.concat([activation_sum,activation_max],1)
        else:
            activation = activation_sum
        
        return activation



class DBoF():
    def __init__(self, feature_size,max_frames,cluster_size,activation, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.activation = activation


    def forward(self, reshaped_input):

        feature_size = self.feature_size
        cluster_size = self.cluster_size
        add_batch_norm = self.add_batch_norm
        max_frames = self.max_frames
        is_training = self.is_training

        cluster_weights = tf.get_variable("cluster_weights",
          [feature_size, cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
        
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases

        if activation == 'glu':
            space_ind = range(cluster_size/2)
            gate_ind = range(cluster_size/2,cluster_size)

            gates = tf.sigmoid(activation[:,gate_ind])
            activation = tf.multiply(activation[:,space_ind],gates)

        elif activation == 'relu':
            activation = tf.nn.relu6(activation)
        
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])

        avg_activation = utils.FramePooling(activation, 'average')
        avg_activation = tf.nn.l2_normalize(avg_activation,1)

        max_activation = utils.FramePooling(activation, 'max')
        max_activation = tf.nn.l2_normalize(max_activation,1)
        
        return tf.concat([avg_activation,max_activation],1)

class NetFV():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):
        cluster_weights = tf.get_variable("cluster_weights",
          [self.feature_size, self.cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
     
        covar_weights = tf.get_variable("covar_weights",
          [self.feature_size, self.cluster_size],
          initializer = tf.random_normal_initializer(mean=1.0, stddev=1 /math.sqrt(self.feature_size)))
      
        covar_weights = tf.square(covar_weights)
        eps = tf.constant([1e-6])
        covar_weights = tf.add(covar_weights,eps)

        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [self.cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)
#         print(f'max frames : {self.max_frames}')
#         print(f'cluster size : {self.cluster_size}')        
        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        if not FLAGS.fv_couple_weights:
            cluster_weights2 = tf.get_variable("cluster_weights2",
              [1,self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        else:
            cluster_weights2 = tf.scalar_mul(FLAGS.fv_coupling_factor,cluster_weights)

        a = tf.multiply(a_sum,cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        fv1 = tf.matmul(activation,reshaped_input)
        
        fv1 = tf.transpose(fv1,perm=[0,2,1])

        # computing second order FV
        a2 = tf.multiply(a_sum,tf.square(cluster_weights2)) 

        b2 = tf.multiply(fv1,cluster_weights2) 
        fv2 = tf.matmul(activation,tf.square(reshaped_input)) 
     
        fv2 = tf.transpose(fv2,perm=[0,2,1])
        fv2 = tf.add_n([a2,fv2,tf.scalar_mul(-2,b2)])

        fv2 = tf.divide(fv2,tf.square(covar_weights))
        fv2 = tf.subtract(fv2,a_sum)

        fv2 = tf.reshape(fv2,[-1,self.cluster_size*self.feature_size])
      
        fv2 = tf.nn.l2_normalize(fv2,1)
        fv2 = tf.reshape(fv2,[-1,self.cluster_size*self.feature_size])
        fv2 = tf.nn.l2_normalize(fv2,1)

        fv1 = tf.subtract(fv1,a)
        fv1 = tf.divide(fv1,covar_weights) 

        fv1 = tf.nn.l2_normalize(fv1,1)
        fv1 = tf.reshape(fv1,[-1,self.cluster_size*self.feature_size])
        fv1 = tf.nn.l2_normalize(fv1,1)

        return tf.concat([fv1,fv2],1)

class NetVLADModelLF(models.BaseModel):
  """Creates a NetVLAD based model.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.netvlad_cluster_size
    hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
    relu = FLAGS.netvlad_relu
    dimred = FLAGS.netvlad_dimred
    gating = FLAGS.gating
    remove_diag = FLAGS.gating_remove_diag
    lightvlad = FLAGS.lightvlad
    vlagd = FLAGS.vlagd

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    

    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])

    if lightvlad:
      video_NetVLAD = LightVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = LightVLAD(128,max_frames,cluster_size//2, add_batch_norm, is_training)
    elif vlagd:
      video_NetVLAD = NetVLAGD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = NetVLAGD(128,max_frames,cluster_size//2, add_batch_norm, is_training)
    else:
      video_NetVLAD = NetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = NetVLAD(128,max_frames,cluster_size//2, add_batch_norm, is_training)

  
    if add_batch_norm:# and not lightvlad:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(reshaped_input[:,0:1024]) 

    with tf.variable_scope("audio_VLAD"):
        vlad_audio = audio_NetVLAD.forward(reshaped_input[:,1024:])

    vlad = tf.concat([vlad_video, vlad_audio],1)

    vlad_dim = vlad.get_shape().as_list()[1] 
    
    
    
    hidden1_weights = tf.get_variable("hidden1_weights",
      [vlad_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
       
    activation = tf.matmul(vlad, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")

    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
   
    if relu:
      activation = tf.nn.relu6(activation)
   

    if gating:
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))
        
        gates = tf.matmul(activation, gating_weights)
 
        if remove_diag:
            #removes diagonals coefficients
            diagonals = tf.matrix_diag_part(gating_weights)
            gates = gates - tf.multiply(diagonals,activation)

       
        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)

        
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)


    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)
  


class NetVLADModelLFChain(models.BaseModel):
  """Creates a NetVLADChain based model.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """
    
  def sub_moe(self, model_input, vocab_size, num_mixtures=None, 
                l2_penalty=1e-8, sub_scope="", **unused_params):
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates-"+sub_scope)
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts-"+sub_scope)

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return final_probabilities

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.netvlad_cluster_size
    hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
    relu = FLAGS.netvlad_relu
    dimred = FLAGS.netvlad_dimred
    gating = FLAGS.gating
    remove_diag = FLAGS.gating_remove_diag
    lightvlad = FLAGS.lightvlad
    vlagd = FLAGS.vlagd

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    



    num_layers = FLAGS.deep_chain_layers
    relu_cells = FLAGS.deep_chain_relu_cells
    
    
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])

    if lightvlad:
      video_NetVLAD = LightVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = LightVLAD(128,max_frames,cluster_size//2, add_batch_norm, is_training)
    elif vlagd:
      video_NetVLAD = NetVLAGD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = NetVLAGD(128,max_frames,cluster_size//2, add_batch_norm, is_training)
    else:
      video_NetVLAD = NetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = NetVLAD(128,max_frames,cluster_size//2, add_batch_norm, is_training)
    
    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(reshaped_input[:,0:1024]) 

    with tf.variable_scope("audio_VLAD"):
        vlad_audio = audio_NetVLAD.forward(reshaped_input[:,1024:])

    vlad = tf.concat([vlad_video, vlad_audio],1)

    vlad_dim = vlad.get_shape().as_list()[1] 
    hidden1_weights = tf.get_variable("hidden1_weights",
      [vlad_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
       
    activation = tf.matmul(vlad, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")

    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
   
    if relu:
      activation = tf.nn.relu6(activation)
   

    if gating:
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))
        
        gates = tf.matmul(activation, gating_weights)
 
        if remove_diag:
            #removes diagonals coefficients
            diagonals = tf.matrix_diag_part(gating_weights)
            gates = gates - tf.multiply(diagonals,activation)

       
        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)
        
    next_input = activation
    support_predictions = []
    sub_scope = 'chain_'
    for layer in range(num_layers):
      sub_prediction = self.sub_moe(next_input, vocab_size, sub_scope=sub_scope+"prediction-%d"%layer)
      sub_relu = slim.fully_connected(
          sub_prediction,
          relu_cells,
          activation_fn=tf.nn.relu,
          weights_regularizer=slim.l2_regularizer(1e-8),
          scope=sub_scope+"relu-%d"%layer)
      relu_norm = tf.nn.l2_normalize(sub_relu, dim=1)
      if lightvlad:
        video_NetVLAD = LightVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
        audio_NetVLAD = LightVLAD(128,max_frames,cluster_size//2, add_batch_norm, is_training)
      elif vlagd:
        video_NetVLAD = NetVLAGD(1024,max_frames,cluster_size, add_batch_norm, is_training)
        audio_NetVLAD = NetVLAGD(128,max_frames,cluster_size//2, add_batch_norm, is_training)
      else:
        video_NetVLAD = NetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
        audio_NetVLAD = NetVLAD(128,max_frames,cluster_size//2, add_batch_norm, is_training)

      with tf.variable_scope(f"video_VLAD_{layer}"):
          vlad_video = video_NetVLAD.forward(reshaped_input[:,0:1024]) 

      with tf.variable_scope(f"audio_VLAD_{layer}"):
          vlad_audio = audio_NetVLAD.forward(reshaped_input[:,1024:])

      vlad = tf.concat([vlad_video, vlad_audio],1)

      vlad_dim = vlad.get_shape().as_list()[1] 
      hidden1_weights = tf.get_variable(f"hidden1_weights_{layer}",
        [vlad_dim, hidden1_size],
        initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))

      activation = tf.matmul(vlad, hidden1_weights)

      if add_batch_norm and relu:
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope=f"hidden1_bn_{layer}")

      else:
        hidden1_biases = tf.get_variable(f"hidden1_biases_{layer}",
          [hidden1_size],
          initializer = tf.random_normal_initializer(stddev=0.01))
        tf.summary.histogram(f"hidden1_biases_{layer}", hidden1_biases)
        activation += hidden1_biases

      if relu:
        activation = tf.nn.relu6(activation)


      if gating:
          gating_weights = tf.get_variable(f"gating_weights_2_{layer}",
            [hidden1_size, hidden1_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))

          gates = tf.matmul(activation, gating_weights)

          if remove_diag:
              #removes diagonals coefficients
              diagonals = tf.matrix_diag_part(gating_weights)
              gates = gates - tf.multiply(diagonals,activation)


          if add_batch_norm:
            gates = slim.batch_norm(
                gates,
                center=True,
                scale=True,
                is_training=is_training,
                scope=f"gating_bn_{layer}")
          else:
            gating_biases = tf.get_variable(f"gating_biases_{layer}",
              [cluster_size],
              initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
            gates += gating_biases

          gates = tf.sigmoid(gates)

          activation = tf.multiply(activation,gates)
      next_input = tf.concat([activation, relu_norm], axis=1)
    main_predictions = self.sub_moe(next_input, vocab_size, sub_scope=sub_scope+"-main")

    
    return {"predictions": main_predictions}

    
class NetFVModelLF(models.BaseModel):
  """Creates a NetFV based model.
     It emulates a Gaussian Mixture Fisher Vector pooling operations

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.fv_cluster_size
    hidden1_size = hidden_size or FLAGS.fv_hidden_size
    relu = FLAGS.fv_relu
    gating = FLAGS.gating

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    video_NetFV = NetFV(1024,max_frames,cluster_size, add_batch_norm, is_training)
    audio_NetFV = NetFV(128,max_frames,cluster_size//2, add_batch_norm, is_training)


    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_FV"):
        fv_video = video_NetFV.forward(reshaped_input[:,0:1024]) 

    with tf.variable_scope("audio_FV"):
        fv_audio = audio_NetFV.forward(reshaped_input[:,1024:])

    fv = tf.concat([fv_video, fv_audio],1)

    fv_dim = fv.get_shape().as_list()[1] 
    hidden1_weights = tf.get_variable("hidden1_weights",
      [fv_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    
    activation = tf.matmul(fv, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
   
    if relu:
      activation = tf.nn.relu6(activation)

    if gating:
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))
        
        gates = tf.matmul(activation, gating_weights)
        
        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)


    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)

# class NetVLADModelLFMultiscale(models.BaseModel):

#   """Creates a NetVLAD based model.

#   Args:
#     model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
#                  input features.
#     vocab_size: The number of classes in the dataset.
#     num_frames: A vector of length 'batch' which indicates the number of
#          frames for each video (before padding).

#   Returns:
#     A dictionary with a tensor containing the probability predictions of the
#     model in the 'predictions' key. The dimensions of the tensor are
#     'batch_size' x 'num_classes'.
#   """

  
#   def cnn(self, 
#           model_input, 
#           l2_penalty=1e-8, 
#           num_filters = [1024, 1024, 1024],
#           filter_sizes = [1,2,3], 
#           sub_scope="",
#           is_training=True,
#           **unused_params):
#     max_frames = model_input.get_shape().as_list()[1]
#     num_features = model_input.get_shape().as_list()[2]

#     shift_inputs = []
#     for i in range(max(filter_sizes)):
#       if i == 0:
#         shift_inputs.append(model_input)
#       else:
#         shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

#     cnn_outputs = []
#     for nf, fs in zip(num_filters, filter_sizes):
#       sub_input = tf.concat(shift_inputs[:fs], axis=2)
#       sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs, 
#                        shape=[num_features*fs, nf], dtype=tf.float32, 
#                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), 
#                        regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
#       cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

#     cnn_output = tf.concat(cnn_outputs, axis=2)
#     cnn_output = slim.batch_norm(
#         cnn_output,
#         center=True,
#         scale=True,
#         is_training=is_training,
#         scope=sub_scope+"cluster_bn")
#     return cnn_output

#   def moe(self,model_input,
#           vocab_size,
#           num_mixtures=None,
#           l2_penalty=1e-8,
#           scopename="",
#           **unused_params):

#     num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

#     gate_activations = slim.fully_connected(
#         model_input,
#         vocab_size * (num_mixtures + 1),
#         activation_fn=None,
#         biases_initializer=None,
#         weights_regularizer=slim.l2_regularizer(l2_penalty),
#         scope="gates"+scopename)
#     expert_activations = slim.fully_connected(
#         model_input,
#         vocab_size * num_mixtures,
#         activation_fn=None,
#         weights_regularizer=slim.l2_regularizer(l2_penalty),
#         scope="experts"+scopename)

#     gating_distribution = tf.nn.softmax(tf.reshape(
#         gate_activations,
#         [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
#     expert_distribution = tf.nn.sigmoid(tf.reshape(
#         expert_activations,
#         [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


#     final_probabilities_by_class_and_batch = tf.reduce_sum(
#         gating_distribution[:, :num_mixtures] * expert_distribution, 1)

#     final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
#                                      [-1, vocab_size])

#     return final_probabilities

#   def create_model(self,
#                    model_input,
#                    vocab_size,
#                    num_frames,
#                    iterations=None,
#                    add_batch_norm=None,
#                    sample_random_frames=None,
#                    cluster_size=None,
#                    hidden_size=None,
#                    is_training=True,
#                    **unused_params):
#     iterations = iterations or FLAGS.iterations
#     add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
#     random_frames = sample_random_frames or FLAGS.sample_random_frames
#     cluster_size = cluster_size or FLAGS.netvlad_cluster_size
#     hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
#     relu = FLAGS.netvlad_relu
#     dimred = FLAGS.netvlad_dimred
#     gating = FLAGS.gating
#     remove_diag = FLAGS.gating_remove_diag
#     lightvlad = FLAGS.lightvlad
#     vlagd = FLAGS.vlagd

#     num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
#     if random_frames:
#       model_input = utils.SampleRandomFrames(model_input, num_frames,
#                                              iterations)
#     else:
#       model_input = utils.SampleRandomSequence(model_input, num_frames,
#                                                iterations)
    


# #     feature_size = model_input.get_shape().as_list()[2]
# #     reshaped_input = tf.reshape(model_input, [-1, feature_size])

#     max_frames = model_input.get_shape().as_list()[1]
#     feature_size = model_input.get_shape().as_list()[2]
#     reshaped_input = tf.reshape(model_input, [-1, feature_size])

#     if lightvlad:
        
#       audio_NetVLAD = LightVLAD(128,max_frames,cluster_size//2, add_batch_norm, is_training)
#     elif vlagd:
        
#       audio_NetVLAD = NetVLAGD(128,max_frames,cluster_size//2, add_batch_norm, is_training)
#     else:
        
#       audio_NetVLAD = NetVLAD(128,max_frames,cluster_size//2, add_batch_norm, is_training)

  
#     if add_batch_norm:# and not lightvlad:
#       reshaped_input = slim.batch_norm(
#           reshaped_input,
#           center=True,
#           scale=True,
#           is_training=is_training,
#           scope="input_bn")


#     with tf.variable_scope("audio_VLAD"):
#         vlad_audio = audio_NetVLAD.forward(reshaped_input[:,1024:])
        
#     num_layers = 2

#     pool_size=2
#     num_filters=[128,128]
#     filter_sizes=[1,2]
#     features_size = sum(num_filters)

#     sub_predictions = []
#     cnn_input = model_input[:,:,:1024]

#     cnn_max_frames = cnn_input.get_shape().as_list()[1]

#     for layer in range(num_layers):
# #       cnn_output = self.cnn(cnn_input, num_filters=num_filters, filter_sizes=filter_sizes, is_training=is_training, sub_scope="cnn%d"%(layer+1))
#       cnn_output = cnn_input
#       cnn_output_relu = tf.nn.relu(cnn_output)

        
#       feature_size = cnn_output_relu.get_shape().as_list()[2]
#       reshaped_input = tf.reshape(cnn_output_relu, [-1, feature_size])
# #       lstm_memory = self.rnn(cnn_output_relu, lstm_size, num_frames, sub_scope="rnn%d"%(layer+1))
#       max_frames = cnn_output_relu.get_shape().as_list()[1]

#       if lightvlad:
#         video_NetVLAD = LightVLAD(feature_size,max_frames,cluster_size//(layer+1), add_batch_norm, is_training)
# #         audio_NetVLAD = LightVLAD(128,max_frames,cluster_size//2, add_batch_norm, is_training)
#       elif vlagd:
#         video_NetVLAD = NetVLAGD(feature_size,max_frames,cluster_size//(layer+1), add_batch_norm, is_training)
# #         audio_NetVLAD = NetVLAGD(128,max_frames,cluster_size//2, add_batch_norm, is_training)
#       else:
#         video_NetVLAD = NetVLAD(feature_size,max_frames,cluster_size//(layer+1), add_batch_norm, is_training)
# #         audio_NetVLAD = NetVLAD(128,max_frames,cluster_size//2, add_batch_norm, is_training)

#       with tf.variable_scope(f"video_VLAD_{layer}"):
#           vlad_video = video_NetVLAD.forward(reshaped_input) 

# #       with tf.variable_scope(f"audio_VLAD_{layer}"):
# #           vlad_audio = audio_NetVLAD.forward(reshaped_input[:,1024:])

#       vlad = tf.concat([vlad_video, vlad_audio],1)

    
#       sub_prediction = self.moe(vlad, vocab_size, scopename="moe%d"%(layer+1))
#       sub_predictions.append(sub_prediction)

#       cnn_max_frames //= pool_size
      

#       max_pooled_cnn_output = tf.reduce_max(
#           tf.reshape(
#               cnn_output_relu[:, :int(cnn_max_frames*2), :], 
#               [-1, cnn_max_frames, pool_size, features_size]
#           ), axis=2)

#       # for the next cnn layer
#       cnn_input = max_pooled_cnn_output
#       num_frames = tf.maximum(num_frames/pool_size, 1)

#     support_predictions = tf.concat(sub_predictions, axis=1)
#     predictions = tf.add_n(sub_predictions) / len(sub_predictions)


#     return {"predictions": predictions}

class NetVLADModelLFGRU(models.BaseModel):
  """Creates a NetVLAD based model.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.netvlad_cluster_size
    hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
    relu = FLAGS.netvlad_relu
    dimred = FLAGS.netvlad_dimred
    gating = FLAGS.gating
    remove_diag = FLAGS.gating_remove_diag
    lightvlad = FLAGS.lightvlad
    vlagd = FLAGS.vlagd
    gru_size = FLAGS.gru_cells_audio
    number_of_layers = FLAGS.gru_layers
    
    
    num_frames_ = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames_,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames_,
                                               iterations)
    

    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])

    
    if lightvlad:
      video_NetVLAD = LightVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
#       audio_NetVLAD = LightVLAD(128,max_frames,cluster_size//2, add_batch_norm, is_training)
    elif vlagd:
      video_NetVLAD = NetVLAGD(1024,max_frames,cluster_size, add_batch_norm, is_training)
#       audio_NetVLAD = NetVLAGD(128,max_frames,cluster_size//2, add_batch_norm, is_training)
    else:
      video_NetVLAD = NetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
#       audio_NetVLAD = NetVLAD(128,max_frames,cluster_size//2, add_batch_norm, is_training)

  
    if add_batch_norm:# and not lightvlad:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(reshaped_input[:,0:1024]) 


    stacked_GRU = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(gru_size)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)
    
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_GRU, model_input[:,:,1024:],
                                         sequence_length=num_frames,
                                         dtype=tf.float32)
    vlad = tf.concat([vlad_video, state],1)
                                
    vlad_dim = vlad.get_shape().as_list()[1] 
    hidden1_weights = tf.get_variable("hidden1_weights",
      [vlad_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
       
    activation = tf.matmul(vlad, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")

    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
   
    if relu:
      activation = tf.nn.relu6(activation)
   

    if gating:
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))
        
        gates = tf.matmul(activation, gating_weights)
 
        if remove_diag:
            #removes diagonals coefficients
            diagonals = tf.matrix_diag_part(gating_weights)
            gates = gates - tf.multiply(diagonals,activation)

       
        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)


    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)
