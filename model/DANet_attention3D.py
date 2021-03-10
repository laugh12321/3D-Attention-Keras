#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Feb 03, 2021 

@file: DANet_attention3D.py
@desc: Dual attention network.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import tensorflow as tf


class Channel_attention(tf.keras.layers.Layer):
    """ 
    Channel attention module 
    
    Fu, Jun, et al. "Dual attention network for scene segmentation." 
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    """
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(Channel_attention, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)
        super(Channel_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()

        proj_query = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                              input_shape[4]))(inputs)
        proj_key = tf.keras.backend.permute_dimensions(proj_query, (0, 2, 1))
        energy = tf.keras.backend.batch_dot(proj_query, proj_key)
        attention = tf.keras.activations.softmax(energy)

        outputs = tf.keras.backend.batch_dot(attention, proj_query)
        outputs = tf.keras.layers.Reshape((input_shape[1], input_shape[2], input_shape[3],
                                           input_shape[4]))(outputs)
        outputs = self.gamma * outputs + inputs

        return outputs


class Position_attention(tf.keras.layers.Layer):
    """ 
    Position attention module 
        
    Fu, Jun, et al. "Dual attention network for scene segmentation." 
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    """
    def __init__(self,
                 ratio = 8,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(Position_attention, self).__init__(**kwargs)
        self.ratio = ratio
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        super(Position_attention, self).build(input_shape)
        self.query_conv = tf.keras.layers.Conv3D(filters=input_shape[-1] // self.ratio, 
                                                 kernel_size=(1, 1, 1), use_bias=False, 
                                                 kernel_initializer='he_normal')
        self.key_conv = tf.keras.layers.Conv3D(filters=input_shape[-1] // self.ratio, 
                                               kernel_size=(1, 1, 1), use_bias=False, 
                                               kernel_initializer='he_normal')
        self.value_conv = tf.keras.layers.Conv3D(filters=input_shape[-1], kernel_size=(1, 1, 1),
                                                 use_bias=False, kernel_initializer='he_normal')
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()

        proj_query = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                              input_shape[4] // self.ratio))(self.query_conv(inputs))
        proj_query = tf.keras.backend.permute_dimensions(proj_query, (0, 2, 1))
        proj_key = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                            input_shape[4] // self.ratio))(self.key_conv(inputs))
        energy = tf.keras.backend.batch_dot(proj_key, proj_query)
        attention = tf.keras.activations.softmax(energy)

        proj_value = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                              input_shape[4]))(self.value_conv(inputs))

        outputs = tf.keras.backend.batch_dot(attention, proj_value)
        outputs = tf.keras.layers.Reshape((input_shape[1], input_shape[2], input_shape[3],
                                           input_shape[4]))(outputs)
        outputs = self.gamma * outputs + inputs

        return outputs
