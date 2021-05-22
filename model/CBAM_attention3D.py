#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mar 10, 2021 

@file: CBAM_attention3D.py
@desc: Convolutional Block Attention Module.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import tensorflow as tf


class channel_attention(tf.keras.layers.Layer):
    """ 
    channel attention module 
    
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    def __init__(self, ratio=8, **kwargs):
        self.ratio = ratio
        super(channel_attention, self).__init__(**kwargs)

    def get_config(self):
        config = super(channel_attention, self).get_config().copy()
        config.update({
            'ratio': self.ratio
        })
        return config

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = tf.keras.layers.Dense(channel // self.ratio,
                                                 activation='relu',
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')
        self.shared_layer_two = tf.keras.layers.Dense(channel,
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')
        super(channel_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        channel = inputs.get_shape().as_list()[-1]

        avg_pool = tf.keras.layers.GlobalAveragePooling3D()(inputs)    
        avg_pool = tf.keras.layers.Reshape((1, 1, 1, channel))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = tf.keras.layers.GlobalMaxPooling3D()(inputs)
        max_pool = tf.keras.layers.Reshape((1, 1, 1, channel))(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        feature = tf.keras.layers.Add()([avg_pool, max_pool])
        feature = tf.keras.layers.Activation('sigmoid')(feature)

        return tf.keras.layers.multiply([inputs, feature])


class spatial_attention(tf.keras.layers.Layer):
    """ spatial attention module 
        
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    def __init__(self, kernel_size=7, **kwargs):
        self.kernel_size = kernel_size
        super(spatial_attention, self).__init__(**kwargs)

    def get_config(self):
        config = super(spatial_attention, self).get_config().copy()
        config.update({
            'kernel_size': self.kernel_size
        })
        return config

    def build(self, input_shape):
        self.conv3d = tf.keras.layers.Conv3D(filters=1, kernel_size=self.kernel_size,
                                             strides=1, padding='same', activation='sigmoid',
                                             kernel_initializer='he_normal', use_bias=False)
        super(spatial_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))(inputs)
        max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))(inputs)
        concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
        feature = self.conv3d(concat)	
            
        return tf.keras.layers.multiply([inputs, feature])


def cbam_block(feature, ratio=8, kernel_size=7):
    """
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    feature = channel_attention(ratio=ratio)(feature)
    feature = spatial_attention(kernel_size=kernel_size)(feature)

    return feature
    