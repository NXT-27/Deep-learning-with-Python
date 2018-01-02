# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 20:53:37 2017

@author: gchoi
"""
import tensorflow as tf
import numpy as np

tf.reset_default_graph()
x0 = np.ones((3, 3, 3))
w0 = np.arange(6).reshape((3, 2))
x = tf.constant(x0)
y = tf.layers.dense(x, 2)
var_dict = {v.op.name: v for v in tf.global_variables()}
assert(var_dict["dense/kernel"].get_shape() == (3, 2))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.assign(var_dict["dense/kernel"], w0))
sess.run(tf.assign(var_dict["dense/bias"], np.zeros((2,))))

expected_y0 = np.tensordot(x0,w0,axes=[(2,),(0,)])
y0 = sess.run(y)
np.testing.assert_allclose(y0, expected_y0)