import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random as rd
from pylab import *
def prepare_mnist_features_and_labels(x, y):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)
  return x, y
(x,y),(tx,ty) = tf.keras.datasets.mnist.load_data()
ds = tf.data.Dataset.from_tensor_slices((x, y))
ds = ds.map(prepare_mnist_features_and_labels)
ds = ds.take(20000).shuffle(20000).batch(100)
model = tf.keras.Sequential()
model.add(layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),)
model.add(layers.Dense(30,input_shape=(784,1),activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer='adam',
             loss=tf.keras.losses.sparse_categorical_crossentropy,
             metrics=[tf.keras.metrics.categorical_accuracy,'acc']
             )
print(model.summary())
print(x.shape,y.shape)
model.fit(ds,epochs=20)
