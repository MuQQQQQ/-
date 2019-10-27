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
x=x.reshape((60000, 28, 28, 1))
tx=tx.reshape((10000, 28, 28, 1))
ds = tf.data.Dataset.from_tensor_slices((x, y))
ds = ds.map(prepare_mnist_features_and_labels)
ds = ds.batch(100)
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
             loss=tf.keras.losses.sparse_categorical_crossentropy,
             metrics=[tf.keras.metrics.categorical_accuracy,'acc']
             )
print(model.summary())
model.fit(ds,epochs=10)
test = model.evaluate(tx, ty)
print(model.metrics_names)
print(test)
