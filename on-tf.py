import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random as rd
from pylab import *
train_x = np.linspace(-100,100,2000)
train_y = np.array([3*i+rd.uniform(-i/10,i/10) for i in train_x])
model = tf.keras.Sequential()
model.add(layers.Dense(1,input_shape=(1,)))
model.compile(optimizer='adam',
             loss='mse',
             )
print(model.summary())
model.fit(train_x, train_y, epochs=1000)
test_x = np.linspace(200,300,5)
result = model.predict(test_x)
print(result)
plot(train_x,train_y,test_x,result)
show()
