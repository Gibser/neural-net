import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


inputs = keras.Input(shape=(1,))
dense = layers.Dense(50, activation='sigmoid')(inputs)
outputs = layers.Dense(1)(dense)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()


X = np.random.randn(600)
Y = np.sin(X)

XTest = np.random.randn(100)
YTest = np.sin(XTest)

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001))

history = model.fit(X, Y, batch_size=600, epochs=10000, validation_split=0.2)
#test_scores = model.evaluate(XTest, YTest, verbose=2)

#print("Test loss:", test_scores[0])
#print("Test accuracy:", test_scores[1])