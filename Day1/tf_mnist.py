import tensorflow as tf

# Keras - High level API over tf , theano and  CNTK 
# L1 - CNTK is Microsoft's stuff about NNs

# tf.keras is keras implementation of tensorflow
# config file keras located @ /home/<user>/.keras/keras.json

# Load MNIST
mnist = tf.keras.datasets.mnist

# Train-Test split
(x_train,y_train),(x_test,y_test)=mnist.load_data()

# Model Network Definition
model = tf.keras.models.Sequential([  # Linearly stacks layers
			tf.keras.layers.Flatten(), # Flattens inputs without affecting batch size | params  channels_first = (batch,channels,..), channels_last(default) = (b
			tf.keras.layers.Dense(512,activation=tf.nn.relu),
			tf.keras.layers.Droput(0.2), # turn off 20% nodes in forward pass
			tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer='adam',
			  loss='sparse_categorical_crossentropy', # TODO
			  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
