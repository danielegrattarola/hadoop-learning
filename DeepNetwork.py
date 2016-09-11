from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import numpy as np
import keras.backend.tensorflow_backend as b

class DeepNetwork:
	
	def __init__(self, input_shape, output_layer, batch_size=32, learning_rate=0.1, dropout_prob=0.1, load_path=None, logger=None):
		b.clear_session() # This is necessary because TF memory management was done by drunk people
		self.model = Sequential()
		self.output_layer = output_layer  # Size of the network output
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.dropout_prob = dropout_prob

		# Neural network

		self.model.add(Dense(1024, input_shape=input_shape))
		self.model.add(Activation('relu'))
		self.model.add(Dropout(self.dropout_prob))

		self.model.add(Dense(1024))
		self.model.add(Activation('relu'))
		self.model.add(Dropout(self.dropout_prob))

		self.model.add(Dense(1024))
		self.model.add(Activation('relu'))
		self.model.add(Dropout(self.dropout_prob))

		self.model.add(Dense(self.output_layer))
		self.model.add(Activation('relu'))

		self.optimizer = Adam()
		self.logger = logger

		# Load the network from saved model
		if load_path is not None:
			self.load(load_path)

		self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['accuracy'])

	def train(self, x, t, nb_epoch=1, validation_data=None):
		x = np.asarray(x)
		t = np.asarray(t)
		return self.model.fit(x, t, batch_size=self.batch_size, nb_epoch=nb_epoch, validation_data=validation_data)

	def predict(self, x):
		# Feed input to the model, return predictions
		x = np.asarray(x)
		return self.model.predict(x)

	def test(self, x, t):
		x = np.asarray(x)
		t = np.asarray(t)
		return self.model.evaluate(x, t, batch_size=self.batch_size)

	def train_on_batch(self, x, t):
		x = np.asarray(x)
		t = np.asarray(t)
		return self.model.train_on_batch(x, t)

	def predict_on_batch(self, x):
		# Feed input to the model, return predictions
		x = np.asarray(x)
		return self.model.predict_on_batch(x)

	def test_on_batch(self, x, t):
		x = np.asarray(x)
		t = np.asarray(t)
		return self.model.test_on_batch(x, t)

	def save(self, filename=None):
		# Save the model and its weights to disk
		print 'Saving...'
		self.model.save_weights(self.logger.path + ('model.h5' if filename is None else filename))

	def load(self, path):
		# Load the model and its weights from path
		print 'Loading...'
		self.model.load_weights(path)
