from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import numpy as np


class DeepNetwork:
	
	def __init__(self, output_layer, input_shape, learning_rate=0.1, dropout_prob=0.1, load_path=None, logger=None):
		self.model = Sequential()
		self.output_layer = output_layer  # Size of the network output
		self.learning_rate = learning_rate
		self.dropout_prob = dropout_prob

		# Define neural network
		self.model.add(BatchNormalization(axis=1, input_shape=input_shape))
		self.model.add(Convolution2D(32, 2, 2, border_mode='valid', subsample=(2, 2)))
		self.model.add(Activation('relu'))

		self.model.add(Flatten())

		self.model.add(BatchNormalization(mode=1))
		self.model.add(Dropout(self.dropout_prob))
		self.model.add(Dense(1024))
		self.model.add(Activation('relu'))

		self.model.add(Dense(self.output_layer))
		self.model.add(Activation('relu'))

		self.optimizer = Adam()
		self.logger = logger

		# Load the network from saved model
		if load_path is not None:
			self.load(load_path)

		self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['accuracy'])

	def train(self, x, t):
		x = np.asarray(x)
		t = np.asarray(t)
		return self.model.train_on_batch(x, t)

	def predict(self, x):
		# Feed input to the model, return predictions
		x = np.asarray(x)
		return self.model.predict_on_batch(x)

	def test(self, x, t):
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
