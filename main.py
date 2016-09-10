import argparse
import pandas as pd
import sklearn.preprocessing as sp
from sklearn import cross_validation
from Logger import Logger
from DeepNetwork import DeepNetwork


# I/O setup
parser = argparse.ArgumentParser()
parser.add_argument('-D', '--dataset', type=str, required=False, default='R1', help='specify on which dataset to train and test; possible datasets are R1, R2, R3, R4, R5, Q2, Q3, Q4 (default: R1)')
parser.add_argument('-l', '--load', type=str, required=False, default='', help='load the neural network from the given file path')
parser.add_argument('-d', '--debug', action='store_true', help='do not print anything to file and do not create the output folder')
parser.add_argument('--epochs', type=int, default=5, help='how many epochs of training should the model do on each LOO fold (default: 5)')
parser.add_argument('--learning_rate', type=float, required=False, default=None, help='custom learning rate for the neural network')
parser.add_argument('--dropout', type=float, required=False, default=0.1, help='custom dropout rate for the neural network (default: 0.1)')
args = parser.parse_args()
if args.debug:
	print 'WARNING: debug mode is enabled, output will not be saved.'
logger = Logger(debug=args.debug)
metrics_file = 'cross_validation_metrics_%s.csv' % args.dataset
prediction_file = 'cross_validation_prediction_%s.csv' % args.dataset

# Constants
TARGET_FEATURE_NAME = 'complTime'

# Read data
dataset = pd.read_csv('./data/%s.csv' % args.dataset, header=0) # Read dataset from CSV
scaled_dataset = pd.DataFrame(sp.scale(dataset, with_mean=False), columns=dataset.keys()) # Normalize data

X = scaled_dataset.drop(TARGET_FEATURE_NAME, axis=1).as_matrix() # Remove targets from dataset
Y = scaled_dataset[TARGET_FEATURE_NAME].as_matrix()

# Network
input_shape = X.shape[1:]
output = 1
DN = DeepNetwork(input_shape, output, logger=logger)
logger.to_csv(metrics_file, DN.model.metrics_names) # Add headers to the output file
logger.to_csv(prediction_file, ['prediction'] + list(dataset.keys())) # Add headers to the output file


# Main loop
loo_indexes = cross_validation.LeaveOneOut(len(dataset)) # Generate a lists of indexed to split the data
for train_idx, test_idx in loo_indexes:
	logger.log("Fold %d of %d" % (test_idx + 1, len(loo_indexes)))
	x_train, x_test = X[train_idx], X[test_idx] # Split the train and test samples
	y_train, y_test = Y[train_idx], Y[test_idx] # Split the train and test targets

	# Train the network
	DN.train(x_train, y_train, nb_epoch=args.epochs)

	# Cross validation on the left out sample
	logger.log("Testing in cross validation...")
	prediction = DN.predict(x_test)
	logger.to_csv(prediction_file, list(prediction[0] + y_test + x_test[0]))
	metrics = DN.test(x_test, y_test)
	logger.log("Test loss: %s" % metrics[0])
	logger.to_csv(metrics_file, metrics)

	# Reset network to train it again
	logger.log('Done fold, resetting network...\n')
	DN = DeepNetwork(input_shape, output, logger=logger)

logger.log('Done. Exiting..')

