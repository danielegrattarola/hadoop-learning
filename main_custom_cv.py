import argparse
import pandas as pd
import sklearn.preprocessing as sp
from sklearn import cross_validation
from sklearn import decomposition
from Logger import Logger
from DeepNetwork import DeepNetwork
from datahelpers import flat2gen


# I/O setup
parser = argparse.ArgumentParser()
parser.add_argument('--drop', nargs='+', required=True, default=None, help='drop all raws in the dataset where the column passed as first value matches the values passed after (e.g. --drop nCores 20 60) and use them as validation set')
parser.add_argument('-D', '--dataset', type=str, required=False, default='R1', help='specify on which dataset to train and test; possible datasets are R1, R2, R3, R4, R5, Q2, Q3, Q4 (default: R1)')
parser.add_argument('-l', '--load', type=str, required=False, default='', help='load the neural network from the given file path')
parser.add_argument('-d', '--debug', action='store_true', help='do not print anything to file and do not create the output folder')
parser.add_argument('--epochs', type=int, default=5, help='how many epochs of training should the model do on each LOO fold (default: 5)')
parser.add_argument('--dropout', type=float, required=False, default=0.1, help='custom dropout rate for the neural network (default: 0.1)')

args = parser.parse_args()
if args.debug:
	print 'WARNING: debug mode is enabled, output will not be saved.'
logger = Logger(debug=args.debug, append='%s_drop_%s_%s' % (args.dataset, '_'.join(args.drop), str(args.epochs)))
logger.log('Epochs: %d' % args.epochs)
logger.log('Dropout probability: %f' % args.dropout)
logger.log('Dataset: %s' % args.dataset)
logger.log('Dropping: %s' % str(args.drop))
logger.log('Starting training...\n')

metrics_file = 'cross_validation_metrics_%s.csv' % args.dataset
prediction_file = 'cross_validation_prediction_%s.csv' % args.dataset

# Constants
TARGET_FEATURE_NAME = 'complTime'

# Read data
dataset = pd.read_csv('./data/%s.csv' % args.dataset, header=0) # Read dataset from CSV
if args.drop is not None:
	dataset_train = dataset[~dataset[args.drop[0]].isin(float(i) for i in args.drop[1:])]
	dataset_valid = dataset[dataset[args.drop[0]].isin(float(i) for i in args.drop[1:])]
scaled_dataset_train = pd.DataFrame(sp.scale(dataset_train, with_mean=False), columns=dataset_train.keys()) # Normalize data
scaled_dataset_valid = pd.DataFrame(sp.scale(dataset_valid, with_mean=False), columns=dataset_valid.keys()) # Normalize data

logger.log('Training set: %s' % str(dataset_train.shape))
logger.log('Validation set: %s' % str(dataset_valid.shape))

# Split the train and validation sets and labels
x_train = scaled_dataset_train.drop(TARGET_FEATURE_NAME, axis=1).as_matrix()
x_valid = scaled_dataset_valid.drop(TARGET_FEATURE_NAME, axis=1).as_matrix()
y_train = scaled_dataset_train[TARGET_FEATURE_NAME].as_matrix()
y_valid = scaled_dataset_valid[TARGET_FEATURE_NAME].as_matrix()


# Network
input_shape = x_train.shape[1:]
output = 1
DN = DeepNetwork(input_shape, output, logger=logger)
logger.to_csv(metrics_file, DN.model.metrics_names) # Add headers to the output file
logger.to_csv(prediction_file, ['prediction'] + list(dataset.keys())) # Add headers to the output file


# Training
# Train the network
DN.train(x_train, y_train, nb_epoch=args.epochs)

prediction = DN.predict(x_valid)
for p in zip(prediction, list(y_valid), list(x_valid)):
	p = list(flat2gen(p))
	logger.to_csv(prediction_file, p)
metrics = DN.test(x_valid, y_valid)
logger.log("Test loss: %s" % metrics[0])
logger.to_csv(metrics_file, metrics)

# Reset network to train it again
logger.log('Done fold, resetting network...\n')
DN = DeepNetwork(input_shape, output, logger=logger)

logger.log('Done. Exiting..')

