import argparse
import pandas as pd
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing as sp

from DeepNetwork import DeepNetwork
from Logger import Logger
from datahelpers import flat2gen

# Constants
TARGET_FEATURE_NAME = 'complTime'
CORES_FEATURE_NAME = 'nCores'
DATASIZE_FEATURE_NAME = 'dataSize'
PREDICTION_FEATURE_NAME = 'prediction'

# I/O setup
parser = argparse.ArgumentParser()
parser.add_argument('--drop', nargs='+', required=True, default=None, help='drop all rows in the dataset where the column passed as first value matches the values passed after (e.g. --drop nCores 20 60) and use them as validation set')
parser.add_argument('-D', '--dataset', type=str, required=False, default='R1', help='specify on which dataset to train and test; possible datasets are R1, R2, R3, R4, R5, Q2, Q3, Q4 (default: R1)')
parser.add_argument('-s', '--save', action='store_true', help='save the neural network model and weights to file after training')
parser.add_argument('-l', '--load', type=str, required=False, default='', help='load the neural network from the given file path')
parser.add_argument('-d', '--debug', action='store_true', help='do not print anything to file and do not create the output folder')
parser.add_argument('--epochs', type=int, default=5, help='how many epochs of training should the model train for (default: 5)')
parser.add_argument('--dropout', type=float, required=False, default=0.1, help='custom dropout rate for the neural network (default: 0.1)')
parser.add_argument('--accept', type=float, default=None, help='the run will be automatically discarded if the mean relative error is higher than the given threshold')

args = parser.parse_args()
if args.debug:
	print 'WARNING: debug mode is enabled, output will not be saved.'
logger = Logger(debug=args.debug, append='%s_drop_%s_%s' % (args.dataset, '_'.join(args.drop), str(args.epochs)))
logger.log('Dropping: %s' % str(args.drop))
logger.log('Epochs: %d' % args.epochs)
logger.log('Dropout probability: %f' % args.dropout)
logger.log('Dataset: %s' % args.dataset)
logger.log('Starting training...\n')

metrics_file = 'raw_val_metrics_%s.csv' % args.dataset
prediction_file = 'raw_val_predictions_%s.csv' % args.dataset

# Read data
dataset = pd.read_csv('./data/%s.csv' % args.dataset, header=0) # Read dataset from CSV
scaled_dataset = pd.DataFrame(sp.scale(dataset, with_mean=False), columns=dataset.keys()) # Normalize data
if args.drop is not None:
	dataset_train = dataset[~dataset[args.drop[0]].isin(float(i) for i in args.drop[1:])]
	dataset_valid = dataset[dataset[args.drop[0]].isin(float(i) for i in args.drop[1:])]
scaled_dataset_train = scaled_dataset[~dataset[args.drop[0]].isin(float(i) for i in args.drop[1:])]
scaled_dataset_valid = scaled_dataset[dataset[args.drop[0]].isin(float(i) for i in args.drop[1:])]

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
logger.to_csv(prediction_file, [PREDICTION_FEATURE_NAME] + list(dataset.keys())) # Add headers to the output file

# Train the network
DN.train(x_train, y_train, nb_epoch=args.epochs)

# Predict on validation set and write to file
prediction = DN.predict(x_valid)
for p in zip(prediction, list(y_valid), list(x_valid)):
	p = list(flat2gen(p))
	logger.to_csv(prediction_file, p)

# Test performance on validation set and write to file
metrics = DN.test(x_valid, y_valid)
logger.log("Test loss: %s" % metrics[0])
logger.to_csv(metrics_file, metrics)
logger.log('Done. Exiting..')

# Post processing
# Read data
data = pd.read_csv(logger.path + prediction_file, header=0)
prediction_data = data[[PREDICTION_FEATURE_NAME, TARGET_FEATURE_NAME, CORES_FEATURE_NAME, DATASIZE_FEATURE_NAME]]

# Compute the scale factor to translate predictions and features to the original scale of the data
scale_factor_target = dataset_train[TARGET_FEATURE_NAME].iloc[0] / scaled_dataset_train[TARGET_FEATURE_NAME].iloc[0]
scale_factor_cores = dataset_train[CORES_FEATURE_NAME].iloc[0] / scaled_dataset_train[CORES_FEATURE_NAME].iloc[0]
scale_factor_datasize = dataset_train[DATASIZE_FEATURE_NAME].iloc[0] / scaled_dataset_train[DATASIZE_FEATURE_NAME].iloc[0]

# Add original scale features to the dataset
prediction_data['os_' + PREDICTION_FEATURE_NAME] = (prediction_data[PREDICTION_FEATURE_NAME] * scale_factor_target).round()
prediction_data['os_' + TARGET_FEATURE_NAME] = (prediction_data[TARGET_FEATURE_NAME] * scale_factor_target).round()
prediction_data['os_' + CORES_FEATURE_NAME] = (prediction_data[CORES_FEATURE_NAME] * scale_factor_cores).round()
prediction_data['os_' + DATASIZE_FEATURE_NAME] = (prediction_data[DATASIZE_FEATURE_NAME] * scale_factor_datasize).round()

# Compute the error metrics on both scaled and original data
RMSE = np.sqrt(np.mean(np.square(np.absolute(prediction_data[PREDICTION_FEATURE_NAME] - prediction_data[TARGET_FEATURE_NAME]))))
os_RMSE = np.sqrt(np.mean(np.square(np.absolute(prediction_data['os_' + PREDICTION_FEATURE_NAME] - prediction_data['os_' + TARGET_FEATURE_NAME]))))

MAE = np.mean(np.absolute(prediction_data[PREDICTION_FEATURE_NAME] - prediction_data[TARGET_FEATURE_NAME]))
os_MAE = np.mean(np.absolute(prediction_data['os_' + PREDICTION_FEATURE_NAME] - prediction_data['os_' + TARGET_FEATURE_NAME]))

MRE = np.mean(np.true_divide(np.absolute(prediction_data[PREDICTION_FEATURE_NAME] - prediction_data[TARGET_FEATURE_NAME]), prediction_data[TARGET_FEATURE_NAME]))
os_MRE = np.mean(np.true_divide(np.absolute(prediction_data['os_' + PREDICTION_FEATURE_NAME] - prediction_data['os_' + TARGET_FEATURE_NAME]), prediction_data['os_' + TARGET_FEATURE_NAME]))

# Write metrics to file
error_metrics_file = 'val_computed_error_metrics_%s.csv' % args.dataset
logger.to_csv(error_metrics_file, 'RMSE,os_RMSE,MAE,os_MAE,MRE,os_MRE')
logger.to_csv(error_metrics_file, [RMSE, os_RMSE, MAE, os_MAE, MRE, os_MRE])
logger.log('RMSE,MAE,MRE')
logger.log([RMSE, MAE, MRE])
logger.log('os_RMSE,os_MAE,os_MRE')
logger.log([os_RMSE, os_MAE, os_MRE])

# Write predicted data to file
prediction_data.to_csv(path_or_buf=logger.path + 'val_interest_data_%s.csv' % args.dataset, mode='a')

# Remove run folder if run was not good
if args.accept is not None and (args.accept < os_MRE or str(raw_input('Keep run? (Y/n) ')) is 'n'):
	logger.log('#### RUN DISCARDED ####')
	shutil.rmtree(logger.path)
	sys.exit(0)
else:
	# Predicted v. real
	# Draw the plot
	DPI = 250
	x = prediction_data[PREDICTION_FEATURE_NAME]
	y = prediction_data[TARGET_FEATURE_NAME]

	fig = plt.figure()
	# Maximize window (not cross-platform)
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())

	ax = fig.add_subplot(111)
	ax.scatter(x, y)
	ax.set_xlabel(PREDICTION_FEATURE_NAME)
	ax.set_ylabel(TARGET_FEATURE_NAME)
	ax = fig.add_subplot(111)
	ax.plot(np.linspace(0, x.max()*1.1, num=100), np.linspace(0, x.max(), num=100))
	ax.set_ylim(ymin=0)
	ax.set_xlim(xmin=0, xmax=x.max()*1.1)

	plt.show()

	# Save image to disk
	fig.savefig(logger.path+'%s_v_%s_%s.png' % (TARGET_FEATURE_NAME, PREDICTION_FEATURE_NAME, args.dataset), dpi=DPI, bbox_inches='tight')

	# Save the model
	if args.save:
		logger.log('Saving model...')
		DN.save()
	sys.exit(1)

