import argparse
import traceback
from Logger import Logger
import pandas as pd
import sklearn.preprocessing as sp
from DeepNetwork import DeepNetwork
from datahelpers import flat2gen

# I/O
parser = argparse.ArgumentParser()
parser.add_argument('-D', '--dataset', type=str, required=True, help='specify the path to the CSV file on which to predict. The csv should have the same feature space on which the loaded model was trained.'
																	 'It\'s also possible to have a target column in the dataset, but make sure that it\'s the first.')
parser.add_argument('-t', '--test', action='store_true', help='use the first column of the dataset as target value against which to test the model\'s performance')
parser.add_argument('-l', '--load', type=str, required=True, help='load the neural network from the given file path')
parser.add_argument('-d', '--debug', action='store_true', help='do not print anything to file and do not create the output folder')
args = parser.parse_args()
if args.debug:
	print 'WARNING: debug mode is enabled, output will not be saved.'
logger = Logger(debug=args.debug)

data = pd.read_csv(args.dataset, header=0) # Read dataset from CSV
scaled_data = pd.DataFrame(sp.scale(data, with_mean=False), columns=data.keys()) # Normalize data

if args.test:
	y = scaled_data[scaled_data.columns[0]].as_matrix()
	x = scaled_data.drop(scaled_data.columns[0], axis=1).as_matrix()
else:
	x = scaled_data.as_matrix()

metrics_file = 'metrics.csv'
prediction_file = 'predictions.csv'

input_shape = x.shape[1:]
output = 1

try:
	DN = DeepNetwork(input_shape, output, logger=logger, load_path=args.load)
except ValueError, e:
	traceback.print_exc()
	raise type(e)(e.message + '\n\nLooks like you\'re trying to feed the wrong data to the model.\n'
							  'Make sure that the dataset which you are using has the same amount of features (i.e. columns which do not contain targets) as the original dataset used for training.\n'
							  'If you are using a dataset which contains target values as well as features, make sure that:\n'
							  '\t1. The target values are in the first column of the dataset\n'
							  '\t2. The -t, --test flag is set')


logger.to_csv(metrics_file, DN.model.metrics_names) # Add headers to the output file
logger.to_csv(prediction_file, ['predicted_value'] + list(scaled_data.keys())) # Add headers to the output file

# Predict on validation set and write to file
prediction = DN.predict(x)
for p in zip(prediction, list(y), list(x)) if args.test else zip(prediction, list(x)):
	p = list(flat2gen(p))
	logger.to_csv(prediction_file, p)

# Test performance on validation set and write to file
if args.test:
	metrics = DN.test(x, y)
	logger.log("Test loss: %s" % metrics[0])
	logger.to_csv(metrics_file, metrics)

logger.log('Done. Exiting..')
