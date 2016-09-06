import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', action='store_true', help='train the model.')
parser.add_argument('-l', '--load', type=str, required=False, default='', help='load the neural network from the given file path.')
parser.add_argument('-d', '--debug', action='store_true', help='do not print anything to file and do not create the output folder.')
parser.add_argument('--learning_rate', type=float, required=False, default=None, help='custom learning rate for the neural network.')
parser.add_argument('--dropout', type=float, required=False, default=0.1, help='custom dropout rate for the neural network.')
args = parser.parse_args()

if args.debug:
	print 'WARNING: debug mode is enabled, output will not be saved.'
