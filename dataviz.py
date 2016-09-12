import matplotlib.pyplot as plt
import os
import pandas
import numpy
import argparse
from sklearn import decomposition
from sklearn import preprocessing as sp
from pandas.tools.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D # Beware of import optimization if you are using an IDE: keep this module even if it seems to be unused


DPI = 250
# Functions
def correlation(dataset, dataset_name):
	# Plot correlation matrix
	print 'Correlation matrix - Dataset %s' % dataset_name
	names = list(dataset.keys())

	# Compute the matrix
	correlations = dataset.corr()

	# Draw the plot
	fig = plt.figure()
	# Maximize window (not cross-platform)
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())

	ax = fig.add_subplot(111)
	cax = ax.matshow(correlations, vmin=-1, vmax=1)
	fig.colorbar(cax)
	ticks = numpy.arange(0,len(names),1)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(names)
	for label in ax.get_xticklabels():
		label.set_rotation(45) # This is needed because labels may overlap if there are too many features in the dataset
	ax.set_yticklabels(names)
	plt.show()

	# Save image to disk
	fig.savefig('./data/viz/correlation_%s.png' % dataset_name, dpi=DPI, bbox_inches='tight')

def PCA(dataset, dataset_name, three_d=False):
	# Plot the target variable against the first one or two PC
	print 'PCA %dD - Dataset %s' % (3 if three_d else 2, dataset_name)
	targets = dataset['complTime']
	features = dataset.drop('complTime', axis=1)

	# Compute PCA
	pca = decomposition.PCA()
	pca.n_components = 2 if three_d else 1
	features_reduced = pca.fit_transform(features)

	# Draw the plot
	fig = plt.figure()
	# Maximize window (not cross-platform)
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())

	if three_d:
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(features_reduced[:,0], features_reduced[:,1], zs=targets)
		ax.set_xlabel('PC1')
		ax.set_ylabel('PC2')
		ax.set_zlabel(targets.name)
	else:
		ax = fig.add_subplot(111)
		ax.scatter(features_reduced[:, 0], targets)
		ax.set_xlabel('PC1')
		ax.set_ylabel(targets.name)

	plt.show()

	# Save image to disk
	fig.savefig('./data/viz/pca_%dD_%s.png' % (3 if three_d else 2, dataset_name), dpi=DPI, bbox_inches='tight')

def scatter(dataset, dataset_name):
	# Compute the scatter matrix with Pandas
	print 'WARNING: scatter matrices will not be automatically saved'
	print 'Scatter matrix - Dataset %s' % dataset_name
	scatter_matrix(dataset)

	# Draw the plot
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	plt.show()


# I/O setup
parser = argparse.ArgumentParser()
parser.add_argument('-D', '--dataset', type=str, required=False, default=None, help='specify on which dataset to operate; possible datasets are R1, R2, R3, R4, R5, Q2, Q3, Q4 (default: all)')
parser.add_argument('--correlation', action='store_true', help='plot correlation matrices for the datasets')
parser.add_argument('--pca', action='store_true', help='plot target values against first PC of features')
parser.add_argument('--pca3d', action='store_true', help='plot target values against first two PCs of features')
parser.add_argument('--scatter', action='store_true', help='plot scatter matrices for the datasets (WARNING: this option does not save the images automatically like the others, so you have to do it by hand)')
args = parser.parse_args()

# Main
output_folder = './data/viz/'
if not os.path.exists(output_folder):
			os.makedirs(output_folder)

for data_name in ['R1', 'R2', 'R3', 'R4', 'R5', 'Q2', 'Q3', 'Q4'] if args.dataset is None else args.dataset.split(','):
	# Read data
	data = pandas.read_csv('./data/%s.csv' % data_name) # Read dataset from CSV
	scaled_data = pandas.DataFrame(sp.scale(data, with_mean=False), columns=data.keys()) # Normalize data

	if args.correlation:
		correlation(scaled_data, data_name)
	if args.pca:
		PCA(scaled_data, data_name)
	if args.pca3d:
		PCA(scaled_data, data_name, three_d=True)
	if args.scatter:
		scatter(scaled_data, data_name)