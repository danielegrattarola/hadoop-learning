# Hadoop Learning

This is the project for the Coumputing Infrastructure course of Politecnico di Milano, academic year 2015/2016. The scripts in the repository implement a deep neural network to approximate the relationship between the performance on map and reduce tasks and the completion time of Hive queries run on distributed Hadoop systems.  
  
The Apache Hadoop framework is an open-source software framework for distributed storage and distributed processing of very large data sets on computer clusters built from commodity hardware.  
Apache Hive is a data warehouse infrastructure built on top of Hadoop for providing data summarization, query, and analysis. Hive gives an SQL-like interface to query data stored in various databases and file systems that integrate with Hadoop.  
Due to the possibly great complexity of the distributed systems on which Hive queries are run, and enormous sizes of modern datawarehouses, it is often of interest to have insights on the performances obtained on specific types of queries.  
  
You can find a detailed recount of my experiments with these scripts and dataset [here](http://exsubstantia.com/ai/Estimating%20performance%20of%20Hadoop%20systems%20with%20deep%20learning.pdf.zip).

## Dependencies
The dependencies for the scripts include: [Keras](http://keras.io/#installation), [scikit-learn](http://scikit-learn.org/stable/install.html), [Pandas](http://pandas.pydata.org/), [h5py](http://packages.ubuntu.com/trusty/python-h5py), [Numpy](https://www.scipy.org/scipylib/download.html) and [Matplotlib](http://matplotlib.org/users/installing.html) (these last two are also dependecies of Keras).   

**WARNING**: since I use the [TensorFlow](https://www.tensorflow.org/versions/r0.10/get_started/index.html) backend for Keras, I performed some memory management which is strictly related to TF. I can't guarantee that the scripts will work on the Theano backend.  

## Setup

To setup the working environmet for the scripts, first run:
```sh
git clone https://gitlab.com/danielegrattarola/hadoop-learning.git
cd hadoop-learning
```  
to download the source code.  

### Data
  
The data used to fit the models consists in eight different datasets, each associated to a specific Hive query run on different hardware configurations. The queries are divided into simple (queries R1, R2, R3, R4, and R5) and complex (queries Q2, Q3, and Q4) and of each query was recorded the completion time on different systems.  
  
The scripts will look for the datasets in the `data` folder, and **the datasets need to be CSV files with the name of the type of query (e.g. R1.csv)**.    
The data folder needs to be added manually with: 
```sh
mkdir data
```   
I cannot provide a download script for the datasets as I do not own the data, but you can contact me if you really need it.   
All other necessary folders will be automatically created by the scripts at runtime.   

## Usage

There are three main scripts that can be used for different purposes:
- `train_leave_one_out.py` will train the newtork using leave-one-out cross validation and output a file with predictions on all datapoints;
- `train_custom.py` requires the user to specify a custom column name and some values, and will perform validation on all datapoints matching this simple query (e.g. validate on all rows with 20 and 40 nCores, train on all the other rows);
- `predict.py` will load the neural network from file and will predict the output on all the features passed in the input csv.   

It is also possible to set options to customize the training procedure; possible options are displayed by running:
```sh
python train_leave_one_out.py -h 
python train_custom -h 
python predict.py -h 
```   
  
Some of the most imporant options are:  
- **D, dataset**: specify on which dataset to train and test; possible datasets are R1, R2, R3, R4, R5, Q2, Q3, Q4 (default: R1);
- **s, save**: save the neural network weights to file after training;
- **l, load /path/to/file.h5**: load the neural network from the given file;
- **d, debug**: do not print anything to file and do not create the output folder;
- **epochs**: how many epochs of training should the model train for (default: 5);
- **dropout**: custom dropout rate for the neural network (default: 0.1);   

but you should still check out the script-specific options if you intend to use these tools for your research.  
**NOTE**: some scripts have mandatory options that must be specified (e.g. the load path for `predict.py`)

### Examples
One of the most common use cases for the scripts is the one where a model is trained on some data, and then the saved model is used to predict on new data at design time.   
In this type of scenario, you would want to do something like this:  
```sh
# Train on dataset R2 for 4 epochs and save the resulting weights for the model
python train_leave_one_out.py --dataset R2 --epochs 4 --save
# Predict on new data (note that the dataset flag now takes the full path to the dataset)
python predict.py --dataset /path/to/new_data.csv --load output/runYYYMMDD-hhmmss/model.h5
# Predict on data of which the targets are known (placed in the first column of the csv) and test the performance (**notice the -t flag**)
python predict.py --dataset /path/to/test_data.csv --load output/runYYYMMDD-hhmmss/model.h5 -t 
```  

To evaluate the generalization or interpolation capabilities of the model, instead, we could do something like the following:   
```sh
# Train on dataset R2 and use all datapoints associated to 20 nCores as validation data
python train_custom.py --drop nCores 20 --dataset R2
```  


### Output

By running any of the three scripts, a custom `runYYYMMDD-hhmmss` folder (the name might have more info appended to it, like the feature being dropped or the number of training epochs: this is done to facilitate the workflow during experiments) will be created in the `output` folder. The folder will contain different CSV files with the data suggested by their filename and some plots of the data.  

#### train_leave_one_out
- **cross_validation_prediction_DD.csv** contains rows where the first column is the prediction of the model on the corresponding datapoint (i.e. the rest of the row);
- **cross_validation_metrics_DD.csv** contains the validation loss and accuracy values;
- **log.txt** contains all the information output by the script during execution;  
  
#### train_custom
- **raw_val_predictions_dd.csv** contains rows where the first column is the prediction of the model on the corresponding datapoint (i.e. the rest of the row), computed for all validation set;
- **raw_val_metrics_DD.csv** contains the validation loss and accuracy values for the prediction on the validation set;
- **val_interest_data_DD.csv** contains rows where the model's prediction, the real target value, the nCores feature and the dataSize feature are represented both in the original scale of the data (prefixed by _os_) and the scale used for training the model;
- **val_computed_error_metrics_DD.csv** contains RMSE, mean absolute error and mean average error for the validation predictions, computed both in the original scale of the data (prefixed by _os_) and the scale used for training the model;
- **complTime_v_prediction_R1.png** is a plot of the model predictions vs. the real values of the validation set;
- **log.txt** will contain all the information output by the script during execution; 

#### predict
- **predictions.csv** contains the model's predictions on the dataset being tested, as well as the scaled datapoint associated to each prediction;
- **metrics.csv** contains the test loss and accuracy (only if the -t flag was set, i.e. the tested dataset contains the real target values);
- **log.txt** contains all the information output by the script during execution;  




