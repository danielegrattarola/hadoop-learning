### Hadoop Learning

This is the project for the Coumputing Infrastructure course of Politecnico di Milano, academic year 2015/2016.  
The scripts in the repository implement a deep neural network to approximate the completion time of Hive queries run on distributed Hadoop systems.  
The Apache Hadoop framework is an open-source software framework for distributed storage and distributed processing of very large data sets on computer clusters built from commodity hardware.  
Apache Hive is a data warehouse infrastructure built on top of Hadoop for providing data summarization, query, and analysis. Hive gives an SQL-like interface to query data stored in various databases and file systems that integrate with Hadoop.  
Due to the possibly great complexity of the distributed systems on which Hive queries are run, and enormous sizes of modern datawarehouses, it is often of interest to have insights on the performances obtained on specific types of queries.  
You can find a detailed recount of my experiments with these scripts and dataset [here](http://exsubstantia.com/ai/Estimating%20performance%20of%20Hadoop%20systems%20with%20deep%20learning.pdf.zip).

# Dependencies
The dependencies for the scripts include: [kears](http://keras.io/#installation), [scikit-learn](http://scikit-learn.org/stable/install.html), [pandas](http://pandas.pydata.org/), [h5py](http://packages.ubuntu.com/trusty/python-h5py), Numpy and Matplotlib (these last two are also dependecies of Keras).   

**WARNING**: since I use the [Tensorflow](https://www.tensorflow.org/versions/r0.10/get_started/index.html) backend for Keras, I performed some memory management which is realted to TF. I can't guarantee that the scripts will work on the Theano backend.  

# Setup

To setup the working environmet for the scripts, first run:
```sh
git clone https://gitlab.com/danielegrattarola/hadoop-learning.git
cd hadoop-learning
```  
to download the source code. 
The data used to fit the models consists in eight different datasets, each associated to a specific Hive query run on different hardware configurations.
The queries are divided into simple (queries R1, R2, R3, R4, and R5) and complex (queries Q2, Q3, and Q4) and of each query was recorded the completion time on different systems.  

The scripts will look for the datasets in the `data` folder, and **the datasets need to be CSV files with the name of the type of query (e.g. R1.csv)**.    
The data folder needs to be added manually with: 
```sh
mkdir data
```   
I cannot provide a download script for the datasets as I do not own the data, but you can contact me if you really need it.   
All other necessary folders will be automatically created by the scripts at runtime.   

# Running the script

There are three main scripts that can be used for different purposes:
- `train_leave_one_out.py` will train the newtork using leave-one-out cross validation and output a file with predictions on all datapoints
- `train_custom_cv.py` requires the user to specify a custom column name and some values, and will perform validation on all datapoints matching this simple query (e.g. validate on all rows with 20 and 40 nCores)
- `predict.py` will load the neural network from file and will predict the output on all the features passed in the input csv   

It is also possible to set options to customize the training procedure; possible options are displayed by running:
```sh
python train_leave_one_out.py -h 
python train_custom_cv -h 
python predict.py -h 
```   
  
Some of the most imporant options are:  
- **D, dataset**: specify on which dataset to train and test; possible datasets are R1, R2, R3, R4, R5, Q2, Q3, Q4 (default: R1)
- **s, save**: save the neural network model and weights to file after training
- **l, load /path/to/file.h5**: load the neural network from the given file path
- **d, debug**: do not print anything to file and do not create the output folder
- **epochs**: how many epochs of training should the model train for (default: 5)
- **dropout**: custom dropout rate for the neural network (default: 0.1)   
but you should still check the script-specific options if you intend to use these tools for your research.   

# Output

By running any of the three scripts, a custom `runYYYMMDD-hhmmss` folder will be created in the `output` folder. The folder will contain different CSV files with the data suggested by their filename and some plots of the data.  
See the help section of the scripts for more details on how to customize this output. 



