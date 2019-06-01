# reformat Ctrl+Alt+L
import pandas as pd
import argparse
import functions

parser = argparse.ArgumentParser(
    description='This Programme try to train an RBF Network with the help of ES Algorithm')

parser.add_argument('--input', help='Path of input data', default="training_set.xlsx")
args = parser.parse_args()

# parameters
min_length_chromosome = -1              # minimum number of basis in  each chromosome
max_length_chromosome = -1              # maximum number of basis in  each chromosome
max_sigma_mutation = -1                 # upper bound for sigma that using in mutation
initial_number_chromosomes = -1         # number of initial generation
dataset_length = 0                      # number of data in dataset
algorithm_mode = 'Classification_2'
# indicate running mode of Algorithm {Classification_2, Classification_n, Regression}
regression_threshold = 0.4              # threshold that indicate running mode


# reading data from file with excel format
dataset_train = pd.read_excel(args.input)
training_set = dataset_train.iloc[:, 0:dataset_train.shape[1]].values
dataset_length = dataset_train.shape[0]

# initialization of parameters
algorithm_mode, \
min_length_chromosome,\
max_length_chromosome,\
max_sigma_mutation,\
initial_number_chromosomes= functions.initialization_parameter(dataset_train, regression_threshold)

# generate initial generation
print(algorithm_mode)
