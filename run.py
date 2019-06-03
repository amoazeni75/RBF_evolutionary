# reformat Ctrl+Alt+L
import pandas as pd
import argparse
import functions

parser = argparse.ArgumentParser(
    description='This Programme try to train an RBF Network with the help of ES Algorithm')

parser.add_argument('--input', help='Path of input data', default="Book1.xlsx")
parser.add_argument('--cminlcr', help='Minimum Ratio of Chromosome Length in Regression', default="0.2")
parser.add_argument('--cmaxlcr', help='Maximum Ratio of Chromosome Length', default="0.4")
parser.add_argument('--init_ch_nu', help='Initial Ratio of Generation', default="0.4")
parser.add_argument('--cmaxs', help='Maximum Ratio of Sigma', default="0.1")
parser.add_argument('--reg_thr', help='threshold that indicate running mode', default="0.4")

args = parser.parse_args()

# parameters
min_length_chromosome = -1              # minimum number of basis in  each chromosome
max_length_chromosome = -1              # maximum number of basis in  each chromosome
max_sigma_mutation = -1                 # upper bound for sigma that using in mutation
initial_number_chromosomes = -1         # number of initial generation
dataset_length = 0                      # number of data in dataset
data_dimension = 0
algorithm_mode = 'Classification_2'
# indicate running mode of Algorithm {Classification_2, Classification_n, Regression}

args.cminlcr = float(args.cminlcr)
args.cmaxlcr = float(args.cmaxlcr)
args.init_ch_nu = float(args.init_ch_nu)
args.cmaxs = float(args.cmaxs)
args.reg_thr = float(args.reg_thr)

# reading data from file with excel format
dataset_train = pd.read_excel(args.input)
training_set = dataset_train.iloc[:, 0:dataset_train.shape[1]].values
dataset_length = dataset_train.shape[0]
data_dimension = dataset_train.shape[1] - 1

# initialization of parameters
algorithm_mode, \
min_length_chromosome,\
max_length_chromosome,\
max_sigma_mutation,\
initial_number_chromosomes= functions.initialization_parameter(data_set=dataset_train,
                                                               regression_threshold=args.reg_thr,
                                                               ratio_min_length_chromosome_reg=args.cminlcr,
                                                               ratio_max_length_chromosome=args.cmaxlcr,
                                                               ratio_max_sigma=args.cmaxs,
                                                               ratio_initial_number_chromosomes=args.init_ch_nu)

# producing initial generation
generation = functions.create_initial_generation(data_set=dataset_train,
                                                         min_length_chromosome=min_length_chromosome,
                                                         max_length_chromosome=max_length_chromosome,
                                                         generation_size=initial_number_chromosomes,
                                                         max_sigma=max_sigma_mutation)

iteration_steps = 1100
for i in range(iteration_steps):
    # selecting parents
    parents = functions.selecting_parents_random_uniform(generation)

    # mutation
    functions.do_mutation(parents, data_dimension)

    # recombination
