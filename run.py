# reformat Ctrl+Alt+L
import pandas as pd
import argparse
import functions
import es_core
import matplotlib.pyplot as plt


# running mode of Algorithm  = {Classification_2, Classification_n, Regression}


def get_argument():
    parser = argparse.ArgumentParser(
        description='This Programme try to train an RBF Network with the help of ES Algorithm')

    parser.add_argument('--input',
                        help='Path of input data',
                        default="regression_test_sin_2_ud.xlsx")
    parser.add_argument('--cminlcr',
                        help='Minimum Number of Chromosome Length in Regression',
                        default="6")
    parser.add_argument('--cmaxlcr',
                        help='Maximum Ratio of Chromosome Length',
                        default="0.25")
    parser.add_argument('--init_ch_nu',
                        help='Initial Ratio of Generation',
                        default="0.35")
    parser.add_argument('--cmaxs',
                        help='Maximum Ratio of Sigma',
                        default="0.1")
    parser.add_argument('--reg_thr',
                        help='threshold that indicate running mode',
                        default="0.4")
    parser.add_argument('--q',
                        help='q in q-tornument',
                        default="3")
    parser.add_argument('--threads',
                        help='number of running threads',
                        default="10")
    parser.add_argument('--iterations',
                        help='number of iterations in ES',
                        default="30")

    args = parser.parse_args()
    args.cminlcr = int(args.cminlcr)
    args.cmaxlcr = float(args.cmaxlcr)
    args.init_ch_nu = float(args.init_ch_nu)
    args.cmaxs = float(args.cmaxs)
    args.reg_thr = float(args.reg_thr)
    args.q = float(args.q)
    args.threads = int(args.threads)
    args.iterations = int(args.iterations)

    return args


def get_dataset(address):
    dataset_train = pd.read_excel(address)
    dataset_train_values = dataset_train.iloc[:, 0:dataset_train.shape[1] - 1].values
    dataset_length = dataset_train_values.shape[0]
    data_dimension = dataset_train_values.shape[1]
    y_star = dataset_train.iloc[:, dataset_train.shape[1] - 1:dataset_train.shape[1]].values
    y_star = y_star.reshape(len(y_star), 1)

    return dataset_train, dataset_train_values, dataset_length, data_dimension, y_star


def main():
    # get parameters from user
    args = get_argument()

    # reading data from file with excel format
    dataset_train, dataset_train_values, dataset_length, data_dimension, y_star = get_dataset(args.input)

    # initialization of parameters
    algorithm_mode, \
    min_length_chromosome, \
    max_length_chromosome, \
    max_sigma_mutation, \
    initial_number_chromosomes = functions.initialization_parameter(data_set=dataset_train,
                                                                    regression_threshold=args.reg_thr,
                                                                    ratio_min_length_chromosome_reg=args.cminlcr,
                                                                    ratio_max_length_chromosome=args.cmaxlcr,
                                                                    ratio_max_sigma=args.cmaxs,
                                                                    ratio_initial_number_chromosomes=args.init_ch_nu)

    all_result = []
    normal_data = False
    my_threads = []

    if algorithm_mode == "Regression":
        args.threads = 1

    # starting threads
    for i in range(args.threads):
        th_i = es_core.es_thread(dataset_train,
                                 dataset_train_values,
                                 dataset_length,
                                 data_dimension,
                                 y_star,
                                 normal_data,
                                 algorithm_mode,
                                 min_length_chromosome,
                                 max_length_chromosome,
                                 max_sigma_mutation,
                                 initial_number_chromosomes,
                                 args.iterations,
                                 args.q,
                                 all_result,
                                 i + 1)
        th_i.start()
        my_threads.append(th_i)
    for i in range(args.threads):
        my_threads[i].join()

    best_res = all_result[0]
    best_res_fit = all_result[0][1]
    for i in range(1, len(all_result)):
        if all_result[i][1] > best_res_fit:
            best_res_fit = all_result[i][1]
            best_res = all_result[i]

    print("######################### Result Report")
    # best_res[0] = y;  best_res[1] = fitness value;    best_res[2] = chromosome
    # best_res[3] = weight;     best_res[4] = g_matrix

    print("The program was implemented in " + algorithm_mode + " mode")
    if algorithm_mode == "Classification_2" or algorithm_mode == "Classification_n":
        print("best fitness : " + str(best_res_fit))
        print("Learning Error : " + str((1 - best_res_fit) * 100))
    elif algorithm_mode == "Regression":
        plt.plot(dataset_train_values, best_res[0])
        plt.ylabel('Y')
        plt.show()
        print("Learning Error : " + str((1 / best_res_fit) * 100))

    print("best chromosome with highest fitness :")
    print(best_res[2])

    print("best chromosome with highest fitness :")
    print(best_res[3])

    print("best chromosome with highest fitness :")
    print(best_res[4])


if __name__ == '__main__':
    main()
