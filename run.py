# reformat Ctrl+Alt+L
import pandas as pd
import argparse
import functions
import es_core
import matplotlib.pyplot as plt
import numpy as np
from time import time
import generate_data as gd


# running mode of Algorithm  = {Classification_2, Classification_n, Regression}


def get_argument():
    parser = argparse.ArgumentParser(
        description='This Programme try to train an RBF Network with the help of ES Algorithm')

    parser.add_argument('--train',
                        help='Path of train data',
                        default="regdata2000.xlsx")
    parser.add_argument('--test',
                        help='Path of test data',
                        default="none")
    parser.add_argument('--cminlcr',
                        help='Minimum Number of Basis',
                        default="5")
    parser.add_argument('--cmaxlcr',
                        help='Maximum Number of Basis',
                        default="15")
    parser.add_argument('--init_ch_nu',
                        help='Initial Size of Generation',
                        default="30")
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
                        default="1")
    parser.add_argument('--iterations',
                        help='number of iterations in ES',
                        default="5")

    args = parser.parse_args()
    args.cminlcr = int(args.cminlcr)
    args.cmaxlcr = int(args.cmaxlcr)
    args.init_ch_nu = int(args.init_ch_nu)
    args.cmaxs = float(args.cmaxs)
    args.reg_thr = float(args.reg_thr)
    args.q = float(args.q)
    args.threads = int(args.threads)
    args.iterations = int(args.iterations)

    return args


def get_dataset(address_train, address_test):
    dataset_train = pd.read_excel(address_train)
    dataset_train_values = dataset_train.iloc[:, 0:dataset_train.shape[1] - 1].values
    dataset_length = dataset_train_values.shape[0]
    data_dimension = dataset_train_values.shape[1]
    y_star = dataset_train.iloc[:, dataset_train.shape[1] - 1:dataset_train.shape[1]].values
    y_star = y_star.reshape(len(y_star), 1)

    dataset_test = []
    dataset_test_values = []
    dataset_test_length = -1
    data_test_dimension = -1
    y_star_test = []
    if address_test != 'none':
        dataset_test = pd.read_excel(address_test)
        dataset_test_values = dataset_test.iloc[:, 0:dataset_test.shape[1] - 1].values
        dataset_test_length = dataset_test_values.shape[0]
        data_test_dimension = dataset_test_values.shape[1]
        y_star_test = dataset_test.iloc[:, dataset_test.shape[1] - 1:dataset_test.shape[1]].values
        y_star_test = y_star_test.reshape(len(y_star_test), 1)

    return dataset_train, \
           dataset_train_values, \
           dataset_length, \
           data_dimension, \
           y_star, \
           dataset_test, \
           dataset_test_values, \
           dataset_test_length, \
           data_test_dimension, \
           y_star_test


def main():
    # get parameters from user
    args = get_argument()

    # reading data from file with excel format
    dataset_train, \
    dataset_train_values, \
    dataset_length, \
    data_dimension, \
    y_star, \
    dataset_test, \
    dataset_test_values, \
    dataset_test_length, \
    data_test_dimension, \
    y_star_test = get_dataset(args.train, args.test)

    # initialization of parameters

    algorithm_mode, \
    min_length_chromosome, \
    max_length_chromosome, \
    max_sigma_mutation, \
    initial_number_chromosomes, \
    number_of_class = functions.initialization_parameter(data_set=dataset_train,
                                                         regression_threshold=args.reg_thr,
                                                         ratio_min_length_chromosome_reg=args.cminlcr,
                                                         ratio_max_length_chromosome=args.cmaxlcr,
                                                         ratio_max_sigma=args.cmaxs,
                                                         ratio_initial_number_chromosomes=args.init_ch_nu)

    all_result = []
    normal_data = False
    my_threads = []

    start_time = time()
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

    end_time = time()
    best_res = all_result[0]
    best_res_fit = all_result[0][1]
    for i in range(1, len(all_result)):
        if all_result[i][1] > best_res_fit:
            best_res_fit = all_result[i][1]
            best_res = all_result[i]

    print("######################### Result Report")
    # best_res[0] = y;  best_res[1] = fitness value;    best_res[2] = chromosome
    # best_res[3] = weight;     best_res[4] = g_matrix
    if algorithm_mode == "Classification_n":
        for i in range(len(best_res[0])):
            best_res[0][i] = np.round(best_res[0][i])

    print("The program was implemented in " + algorithm_mode + " mode")
    if algorithm_mode == "Classification_2" or algorithm_mode == "Classification_n":
        print("best fitness : " + str(best_res_fit))
        print("Learning Error : " + str((1 - best_res_fit) * 100))
        functions.draw_result_classification(best_res[0],
                                             y_star,
                                             best_res_fit * 100,
                                             dataset_length,
                                             number_of_class,
                                             dataset_train_values)
    elif algorithm_mode == "Regression":
        functions.draw_result_regression(best_res[0],
                                         y_star,
                                         100 - ((1 / best_res_fit) * 100)[0][0],
                                         dataset_length)
        print("Learning Error : " + str(((1 / best_res_fit) * 100)[0][0]) + "%")

    print("best chromosome with highest fitness :")
    print(np.array(best_res[2]).reshape(len(best_res[2]), 1))

    print("weights:")
    print(best_res[3])

    print("G Matrix :")
    print(np.array(best_res[4]).reshape(len(best_res[4]), len(best_res[4][0])))

    print("Algorithm Time : " + '%.2f' % (end_time - start_time) + "s")

    print("#####################comapre result in classification")
    if dataset_test_length != -1:
        list_evaluated_test, \
        list_w_matrices, \
        list_y_matrices, \
        list_g_matrices = functions.evaluate_generation(dataset_test_values,
                                                        [best_res[2]],
                                                        y_star_test,
                                                        algorithm_mode)
        print("Accuracy in Test Set is: " + str(list_evaluated_test[0][1] * 100))
        print("class labels: ")
        if algorithm_mode == "Classification_n":
            for i in range(len(list_y_matrices)):
                list_y_matrices[i] = np.round(list_y_matrices[i])
        print(str(list_y_matrices))

        functions.draw_result_classification(list_y_matrices[0],
                                             y_star_test,
                                             list_evaluated_test[0][1] * 100,
                                             dataset_test_length,
                                             number_of_class,
                                             dataset_test_values)


if __name__ == '__main__':
    main()
