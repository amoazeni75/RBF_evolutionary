# reformat Ctrl+Alt+L
import pandas as pd
import argparse
import functions
from sklearn.preprocessing import MinMaxScaler


# running mode of Algorithm  = {Classification_2, Classification_n, Regression}


def get_argument():
    parser = argparse.ArgumentParser(
        description='This Programme try to train an RBF Network with the help of ES Algorithm')

    parser.add_argument('--input', help='Path of input data', default="simple_set1.xlsx")
    parser.add_argument('--cminlcr', help='Minimum Ratio of Chromosome Length in Regression', default="0.2")
    parser.add_argument('--cmaxlcr', help='Maximum Ratio of Chromosome Length', default="0.2")
    parser.add_argument('--init_ch_nu', help='Initial Ratio of Generation', default="0.4")
    parser.add_argument('--cmaxs', help='Maximum Ratio of Sigma', default="0.1")
    parser.add_argument('--reg_thr', help='threshold that indicate running mode', default="0.4")
    parser.add_argument('--q', help='threshold that indicate running mode', default="3")

    args = parser.parse_args()
    args.cminlcr = float(args.cminlcr)
    args.cmaxlcr = float(args.cmaxlcr)
    args.init_ch_nu = float(args.init_ch_nu)
    args.cmaxs = float(args.cmaxs)
    args.reg_thr = float(args.reg_thr)
    args.q = float(args.q)

    return args


def get_dataset(address):
    dataset_train = pd.read_excel(address)
    dataset_train_values = dataset_train.iloc[:, 0:dataset_train.shape[1] - 1].values
    dataset_length = dataset_train_values.shape[0]
    data_dimension = dataset_train_values.shape[1]
    y_star = dataset_train.iloc[:, dataset_train.shape[1] - 1:dataset_train.shape[1]].values
    y_star = y_star.reshape(len(y_star), 1)

    return dataset_train, dataset_train_values, dataset_length, data_dimension, y_star


def evolutionary_strategy_core(dataset_train, dataset_train_values, dataset_length, data_dimension,
                               y_star, normal_data, algorithm_mode, min_length_chromosome,
                               max_length_chromosome, max_sigma_mutation, initial_number_chromosomes,
                               iteration_steps, q, all_result):
    # Print Information
    functions.print_algorithm_parameters(dataset_length,
                                         initial_number_chromosomes,
                                         min_length_chromosome,
                                         max_length_chromosome,
                                         max_sigma_mutation)
    # normalizing data
    if normal_data:
        sc = MinMaxScaler(feature_range=(0, 1))
        dataset_train_values = sc.fit_transform(dataset_train_values)
        dataset_train = sc.fit_transform(dataset_train)

    # producing initial generation
    generation = functions.create_initial_generation(data_set=dataset_train_values,
                                                     min_length_chromosome=min_length_chromosome,
                                                     max_length_chromosome=max_length_chromosome,
                                                     generation_size=initial_number_chromosomes,
                                                     max_sigma=max_sigma_mutation,
                                                     data_set_raw=dataset_train)

    for i in range(iteration_steps):
        # selecting parents
        parents = functions.selecting_parents_random_uniform(generation)

        print("parent selected in " + str(i))
        # mutation
        functions.do_mutation(parents, data_dimension)

        # recombination
        childes = functions.recombination_chromosomes(parents)

        # evaluate parents and childes
        evaluated = functions.evaluate_generation(dataset_train_values, parents + childes, y_star, algorithm_mode)

        # selection for next iteration
        del generation
        del childes
        del parents
        generation = functions.select_based_on_q_tournament(evaluated, q, initial_number_chromosomes)
        print("iteration " + str(i) + " completed")

    # select best chromosome
    y_result = functions.get_final_result(dataset_train_values, generation, y_star, algorithm_mode)
    all_result.append(y_result)


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
    iteration_steps = 20

    for i in range(5):
        evolutionary_strategy_core(dataset_train, dataset_train_values, dataset_length, data_dimension,
                                   y_star, normal_data, algorithm_mode, min_length_chromosome,
                                   max_length_chromosome, max_sigma_mutation, initial_number_chromosomes,
                                   iteration_steps, args.q, all_result)

    best_res = all_result[0][0]
    best_res_fit = all_result[0][1]
    for i in range(1, len(all_result)):
        if all_result[i][1] > best_res_fit:
            best_res_fit = all_result[i][1]
            best_res = all_result[i][0]
    print(best_res)
    print("fitness : " + str(best_res_fit))


if __name__ == '__main__':
    main()
