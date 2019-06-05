import math
import numpy as np
import random as nd


def create_initial_generation(data_set, min_length_chromosome, max_length_chromosome, generation_size,
                              max_sigma, data_set_raw):
    initial_chromosomes = []

    # create list of chromosome length with the size of generation_size
    chromosomes_length = np.random.randint(min_length_chromosome, max_length_chromosome, generation_size)
    # create list of sigma values with the size of generation_size
    makhraj = (data_set.shape[0] * data_set.shape[1]) ** (1 / float(data_set.shape[1]))
    radial = get_farthest_distance(data_set) / makhraj
    sigma_values = np.random.uniform(0.1 * max_sigma, max_sigma, generation_size)

    for i in range(generation_size):
        current_chromosome = []
        radius_values = np.abs(np.random.normal(radial, radial * 0.75, chromosomes_length[i]))
        for j in range(chromosomes_length[i]):
            for k in range(1, data_set_raw.shape[1]):
                mmin = data_set_raw[k].min()
                mmax = data_set_raw[k].max()
                current_chromosome.append(nd.uniform(mmin, mmax))
            current_chromosome.append(radius_values[j])

        current_chromosome.append(sigma_values[i])
        initial_chromosomes.append(current_chromosome)

    return initial_chromosomes


# we take the minimum length of chromosome equal to number of class in
# classification mode and 0.2 * class_labels_size in regression mode
# also we take maximum length of chromosome = 0.4 * data_set_length
# initial_number_chromosomes = 0.4 * data set size
def initialization_parameter(data_set,
                             regression_threshold,
                             ratio_min_length_chromosome_reg,
                             ratio_max_length_chromosome,
                             ratio_max_sigma,
                             ratio_initial_number_chromosomes):
    # first finding algorithm mode
    class_labels_size = len(data_set[data_set.shape[1]].unique())
    if class_labels_size <= 2:
        algorithm_mode = 'Classification_2'
        min_length_chromosome = class_labels_size
    elif class_labels_size < int(regression_threshold * data_set.shape[0]):
        algorithm_mode = 'Classification_n'
        min_length_chromosome = class_labels_size
    else:
        algorithm_mode = 'Regression'
        min_length_chromosome = int(ratio_min_length_chromosome_reg * class_labels_size)

    max_length_chromosome = int(data_set.shape[0] * ratio_max_length_chromosome)

    max_range, min_range = get_max_min_range_dataset(data_set)
    max_sigma_mutation = (max_range - min_range) * ratio_max_sigma
    initial_number_chromosomes = int(ratio_initial_number_chromosomes * data_set.shape[0])

    return algorithm_mode, \
           min_length_chromosome, \
           max_length_chromosome, \
           max_sigma_mutation, \
           initial_number_chromosomes


def get_max_min_range_dataset(data_set):
    max_range = -9223372036854775807
    min_range = 9223372036854775807
    for i in range(1, data_set.shape[1]):
        max_range = max(max_range, data_set[i].max())
        min_range = min(min_range, data_set[i].min())
    return max_range, min_range


def get_farthest_distance(dataset):
    max_distance = 0
    for i in range(dataset.shape[0]):
        node1 = dataset[i]
        for j in range(i + 1, dataset.shape[0]):
            node2 = dataset[j]
            max_distance = max(max_distance, get_distance(node1, node2))
    return max_distance


def get_distance(p1, p2):
    sum_num = 0
    for i in range(len(p1)):
        sum_num += (p1[i] - p2[i]) ** 2
    sum_num = math.sqrt(sum_num)
    return sum_num


def selecting_parents_random_uniform(seed):
    parents = []
    indexes = np.random.randint(0, len(seed), 7 * len(seed))
    for i in range(len(indexes)):
        parents.append(seed[indexes[i]])
    return parents


def do_mutation(generation, dimension_size):
    # first we must mutate sigma and then each gene
    for i in range(len(generation)):
        # for j in range(dimension_size, len(generation[i]) - 1, 3):
        #     generation[i][j] = generation[i][j] * math.exp(-(1 / math.sqrt(dimension_size) * np.random.normal(0, 1)))
        #     for k in range(j - dimension_size, j):
        #         generation[i][k] = generation[i][k] + generation[i][j] * np.random.normal(0, 1)
        generation[i][-1] = generation[i][-1] * math.exp(-((1 / math.sqrt(dimension_size)) * nd.normalvariate(0, 1)))
        for j in range(len(generation[i]) - 1):
            generation[i][j] = generation[i][j] + generation[i][-1] * nd.normalvariate(0, 1)


def recombination_chromosomes(generation):
    childes = []
    for i in range(len(generation)):
        chromosome1 = generation[i]
        for j in range(i + 1, len(generation)):
            chromosome2 = generation[j]
            if nd.uniform(0, 1) <= 0.4:
                childes.append(do_recombination(chromosome1, chromosome2))

    return childes


def do_recombination(ch1, ch2):
    if len(ch1) <= len(ch2):
        lower_length = ch1
        higher_length = ch2
    else:
        lower_length = ch2
        higher_length = ch1

    new_child = []
    for i in range(len(lower_length)):
        new_child.append((higher_length[i] + lower_length[i]) / 2.0)

    for i in range(len(lower_length), len(higher_length)):
        new_child.append(higher_length[i])

    return new_child


def evaluate_generation(dataset, generation, y_star, running_mode):
    dimension = dataset.shape[1]
    # first calculate G matrix
    list_g_matrices = [[] for i in range(len(generation))]
    for i in range(dataset.shape[0]):
        data_i = dataset[i]
        for j in range(len(generation)):
            list_g_matrices[j].append(get_gi_of_a_data(data_i, generation[j], dimension))

    # second calculate weights
    list_w_matrices = []
    for i in range(len(generation)):
        list_w_matrices.append(calculate_weight_chromosome_i(list_g_matrices[i], y_star))

    # third calculate Y matrix
    list_y_matrices = []
    for i in range(len(generation)):
        list_y_matrices.append(calculate_y_out(list_g_matrices[i], list_w_matrices[i]))

    # check what is running mode and then calculate corresponding error
    list_evaluated = []

    if running_mode == "Regression":
        for i in range(len(generation)):
            fitness = fitness_regression(list_y_matrices[i], y_star)
            list_evaluated.append([generation[i], fitness])
    elif running_mode == "Classification_2":
        for i in range(len(generation)):
            fitness = fitness_classification_2(list_y_matrices[i], y_star)
            # m = (len(generation[i]) - 1) / (dimension + 1)
            # fitness /= m
            list_evaluated.append([generation[i], fitness[0]])
    else:
        for i in range(len(generation)):
            fitness = fitness_classification_n(list_y_matrices[i], y_star)
            list_evaluated.append([generation[i], fitness])

    return list_evaluated


def get_gi_of_a_data(data, chromosome, dimension_size):
    gi = []
    for index_radius in range(dimension_size, len(chromosome) - 1, (dimension_size + 1)):
        radius = chromosome[index_radius] ** 2
        sum_data_dis_center = 0
        index_data = 0
        for index_center in range(index_radius - dimension_size, index_radius):
            sum_data_dis_center += (data[index_data] - chromosome[index_center]) ** 2
            index_data += 1

        sum_data_dis_center = -sum_data_dis_center / radius
        sum_data_dis_center = np.exp(sum_data_dis_center)
        gi.append(sum_data_dis_center)

    return gi


def calculate_weight_chromosome_i(gi, y_star):
    g_new = np.array(gi)
    g_new_transpose = g_new.transpose()
    temp = np.dot(g_new_transpose, g_new)
    temp = temp + 0.001 * np.identity(g_new_transpose.shape[0], dtype=float)
    temp = np.linalg.inv(temp)
    temp = np.dot(temp, g_new_transpose)
    temp = np.dot(temp, y_star)
    return temp


def calculate_y_out(gi, wi):
    return np.dot(gi, wi)


def fitness_regression(y_out, y_star):
    fitness = 0
    fitness = np.transpose(y_out - y_star)
    fitness = np.dot(fitness, y_out - y_star)
    fitness /= 2
    fitness = 1 / fitness
    return fitness


def fitness_classification_2(y_out, y_star):
    fitness = 0
    for i in range(len(y_out)):
        # fitness += np.abs(np.sign(y_out[i]) - y_star[i])
        fitness += np.abs((np.sign(y_out[i]) + 1) / 2 - y_star[i])
    fitness = fitness / (len(y_star))
    fitness = 1 - fitness
    return fitness


def fitness_classification_n(y_out, y_star):
    fitness = 0
    # ??
    return fitness


def select_based_on_q_tournament(generation, q, selection_size):
    selected_chromosome = []
    q = int(q)

    no_need_return_back_selected = False
    if len(generation) > selection_size:
        no_need_return_back_selected = True

    for i in range(selection_size):
        q_selected = nd.sample(range(0, len(generation)), q)
        fit_best = generation[q_selected[0]][1]
        index_b = 0
        best = generation[q_selected[0]]
        for j in range(1, q):
            if (generation[q_selected[j]][1] == best[1] and len(generation[q_selected[j]][0]) < len(best[0])) \
                    or (generation[q_selected[j]][1] > best[1]):
                fit_best = generation[q_selected[j]][1]
                index_b = j
                best = generation[q_selected[j]]

        selected_chromosome.append(best[0])

        if no_need_return_back_selected:
            generation.pop(q_selected[index_b])

    return selected_chromosome


def get_final_result(dataset, chromosome, y_star, running_mode):
    eval = evaluate_generation(dataset, chromosome, y_star, running_mode)
    best_ch = eval[0][0]
    best_fit = eval[0][1]
    for i in range(len(eval)):
        if eval[i][1] > best_fit:
            best_fit = eval[i][1]
            best_ch = eval[i][0]

    chromosome = best_ch
    # first calculate G matrix
    dimension = dataset.shape[1]
    g_matrix = []
    for i in range(dataset.shape[0]):
        data_i = dataset[i]
        g_matrix.append(get_gi_of_a_data(data_i, chromosome, dimension))

    # second calculate weights
    w_matrix = calculate_weight_chromosome_i(g_matrix, y_star)

    # third calculate Y matrix
    y_matrix = calculate_y_out(g_matrix, w_matrix)

    return [y_matrix, best_fit]


def print_algorithm_parameters(dataset_length,
                               initial_number_chromosomes,
                               min_length_chromosome,
                               max_length_chromosome,
                               max_sigma_mutation):
    print("Dataset size : " + str(dataset_length))
    print("Initial generation size : " + str(initial_number_chromosomes))
    print("Min Length of Chromosome : " + str(min_length_chromosome))
    print("Max Length of Chromosome : " + str(max_length_chromosome))
    print("Max Sigma in Mutation : " + str(max_sigma_mutation))
    print("##############################")
