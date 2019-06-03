from numpy import random
import math


def create_initial_generation(data_set, min_length_chromosome, max_length_chromosome, generation_size, max_sigma):
    initial_chromosomes = []
    max_range, min_range = get_max_min_range_dataset(data_set)
    # create list of chromosome length with the size of generation_size
    chromosomes_length = random.randint(min_length_chromosome, max_length_chromosome, generation_size)
    # create list of sigma values with the size of generation_size
    makhraj = (data_set.shape[0] * (data_set.shape[1] - 1)) ** (1 / float(data_set.shape[1] - 1))
    radial = get_farthest_distance(data_set) / makhraj
    sigma_values = random.uniform(0.1 * (max_range - min_range), 0.3 * (max_range - min_range), generation_size)

    for i in range(generation_size):
        current_chromosome = []
        radius_values = random.normal(radial, radial * 0.5, chromosomes_length[i])
        # b_values = random.uniform((max_range - min_range) * 0.2, (max_range - min_range) * 0.8, chromosomes_length[i])
        for j in range(chromosomes_length[i]):
            for k in range(1, data_set.shape[1]):
                current_chromosome.append(random.uniform(data_set[k].min(), data_set[k].max()))
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
        node1 = dataset.iloc[[i], 0: dataset.shape[1] - 1]
        for j in range(i + 1, dataset.shape[0]):
            node2 = dataset.iloc[[j], 0: dataset.shape[1] - 1]
            max_distance = max(max_distance, get_distance(node1, node2))
    return max_distance


def get_distance(p1, p2):
    p1 = p1.values
    p2 = p2.values
    sum_num = 0
    for i in range(p1.shape[1]):
        sum_num += (p1[0][i] - p2[0][i]) ** 2
    sum_num = math.sqrt(sum_num)
    return sum_num


def selecting_parents_random_uniform(seed):
    return random.choice(seed, 7 * len(seed))


def do_mutation(generation, dimension_size):
    # first we must mutate sigma and then each gene
    for i in range(len(generation)):
        # for j in range(dimension_size, len(generation[i]) - 1, 3):
        #     generation[i][j] = generation[i][j] * math.exp(-(1 / math.sqrt(dimension_size) * random.normal(0, 1)))
        #     for k in range(j - dimension_size, j):
        #         generation[i][k] = generation[i][k] + generation[i][j] * random.normal(0, 1)
        generation[i][-1] = generation[i][-1] * math.exp(-(1 / math.sqrt(dimension_size) * random.normal(0, 1)))
        for j in range(len(generation[i]) - 1):
            generation[i][j] = generation[i][j] + generation[i][-1] * random.normal(0, 1)


def recombination_chromosomes(generation):
    childes = []
    for i in range(len(generation)):
        chromosome1 = generation[i]
        for j in range(i + 1, len(generation)):
            chromosome2 = generation[j]
            # if random.uniform(0, 1) <= 0.4:
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


def evaluate_generation(generation, running_mode):
    return []
