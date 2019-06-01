import random


def create_initial_generation(inp, generation_size):
    indexes = random.sample(range(inp.shape[0]), int(generation_size))

    return indexes


# we take the minimum length of chromosome equal to number of class in
# classification mode and 0.2 * class_labels_size in regression mode
# also we take maximum length of chromosome = 0.4 * data_set_length
# initial_number_chromosomes = 0.4 * data set size

@TODO min_length_chromosome,max_length_chromosome,\
      initial_number_chromosomes, max_sigma_mutation
        give as parameter

def initialization_parameter(data_set, regression_threshold):
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
        min_length_chromosome = int(0.2 * class_labels_size)

    max_length_chromosome = int(data_set.shape[0] * 0.4)

    max_range = -9223372036854775807
    min_range = 9223372036854775807
    for i in range(1, data_set.shape[1]):
        max_range = max(max_range, data_set[i].max())
        min_range = min(min_range, data_set[i].min())
    max_sigma_mutation = (max_range - min_range) * 0.1
    initial_number_chromosomes = 0.4 * data_set.shape[0]

    return algorithm_mode, \
           min_length_chromosome,\
           max_length_chromosome, \
           max_sigma_mutation,\
           initial_number_chromosomes
