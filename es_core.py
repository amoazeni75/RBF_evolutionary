import functions
from sklearn.preprocessing import MinMaxScaler
import threading


class es_thread(threading.Thread):
    def __init__(self,
                 dataset_train,
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
                 iteration_steps,
                 q,
                 all_result,
                 thread_number):
        threading.Thread.__init__(self)
        self.dataset_train = dataset_train
        self.dataset_train_values = dataset_train_values
        self.dataset_length = dataset_length
        self.data_dimension = data_dimension
        self.y_star = y_star
        self.normal_data = normal_data
        self.algorithm_mode = algorithm_mode
        self.min_length_chromosome = min_length_chromosome
        self.max_length_chromosome = max_length_chromosome
        self.max_sigma_mutation = max_sigma_mutation
        self.initial_number_chromosomes = initial_number_chromosomes
        self.iteration_steps = iteration_steps
        self.q = q
        self.all_result = all_result
        self.thread_number = thread_number

    def run(self):
        # Print Information
        functions.print_algorithm_parameters(self.dataset_length,
                                             self.initial_number_chromosomes,
                                             self.min_length_chromosome,
                                             self.max_length_chromosome,
                                             self.max_sigma_mutation,
                                             self.thread_number)
        # normalizing data
        if self.normal_data:
            sc = MinMaxScaler(feature_range=(0, 1))
            self.dataset_train_values = sc.fit_transform(self.dataset_train_values)
            self.dataset_train = sc.fit_transform(self.dataset_train)

        # producing initial generation
        generation = functions.create_initial_generation(data_set=self.dataset_train_values,
                                                         min_length_chromosome=self.min_length_chromosome,
                                                         max_length_chromosome=self.max_length_chromosome,
                                                         generation_size=self.initial_number_chromosomes,
                                                         max_sigma=self.max_sigma_mutation,
                                                         data_set_raw=self.dataset_train)

        for i in range(self.iteration_steps):
            # selecting parents
            parents = functions.selecting_parents_random_uniform(generation)

            # mutation
            functions.do_mutation(parents, self.data_dimension)

            # recombination
            childes = functions.recombination_chromosomes(parents)

            generation = functions.select_and_evaluate(parents + childes,
                                                       self.q,
                                                       self.initial_number_chromosomes,
                                                       self.dataset_train_values,
                                                       self.y_star,
                                                       self.algorithm_mode)

            print("iteration " + str(i + 1) + " completed" + " Thread #" + str(self.thread_number))

        # select best chromosome
        y_result = functions.get_final_result(self.dataset_train_values,
                                              generation,
                                              self.y_star,
                                              self.algorithm_mode)
        self.all_result.append(y_result)
