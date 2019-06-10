import numpy as np


# NUM_SAMPLES = 100
# X = np.random.uniform(0., 1., NUM_SAMPLES)
# X = np.sort(X, axis=0)
# noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
# y = np.sin(4 * np.pi * X) + noise
#
# plt.plot(X, y)
# plt.ylabel('some numbers')
# plt.show()
# #sleep(10)  # Time in seconds
#
# df = pd.DataFrame({'1': X, '2': y})
# df.to_excel('NamesAndAges.xlsx')

def create_samples_classification(center,
                                  radius,
                                  number_of_sample,
                                  dimension_size,
                                  label_number,
                                  result_list):
    samples = np.random.normal(center,
                               radius,
                               size=[number_of_sample, dimension_size])

    for i in range(len(samples)):
        result_list.append(np.concatenate((samples[i], np.array([label_number]))))


def get_sample_2_class():
    class_2_train = []
    create_samples_classification(10, 10, 150, 2, -1, class_2_train)
    create_samples_classification(40, 10, 150, 2, 1, class_2_train)

    class_2_test = []
    create_samples_classification(10, 10, 50, 2, -1, class_2_test)
    create_samples_classification(40, 10, 50, 2, 1, class_2_test)

    return class_2_train, class_2_test


def main():
    c1, c2 = get_sample_2_class()
    for i in range(len(c1)):
        print(*c1[i], sep=", ")
    print("####")
    for i in range(len(c2)):
        print(*c2[i], sep=", ")


if __name__ == "__main__":
    main()
