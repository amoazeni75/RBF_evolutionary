import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import pandas as pd


NUM_SAMPLES = 100
X = np.random.uniform(0., 1., NUM_SAMPLES)
X = np.sort(X, axis=0)
noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
y = np.sin(4 * np.pi * X) + noise

plt.plot(X, y)
plt.ylabel('some numbers')
plt.show()
#sleep(10)  # Time in seconds

df = pd.DataFrame({'1': X, '2': y})
df.to_excel('NamesAndAges.xlsx')
