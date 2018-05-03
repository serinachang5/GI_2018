import numpy as np
from generator_util import helper_generator

x = {'hey': np.arange(10)}
y = np.arange(10)
t = helper_generator(x, y, 1, True)
for _ in range(20):
    print(next(t))
