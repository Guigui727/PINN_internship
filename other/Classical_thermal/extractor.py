import numpy as np
import matplotlib.pyplot as plt

f = open('plots.csv', 'r')
content = f.read()
lines = content.split('\n')
lines = lines[2:-1:2]
values = [x.split(';') for x in lines]
arr = np.array(values, dtype=np.float32)

np.save("abaqus.npy", arr)