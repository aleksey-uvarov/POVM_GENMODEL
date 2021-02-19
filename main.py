from noisygeneration import PaMPS
import numpy as np
import matplotlib.pyplot as plt

#sampler = PaMPS(POVM='Pauli', Number_qubits=2, MPS='tensor.txt', p=0.75)
sampler = PaMPS(POVM='Pauli', Number_qubits=2, MPS='GHZ', p=0.75)
sampler.samples(Ns=100000, fname='train.txt')

data = np.loadtxt('train.txt')

outcomes = np.sum(data * np.concatenate([np.arange(6), np.arange(6) * 6]), axis=1)
plt.hist(outcomes, 72)
plt.xlim([-2, 38])
plt.show()
