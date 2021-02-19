from noisygeneration import PaMPS
import numpy as np
import matplotlib.pyplot as plt

### Preparing the samples

#sampler = PaMPS(POVM='Pauli', Number_qubits=2, MPS='tensor.txt', p=0.75)
sampler = PaMPS(POVM='Pauli', Number_qubits=2, MPS='GHZ', p=0.0001)
sampler.samples(Ns=50000, fname='train.txt')

data = np.loadtxt('train.txt')

outcomes = np.sum(data * np.concatenate([np.arange(6), np.arange(6) * 6]), axis=1)
#plt.hist(outcomes, 72)
#plt.xlim([-2, 38])
#plt.show()

###Training the network

from models.RNN import rnn
import tensorflow as tf


model = rnn.LatentAttention(data='train.txt', 
                        K=6, 
                        Number_qubits=2, 
                        #latent_rep_size=int(sys.argv[2]), 
                        #gru_hidden=int(sys.argv[3]), 
                        #decoder='TimeDistributed_mol', 
                        #Nsamples = 100 
                        )

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)


model.train()
#model.train()


gen = model.generation('TimeDistributed_mol')

print(gen[0])
W = gen[0].eval(session=sess)
#print(W)
outcomes_gen = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    result = W[i, :, :]    
    outcomes_gen[i] = np.where(result[0] >0)[0][0] + np.where(result[1]>0)[0][0] * 6
    
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].hist(outcomes, 72)
axs[0].set_xlim([-2, 38])

axs[1].hist(outcomes_gen, 72)
axs[1].set_xlim([-2, 38])

plt.show()
