import tensorflow as tf 
import numpy as np
from rbm import RBM
import matplotlib.pyplot as plt
import sys

import warnings
warnings.filterwarnings("ignore")

### I suspect that the sampling procedure is somehow flawed -- Alexey

clean = int(sys.argv[1])
#print(clean + "====================")

if clean==1:
    train_data = 'train_clean.txt'
    samples_data = 'KLs_clean.txt'
    p = 0
else:
    train_data = 'train_noisy.txt'
    samples_data = 'KLs_noisy.txt'
    p = 0.2
    
print(samples_data)

def sample_RBM(model, sess):

    hidden, visible = model.stochastic_maximum_likelihood(num_iterations=5, cdimages=None)
    #print(visible.shape)
    W = visible.eval(session=sess)
    for i in range(9):
        W2 = visible.eval(session=sess)
        W = np.concatenate([W, W2])
    return np.sum(W * np.concatenate([np.arange(4), np.arange(4) * 4]), axis=1)



#KLs = []
#classical_fidelities = []

data = np.loadtxt(train_data)
outcomes_data = np.sum(data * np.concatenate([np.arange(4), np.arange(4) * 4]), axis=1)
results_data, counts_data = np.unique(outcomes_data, return_counts=True)
probabilities_data = counts_data / np.sum(counts_data)

sess = tf.Session()

outcomes = [list(tf.one_hot(i, 4).eval(session=sess)) 
            + list(tf.one_hot(j, 4).eval(session=sess))
            for i in range(4) for j in range(4)]

#with open(samples_data, 'w') as f:
f = open(samples_data, 'w')
for i in range(100):

    #b = np.load('RBM_parameters/parameters_nH8_L2_p0.1_epoch{0:}.npz'.format(i+1))
    b = np.load('RBM_parameters/parameters_nH16_L2_p{1:}_epoch{0:}.npz'.format(i+1, p))
    model = RBM(num_hidden=16, num_visible=2, num_state_vis=4, num_state_hid=2, num_samples=8) 
    model.weights = b['weights']
    model.visible_bias = b['visible_bias']
    model.hidden_bias = b['hidden_bias']

    init_op = tf.initialize_all_variables()
    sess.run(init_op)


    
    
    #outcomes_sample = sample_RBM(model, sess)
    #results_samp, counts_samp = np.unique(outcomes_sample, return_counts=True)

    free_energies = model.free_energy(outcomes).eval(session=sess)
    probabilities_model = np.exp(free_energies) #sic
    probabilities_model = probabilities_model / np.sum(probabilities_model)

    KL = -np.sum(probabilities_data * np.log((probabilities_model.T / probabilities_data)))
    #fidelity = np.sum(probabilities_data * np.sqrt((probabilities_model.T / probabilities_data)))
    #classical_fidelities.append(fidelity)

    #if not (results_data == results_samp).all():
        #KL = np.infty
    #else:
        #counts_samp = counts_samp / np.sum(counts_samp)
        #counts_data = counts_data / np.sum(counts_data)
        #KL = np.sum(counts_data * np.log(counts_data  / counts_samp))
    #print(KL)
    #KLs.append(KL)
    
    #if i==200:
        #plt.plot(probabilities_data, 'r')
        #plt.plot(probabilities_model, 'b')
        #plt.show()
    if (i % 10) == 0:
        print('========={0:}========'.format(i))
        print(KL)
        
    f.write(str(KL) + "\n")
f.close()
#plt.plot(KLs)
#plt.show()
#np.savetxt('KLs.txt', KLs)

#W = T.eval(session=sess)


#u = model.sample_v_given(model.hidden_samples)
#print(u)
#W = u.eval(session=sess)

#model.num_samples = 1024

#hidden, visible = model.stochastic_maximum_likelihood(num_iterations=1000, cdimages=None)

#print(visible.shape)

#W = visible.eval(session=sess)

#for i in range(5):
    #W2 = visible.eval(session=sess)
    #W = np.concatenate([W, W2])
    #print(W.shape)

#outcomes_sample = np.sum(W * np.concatenate([np.arange(4), np.arange(4) * 4]), axis=1)
#plt.hist(outcomes_sample, 50)
#plt.xlim([-2, 18])
#plt.show() 

#plt.plot(classical_fidelities)
#plt.show()



