import tensorflow as tf 
import numpy as np
from rbm import RBM
import matplotlib.pyplot as plt


def sample_RBM(model, sess):

    hidden, visible = model.stochastic_maximum_likelihood(num_iterations=1000, cdimages=None)
    #print(visible.shape)
    W = visible.eval(session=sess)
    for i in range(9):
        W2 = visible.eval(session=sess)
        W = np.concatenate([W, W2])
    return np.sum(W * np.concatenate([np.arange(4), np.arange(4) * 4]), axis=1)



KLs = []

data = np.loadtxt('train.txt')
outcomes_data = np.sum(data * np.concatenate([np.arange(4), np.arange(4) * 4]), axis=1)

results_data, counts_data = np.unique(outcomes_data, return_counts=True)


sess = tf.Session()



for i in range(10, 20, 2): 
    print('========={0:}========'.format(i))
    #b = np.load('RBM_parameters/parameters_nH8_L2_p0.1_epoch{0:}.npz'.format(i+1))
    b = np.load('RBM_parameters/parameters_nH8_L2_p0_epoch{0:}.npz'.format(i+1))
    model = RBM(num_hidden=8, num_visible=2, num_state_vis=4, num_state_hid=2) 
    model.weights = b['weights']
    model.visible_bias = b['visible_bias']
    model.hidden_bias = b['hidden_bias']

    init_op = tf.initialize_all_variables()

    sess.run(init_op)

    
    outcomes_sample = sample_RBM(model, sess)
    results_samp, counts_samp = np.unique(outcomes_sample, return_counts=True)


    if not (results_data == results_samp).all():
        KL = np.infty
    else:
        counts_samp = counts_samp / np.sum(counts_samp)
        counts_data = counts_data / np.sum(counts_data)
        KL = np.sum(counts_data * np.log(counts_data  / counts_samp))
    print(KL)
    KLs.append(KL)
    
    with open('KLs.txt', 'a') as f:
        f.write(str(KL) + "\n")
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




