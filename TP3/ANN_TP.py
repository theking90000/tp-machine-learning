import numpy as np
from scipy.stats import truncnorm
from classe_ann import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

n_sample=5000
data, labels = make_blobs(n_samples=n_sample, n_features=3, random_state=0, centers=2, cluster_std=1.2)
size_of_learn_sample = int(n_sample * 0.8)
size_of_test_sample = n_sample-size_of_learn_sample
X_learn_data = data[:size_of_learn_sample]
X_test_data = data[-size_of_test_sample:]
y_learn_data = labels[:size_of_learn_sample]
y_test_data = labels[-size_of_test_sample:]

simple_network = NeuralNetwork(no_of_in_nodes=3, 
                               no_of_out_nodes=2, 
                               no_of_hidden_nodes=4,
                               learning_rate=0.1,
                               bias=1)
    
labels = (np.arange(2) == y_learn_data.reshape(y_learn_data.size, 1))
labels = labels.astype(np.float)

n = len(X_learn_data[:,1])
epoch = 10
for e in range(epoch):
    perm = np.arange(n)
    perm = np.random.permutation(perm)
    for i in perm:
        simple_network.train(X_learn_data[i], labels[i])
    correct, wrong = simple_network.evaluate(X_learn_data, y_learn_data)
    print(wrong, correct)


    

correct, wrong = simple_network.evaluate(X_test_data, y_test_data)
print(wrong, correct)
