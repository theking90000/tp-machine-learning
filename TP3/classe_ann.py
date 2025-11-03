import numpy as np
from scipy.stats import truncnorm

def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class NeuralNetwork:
            
    def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes, learning_rate, bias=None):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.learning_rate = learning_rate 
        self.bias = bias
        self.create_weight_matrices()
    
        
    def create_weight_matrices(self):
        bias_node = 1 if self.bias else 0 
        rad = 1 / np.sqrt(self.no_of_in_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, 
                                        self.no_of_in_nodes + bias_node))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes, 
                                         self.no_of_hidden_nodes + bias_node))

        
    def train(self, input_vector, target_vector):
        # verification de forme
        input_vector = np.array(input_vector)
        input_vector = input_vector.reshape(input_vector.size, 1)    
        target_vector = np.array(target_vector)
        target_vector = target_vector.reshape(target_vector.size, 1)

        # Neurones d'entree
        if self.bias:
            # Ajout d'une coordonnee pour le biais
            input_vector = np.concatenate( (input_vector, [[self.bias]]) )

        # Neurones de la couche cachee
        output_vector_hidden = activation_function(self.weights_in_hidden @ input_vector)
        if self.bias:
            output_vector_hidden = np.concatenate( (output_vector_hidden, [[self.bias]]) )
            
        # Neurones de sortie
        output_vector_network = activation_function(self.weights_hidden_out @ output_vector_hidden)
        
        ###########################################################################################        

        # Equation 1 - Erreur de sortie        
        output_error = 
        delta_last = 
        
        # Equation 2 - Retropropagation de l'erreur
        hidden_errors = 
        delta_hidden = 
        
        # Equation 3 et 4 - Variation du cout par rapport aux biais
        Eq34_out_hidden = 
        Eq34_in_hidden = 
        
        # Mise à jour des poids (/!\ aux neurones de biais)
        # Poids en sortie de la couche cachee
        self.weights_hidden_out = 
        
        # Poids en entree de la couche cachee            
        self.weights_in_hidden =


           
    def run(self, x):
        # verification de forme
        x = np.array(x)
        x = x.reshape(x.size, 1)
        if self.bias:
            # ajout de la coordonnee biais
            x = np.concatenate( (x, [[1]]) )
        h = activation_function(self.weights_in_hidden @ x)
        if self.bias:
            h = np.concatenate( (h, [[1]]) )
        y = activation_function(self.weights_hidden_out @ h)
        return y
            
    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs
