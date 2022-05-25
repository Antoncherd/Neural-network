import numpy as np


# function sigmoid
def sigmoid(x):
    return (2 / (1 + np.exp(-x)))-1
# derivative of sigmoid
def df(x):
    return (0.5*(1+x)*(1-x))


# object neuron declaration
class Neuron:

    def __init__(self, n_in_link):
        self.in_data = []  # inputs
        for i in range(n_in_link):
            self.in_data.append(0)

        self.grad = 0  # gradient parameter

        self.weights = []  # input weights, 0 for additional
        for i in range(n_in_link):
            self.weights.append(0)
        self.weight_extra = 0

        # output
        sum = 0
        weight_extra = self.weight_extra
        for i in range(n_in_link):
            sum += self.in_data[i] * self.weights[i]
        self.out_data = sigmoid(sum) + weight_extra

        # normalized gradient
        self.normalized_grad = []
        for i in range(n_in_link):
            grad = self.grad
            weights_i = self.weights[i]
            self.normalized_grad.append(weights_i * grad)

    # neuron computation
    def compute(self, in_array):
        self.in_data = in_array
        summ = 0
        weight_extra = self.weight_extra
        for i in range(len(in_array)):
            summ += in_array[i] * self.weights[i]
        self.out_data = sigmoid(summ) + weight_extra

    def grad_normalize(self, grad):
        normalized_grad = self.normalized_grad
        for i in range(len(self.weights)):
            weights_i = self.weights[i]
            self.normalized_grad[i] = weights_i * grad

    def in_data(self):
        return self.in_data

    def out_data(self):
        return self.out_data

    def in_learning(self):
        return self.in_learning

    def delta_learning(self):
        return self.delta_learning

# forward neuronet computation function 
def compute_full(in_array, matrix_neurons):
    in_array_variable = in_array
    in_array_variable_rewrite = []
    # all hidden layers
    for i in range(len(matrix_neurons)-1):

        for j in range(len(matrix_neurons[i])):
            matrix_neurons[i][j].compute(in_array_variable)
            in_array_variable_rewrite.append(matrix_neurons[i][j].out_data)
        in_array_variable = in_array_variable_rewrite
        in_array_variable_rewrite = []
    # last neuron


    matrix_neurons[len(matrix_neurons)-1][0].compute(in_array_variable)
    return matrix_neurons[len(matrix_neurons)-1][0].out_data

def back_propagation(matrix_neurons, ideal_out, par_L):
    # for last neuron
    out = matrix_neurons[ len(matrix_neurons) - 1 ][ len(matrix_neurons[len(matrix_neurons)-1]) - 1].out_data
    grad = (out - ideal_out) * df(out) # computed gradient for last neuron
    matrix_neurons[len(matrix_neurons)-1][len(matrix_neurons[len(matrix_neurons)-1])-1].grad = grad # write gradient
    matrix_neurons[len(matrix_neurons)-1][len(matrix_neurons[len(matrix_neurons)-1])-1].grad_normalize(grad) # normalized gradient
    in_array = matrix_neurons[len(matrix_neurons)-1][len(matrix_neurons[len(matrix_neurons)-1])-1].in_data # write inputs for neuron
    for k in range(len(in_array)):
        weight = matrix_neurons[len(matrix_neurons)-1][len(matrix_neurons[len(matrix_neurons)-1])-1].weights[k]
        grad = matrix_neurons[len(matrix_neurons)-1][len(matrix_neurons[len(matrix_neurons)-1])-1].grad
        matrix_neurons[len(matrix_neurons)-1][len(matrix_neurons[len(matrix_neurons)-1])-1].weights[k] = weight - par_L * grad * in_array[k] # change weight[k]
        weight_extra = matrix_neurons[len(matrix_neurons)-1][len(matrix_neurons[len(matrix_neurons)-1])-1].weight_extra
        matrix_neurons[len(matrix_neurons)-1][len(matrix_neurons[len(matrix_neurons)-1])-1].weight_extra = weight_extra - par_L * grad # change weight_extra


    for i in range(2, len(matrix_neurons)): # here we go back through layers
        for j in range(0, len(matrix_neurons[len(matrix_neurons) - i])): # here we go through current layer
            out = matrix_neurons[len(matrix_neurons) - i][j].out_data
            grad = 0
            for k in range(len(matrix_neurons[len(matrix_neurons) - i + 1])): # here we sum gradients from next layer
                grad_next = matrix_neurons[len(matrix_neurons) - i + 1][k].normalized_grad[j]
                grad += grad_next
            grad = grad * df(out) # computed gradient for j neuron from N-i layer
            matrix_neurons[len(matrix_neurons) - i][j].grad = grad  # write gradient
            matrix_neurons[len(matrix_neurons) - i][j].grad_normalize(grad)  # normalized gradient
            in_array = matrix_neurons[len(matrix_neurons) - i][j].in_data  # write inputs for neuron
            for k in range(len(in_array)): # here we change weights for j neuron from N-i layer
                weight = matrix_neurons[len(matrix_neurons) - i][j].weights[k]
                grad = matrix_neurons[len(matrix_neurons) - i][j].grad
                matrix_neurons[len(matrix_neurons) - i][j].weights[k] = weight - par_L * grad * in_array[k]  # change weight[k]
                weight_extra = matrix_neurons[len(matrix_neurons) - i][j].weight_extra
                matrix_neurons[len(matrix_neurons) - i][j].weight_extra = weight_extra - par_L * grad  # change weight_extra


training_inputs = ((-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1))
training_outputs = (-0.1, 0.9, -0.9, 0.5, -0.9, 0.9, -0.5, -0.9)
#training_inputs = ((-1, -1), (-1, 1), (1, -1), (1, 1))
#training_outputs = (-0.9, 0.9, 0.9, -0.9)
par_L = 0.01  # параметр обучения

def print_matrix_neuron (matrix_neuron):
    for i in range(len(matrix_neuron)):
        array_out_i = []
        array_extra_i = []
        for j in range(len(matrix_neuron[i])):
            array_out_i.append(matrix_neuron[i][j].out_data)
            array_extra_i.append(matrix_neuron[i][j].weight_extra)
        print("out_data from layer ", i, ":", array_out_i, "extra_weight = ", array_extra_i)


# initial neuronet state 
Weight1 = np.array([ [-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])
Weight2 = np.array([0.2, 0.3])
#Weight1 = np.array([[1.25, 1.32, -2.46],[-0.009, -0.392, 0.36]])
#Weight2 = np.array([-3.22, 0.231])
matrix = []
for i in range(3): # amount layers
    matrix.append([])

for i in range(5): # amount neurons in 0 layer
    matrix[0].append(Neuron(len(training_inputs[0])))
    matrix[0][i].weights = np.random.sample(len(training_inputs[0]))
for i in range(3): # amount neurons in 0 layer
    matrix[1].append(Neuron(len(matrix[0])))
    matrix[1][i].weights = np.random.rand(len(matrix[0]))

matrix[2].append(Neuron(len(matrix[1])))   # out neuron
matrix[2][0].weights = np.random.rand(len(matrix[1]))
#matrix[2].append(neuron(3))
#matrix[2][0].weights = np.random.sample(3)

real_out = compute_full(training_inputs[2], matrix)
print(training_inputs[2], real_out)
print_matrix_neuron(matrix)

for i in range(5000):
    # 
    # проход по всем тренировочным входам
    random_input_number = np.random.randint(0, len(training_inputs)-1)
    # error calculation
    real_out = compute_full(training_inputs[random_input_number], matrix)
    #print(i, k1, training_inputs[k1], err, neuron_out.weights[0], neuron_out.delta_learning)
    # поправки к выходному нейрону
    back_propagation(matrix, training_outputs[random_input_number], par_L)


# calculation
print("Нейросеть настроена")
print_matrix_neuron(matrix)
for i in range(len(training_inputs)):
    a = compute_full(training_inputs[i], matrix) #choose better name
    print(training_inputs[i], a, training_outputs[i])


