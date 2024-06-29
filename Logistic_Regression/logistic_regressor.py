import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs
import random


def data_generator(cluster = 2):
  '''
  This function generates data X and Y of given cluster numbers
  X represents a point in 2D coordinates and Y is label which will be either 1 or 0 in case of 2 clusters
  '''
  X, Y = make_blobs(n_samples=500, centers=2, n_features=2)
  return X,Y

 

def sigmoid(x):
    return (1/(1+np.exp(-x)))



def initialize_parameters():
    '''
    Intializes parameters to a random value
    '''

    # Initializing Coefficients
    w1 = random.random() * 0.001
    w2 = random.random() * 0.001
    b = random.random() * 0.001

    parameters = {
        "w1" : w1,
        "w2" : w2,
        "b" : b,
    }

    print(" \n Randomly initialized initial parameters : \n")
    print(parameters)
    print("\n")
    return parameters


def predict(x,parameters):
    '''
    Outputs the function value based on x
    '''
    w1,w2,b = parameters["w1"],parameters["w2"],parameters["b"] 
    return sigmoid(w1 * x[0] + w2 * x[1] + b)
 

def predict_array(x,parameters):
    '''
    Outputs the function value based on x
    '''
    w1,w2,b = parameters["w1"],parameters["w2"],parameters["b"] 
    return sigmoid(w1 * x[:,0] + w2 * x[:,1] + b)
 
def binary_cross_entropy_loss (Y_hat, Y):
    return -(Y * np.log(Y_hat) + (1 - Y)* np.log(1 - Y_hat))


def gradient_descent(X,Y,parameters):
    '''
    Given data X and Y along with the current parameters,
    calculates and returns the gradient of each parameters as a dictionary
    
    '''
    grad_w1,grad_w2,grad_b = 0.,0.,0.

    for i in range(0,X.shape[0]):
        y_hat = predict(X[i],parameters) 
        error = y_hat - Y[i]
        grad_w1 += error * X[i][0]
        grad_w2 += error * X[i][1]
        grad_b += error 
    
    grad_w1 /= X.shape[0]
    grad_w2 /= X.shape[0]
    grad_b /= X.shape[0]

    grad = {
        "grad_w1" : grad_w1 ,
        "grad_w2" : grad_w2 ,
        "grad_b" : grad_b
    }

    return grad


def update_weights(parameters,grad, learning_rate):
    '''
    Updates the parameters and returns as a dictionary
    '''


    w1,w2,b = parameters["w1"],parameters["w2"],parameters["b"]
    w1 -= learning_rate * grad["grad_w1"]
    w2 -= learning_rate * grad["grad_w2"]
    b -= learning_rate * grad["grad_b"]

    updated_parameters = {
        "w1" : w1,
        "w2" : w2,
        "b" : b
    }
    return updated_parameters


 
def train(X,Y,learning_rate = 0.001 ,epoch = 100, iteration_data = False , param_data = False):
    '''
     Trains a Logistic Regression Model.
    
    '''

    initial_parameters = initialize_parameters()
    dataset_length = X.shape[0]
    epoch_error = []
    param_list = []
    for i in range(0,epoch+1):
        grad = gradient_descent(X,Y,initial_parameters)
        initial_parameters = update_weights(initial_parameters,grad,learning_rate)
        Y_hat = predict_array(X,initial_parameters) 
        cost  =  binary_cross_entropy_loss(Y_hat , Y)
        # Display current epoch and error
        cost_value = np.sum(cost)/dataset_length
        if i%10 == 0 :
            print(f'Epoch {i}, Error: {cost_value}')
        epoch_error.append(cost_value)
        param_list.append(initial_parameters)

    print("\n Training Completed, Final Parameters are : " ,initial_parameters , "\n")
 
    if iteration_data and param_data :
        return initial_parameters, epoch_error, param_list
    elif iteration_data and not param_data :
        return initial_parameters, epoch_error
    elif param_data and not iteration_data:
        return initial_parameters,param_data
    else :
        return initial_parameters    
    

def plot_epochs(iteration_data):
    '''
    Plots epochs vs cost graph
    '''
    plt.figure(figsize= (10,5)) 
    plt.plot(iteration_data)
    plt.xlabel("Epochs")
    plt.ylabel(" Cross Entropy Loss")
    plt.title("Binary Cross Entropy Loss vs Epoch ")
    plt.show()



def plot(X, Y, parameters):
    w1,w2,b = parameters["w1"],parameters["w2"],parameters["b"] 
    # Calculating the intercept and gradient of the decision boundary.
    c = -b/w2
    m = -w1/w2
    xmin, xmax = np.min(X[:,0]),np.max(X[:,0])
    ymin, ymax = np.min(X[:,1]),np.max(X[:,1])
    xd = np.array([xmin, xmax])
    yd = m*xd + c
    plt.figure(figsize=(8, 6))
    plt.title("Final Decision Boundary Plot")
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y, s=50, cmap ="Spectral")
    plt.plot(xd, yd, 'k', lw=1, ls='--') # draw line
    plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.3)
    plt.fill_between(xd, yd, ymax, color='tab:red', alpha=0.3)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.ylabel('X 2')
    plt.xlabel('x 1')
    plt.show()


def animation_plot(X, Y, param_data ):
    '''
    Displays animation of the algorithm learning to fit the data using gradient descent.
    '''
    fig, ax = plt.subplots()
    xmin, xmax = np.min(X[:,0]),np.max(X[:,0])
    ymin, ymax = np.min(X[:,1]),np.max(X[:,1])
    def update(i):
        ax.clear()
        parameters = param_data[i]
        w1,w2,b = parameters["w1"],parameters["w2"],parameters["b"]  
        c = -b/w2
        m = -w1/w2
        xd = np.array([xmin, xmax])
        yd = m*xd + c
        ax.set_title(' Decision Boundary Plot : Epoch ' +  str(i))
        ax.scatter(X[:, 0], X[:, 1], marker="o", c=Y, s=50, cmap ="Spectral")
        ax.plot(xd, yd, 'k', lw=1, ls='--') # draw line
        ax.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.3)
        ax.fill_between(xd, yd, ymax, color='tab:red', alpha=0.3)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel('X 2')
        ax.set_xlabel('x 1')

    ani = FuncAnimation(fig, update, frames=len(param_data), interval=100)
    ani.save(filename="logistic_regression.gif", writer="pillow")
    plt.show()
