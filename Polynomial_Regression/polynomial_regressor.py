import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
def data_generator(param_list):
  '''
  This function generates data X and Y using function A.x^2 + B.X + C . Based on the parameters provided 
  '''
  a,b,c  = param_list
  x = np.linspace(-5, 5, 20) 
  y = a*np.power(x,2)+b*x+ c
  return x,y

def initialize_parameters():
    '''
    Intializes parameters to a random value
    '''

    # Initializing Coefficients
    a=  random.random()
    b = random.random()
    c = random.random()
    parameters = {
        "a" : a,
        "b" : b,
        "c" : c
    }

    print(" \n Randomly initialized initial parameters : \n")
    print(parameters)
    print("\n")
    return parameters


def predict(x,parameters):
    '''
    Outputs the function value based on x
    '''
    a,b,c = parameters["a"],parameters["b"],parameters["c"]
    return (a*(x**2)+b*x+ c)

def gradient_descent(X,Y,parameters):
    '''
    Given data X and Y along with the current parameters,
    calculates and returns the gradient of each parameters as a dictionary
    
    '''
    grad_a,grad_b,grad_c = 0,0,0
    for i in range(0,X.shape[0]):
        error = predict(X[i],parameters) - Y[i] 
        grad_a += error * X[i]**2
        grad_b += error * X[i]
        grad_c += error 
    
    grad_a /= X.shape[0]
    grad_b /= X.shape[0]
    grad_c /= X.shape[0]

    grad = {
        "grad_a" : grad_a ,
        "grad_b" : grad_b ,
        "grad_c" : grad_c
    }

    return grad

def update_weights(parameters,grad, learning_rate):
    '''
    Updates the parameters and returns as a dictionary
    '''


    a,b,c = parameters["a"],parameters["b"],parameters["c"]
    a -= learning_rate * grad["grad_a"]
    b -= learning_rate * grad["grad_b"]
    c -= learning_rate * grad["grad_c"]

    updated_parameters = {
        "a" : a,
        "b" : b,
        "c" : c
    }
    return updated_parameters
    

def train(X,Y,learning_rate = 0.001 ,epoch = 100, iteration_data = False , param_data = False):
    '''
     Trains a polynomial regression of second order and returns learnt parameter.
    
    '''

    initial_parameters = initialize_parameters()
    dataset_length = X.shape[0]
    epoch_error = []
    param_list = []
    for i in range(0,epoch+1):
        grad = gradient_descent(X,Y,initial_parameters)
        initial_parameters = update_weights(initial_parameters,grad,learning_rate)
        Y_hat =  (initial_parameters["a"] * X**2 + initial_parameters["b"] * X + initial_parameters["c"])
        total_error =  Y_hat - Y
        # Display current epoch and error
        cost_value = np.sum(total_error**2)/dataset_length
        if i%10 == 0 :
            print(f'Epoch {i}, Error: {cost_value}')
        epoch_error.append(cost_value)
        param_list.append(initial_parameters)

    print("\n")
    print("Training Completed, Final Parameters are : ")
    print(initial_parameters)
    
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
    plt.ylabel("Cost Value")
    plt.title("Cost vs Epoch ")
    plt.show()

def plot(X, Y, parameters):
    '''
    Plots actual data vs fitted polynomial graph on same fig
    '''

    # Extracting parameters
    a, b, c = parameters["a"], parameters["b"], parameters["c"]
    
    # Plotting
    plt.scatter(X, Y, color='b')  # Plotting original data points
    plt.plot(X, a * X**2 + b * X + c, color='r', linewidth=1)  # Plotting fitted polynomial
    
    # Adding labels and title
    plt.title('Final Polynomial Fitting using Gradient Descent')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Adding legend and grid
    plt.legend(['Original Data', 'Fitted Polynomial'])
    plt.grid(True)
    
    # Displaying the plot
    plt.show()

def animation_plot(X, Y, param_data, save_file = False):
    '''
    Displays animation of the algorithm learning to fit the data using gradient descent.
    '''
    fig, ax = plt.subplots()

    def update(i):
        ax.clear()
        parameters = param_data[i]
        a, b, c = parameters["a"], parameters["b"], parameters["c"]
        ax.scatter(X, Y, color='b')  # Plotting original data points
        ax.plot(X, a * X**2 + b * X + c, color='r', linewidth=1)  # Plotting fitted polynomial
        ax.set_title('Polynomial Fitting using Gradient Descent: Epoch ' +  str(i) )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(['Original Data', 'Fitted Polynomial'])
        ax.grid(True)

    ani = FuncAnimation(fig, update, frames=len(param_data), interval=200)
    #ani.save(filename="polynomial_regression.gif", writer="pillow")
    plt.show()

    

 

        