import polynomial_regressor
import argparse


def get_param_list(arg):
    return tuple(map(float,arg.split(",")))

if __name__ == "__main__" :
    message = '''
    Please input the parameter A, B , C. 
    These parameter will be used to generate the data on which the polynomial regression will be 
    done using Gradient Descent Algorithm. If you want to do linear regression, just make A = 0
    F(x) = Ax^2 + BX + C
    \n
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type = get_param_list , help = " \n Pass the 3 parameters value seperated by comma")
    parser.add_argument("--epochs", type = int , help = " \n Pass the epochs to train for")
    parser.add_argument("--alpha", type = float , help = "\n Pass the learning rate")
    args = parser.parse_args()
    X,Y = polynomial_regressor.data_generator(args.param)
    parameters, iteration_data , param_data = polynomial_regressor.train(X,Y,learning_rate= args.alpha ,epoch = args.epochs,iteration_data = True , param_data = True)
    polynomial_regressor.animation_plot(X,Y,param_data)
    polynomial_regressor.plot_epochs(iteration_data)
    polynomial_regressor.plot(X,Y,parameters)



