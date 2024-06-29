import logistic_regressor
import argparse


if __name__ == "__main__" :
   
    parser = argparse.ArgumentParser() 
    parser.add_argument("--epochs", type = int , help = " \n Pass the epochs to train for")
    parser.add_argument("--alpha", type = float , help = "\n Pass the learning rate")
    args = parser.parse_args()
    X,Y = logistic_regressor.data_generator(2)
    parameters, iteration_data , param_data = logistic_regressor.train(X,Y,learning_rate= args.alpha ,epoch = args.epochs,iteration_data = True , param_data = True)
    logistic_regressor.animation_plot(X,Y,param_data)
    logistic_regressor.plot_epochs(iteration_data)
    logistic_regressor.plot(X,Y,parameters)



