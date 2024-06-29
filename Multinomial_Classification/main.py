import multi_regressor
import argparse


if __name__ == "__main__" :
   
    parser = argparse.ArgumentParser() 
    parser.add_argument("--epochs", type = int , help = " \n Pass the epochs to train for")
    parser.add_argument("--alpha", type = float , help = "\n Pass the learning rate")
    parser.add_argument("--clusters", type = int , help = "\n Pass the number of cluster to generate")
    args = parser.parse_args()
    X,Y = multi_regressor.data_generator(args.clusters) 

    multi_regressor.plt.figure(figsize=(8, 6))
    multi_regressor.plt.title(str(args.clusters) +" Blobs")
    multi_regressor.plt.scatter(X[:, 0], X[:, 1], marker="*", c= Y, s=50, cmap ="Spectral") # Here Y is passed in color argument to show different colors for different classes
    multinomial_classifier = multi_regressor.multinomial_regressor(args.clusters)
    multinomial_classifier.train(X,Y,learning_rate= args.alpha ,epochs = args.epochs)
    multinomial_classifier.plot_epochs()
    multinomial_classifier.plot_decision_boundary(X,Y)
    multinomial_classifier.show_training_animation(X,Y,"multinomial_animation.gif")
 



