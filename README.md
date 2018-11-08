
NaiveBayes_emailClassifier
Input File:
The input file SMSSpamCollection has 5574 ham/spam labeled messages.

How to run:
Download the input file.
run the python code. It asks for path of the downloaded file. Give the absolute path to file

Summary Of the Code:
The dataset is divided into 75:25 ratio for train and test.
The Likelihood , Prior is calculated and stored in python dictionary during training.
The testing phase calculates Posterior Probability .
The output is confusion matrix for hyperparameter(alpha) as 0.1. 
training/testing accuracy and Fscore is plotted for different value hyperparameter.
