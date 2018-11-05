# Lab 2 of Neural Network from Scratch

Welcome to Lab 2 of the machine learning series.

To let you familiarize with the vectorized implementation using numpy, we have created a little task exercise 2a. Run the exercise2a_notebook in jupyter notebook environment. There you can follow instructions and implement two loss functions namely Mean Square Error and Cross Entropy. You will also learn about, when dealing with a large amount of data, how much faster the vectorized implementation is when compared with the traditional implementation using for-loop.

To run the exercise 2a, you must install Jupyter in your computer. Follow the installation instructions in our repo python-basics https://github.com/vtcstem/python-basics for reference.

The codes for forward propagation have been implemented in neuralnetwork.py. You may check your answer with your lab 1 exercise.

Try to implement the back propagation. When you have done, execute run_example.py. 

If you have coded correctly, you can try using other activation function e.g. tanh for the forward and back propagation. You can also try the parameters e.g. learning rate, number of iterations, and even the dataset to test the performance of your neural network.

To show the parameters you can tweek, run in the terminal 

**python run_example.py -h** and display the help page.

usage: run_examples.py [-h] [--output] [-d DATASET] [-lr LR] [-i ITER]

optional arguments:
  -h, --help            show this help message and exit
  --output              output trained weights
  -d DATASET, --dataset DATASET
                        choose dataset: moon / planar
  -lr LR, --learningrate LR
                        define the learning rate
  -i ITER, --iteration ITER
                        define the learning rate
