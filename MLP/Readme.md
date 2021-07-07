# Multi Layer Perceptron

### Introduction:
Multi layer perceptron (MLP) is a supplement of feed forward neural network. It consists of three types of layers—the input layer, output layer and hidden layer, as shown in Fig. The input layer receives the input signal to be processed. The required task such as prediction and classification is performed by the output layer. An arbitrary number of hidden layers that are placed in between the input and output layer are the true computational engine of the MLP. Similar to a feed forward network in a MLP the data flows in the forward direction from input to output layer. The neurons in the MLP are trained with the back propagation learning algorithm. MLPs are designed to approximate any continuous function and can solve problems which are not linearly separable. The major use cases of MLP are pattern classification, recognition, prediction and approximation.

![alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0065245819300506-f14-03-9780128187562.jpg)

The computations taking place at every neuron in the output and hidden layer are as follows,
* **o(x)=G(b(2)+W(2)h(x))  --- (1)**
* **h(x)=s(b(1)+W(1)x)     --- (2)**

with bias vectors b(1), b(2); weight matrices W(1), W(2) and activation functions G and s. The set of parameters to learn is the set θ = {W(1), b(1), W(2), b(2)}. Typical choices for s include tanh function with tanh(a) = (e^a − e^−a)/(e^a + e^−a) or the logistic sigmoid function, with sigmoid(a) = 1/(1 + e^−a) or Relu etc.

### Model:
The model consist of Multi layer perceptron with 3 layers of nodes. The hidden layer has 5, the input layer has 4 and the output layer has 3 neurons. During training, all the weights and biases are updated after processing single training sample and the method is called stochastic training. The update rule uses learning rate of 0.1 and momentum factor of 0.001. The code is run for 500 epochs. The activation function is sigmoid of the form f(x) = 1 / 1 + exp(-x) Its range is between 0 and 1. It is S — shaped curve (**Note: In case of Deep Neural Network Sigmoid activation function can cause vanishing gradient problem, because the gradient of sigmoid activation function will be ranges between 0-0.25. so to avoid this problem we generally use ReLU or leaky ReLU activation function in hidden layers**). The performance of classification is measured using confusion matrix. The code computes specificity, sensitivity, precision, recall, accuracy and F-score. The model uses 5-fold cross validation to improve the prediction accuracy and to reduce variance. The dataset is divided into 5 folds wherein 4 subsamples are used to train the network and the 5th subsample tests the model.The process is repeated for each fold of the dataset. The dataset is Iris dataset, Which contains five columns such as Petal Length, Petal Width, Sepal Length, Sepal Width and Species Type. Cross_validation_split divides the dataset into k folds of equal samples. Run_algorithm takes one fold at a time, considers the fold as test set and trains the network with samples of the remaining (merged) folds. Back_propagation function initializes the network by assigning random weights to the connections of input, hidden and output layers. The train_network trains the network by first propagating the inputs through a system of weighted connections. For each input sample, the intermediate layer computes weighted sum, derives activation function value and forwards the output of the neuron to the next layer. The output layer compares the expected and the obtained neuron value and back propagates the difference so that the weight adjustments can minimise error for the next iteration. activate function computes the weighted sum of inputs, transfer function calculates neuron output due to the activation function, backward_propagate_error derives error at every layer using the delta rule, and update_weight stores the modified weights of neurons at each layer of the network. Confusion_matrix uses the actual and predicted output to generate values for false positives, false negatives, true positives and true negatives.
#### Model Results
1. For Fold 1:
   *  Confusion matrix : 11   0   0
                         0    7   1
                         0    0  11
   *  Sensitivity:  [1.00 0.87 1.00]
   *  Specificity : [1.00 1.00 0.94]
   *  Precision :   [1.00 1.00 0.91]
   *  Recall :      [1.00 0.87 1.00]
   *  FScore :      [1.00 0.933 0.95]
   *  Accuarcy: 96.66

2. For Fold 2:
   *  Confusion matrix :  10   0   0
                          0    9   2
                          0    0   9
   *  Sensitivity:  [1.00 0.81 1.00]
   *  Specificity : [1.00 1.00 0.90]
   *  Precision :   [1.00 1.00 0.81]
   *  Recall :      [1.00 0.81 1.00]
   *  FScore :      [1.00 0.90 0.90]
   *  Accuarcy: 93.33

3. For Fold 3:
   *  Confusion matrix :   6   0   0
                           0  10   0
                           0   0  14
   *  Sensitivity:  [1.00 1.00 1.00]
   *  Specificity : [1.00 1.00 1.00]
   *  Precision :   [1.00 1.00 1.00]
   *  Recall :      [1.00 1.00 1.00]
   *  FScore :      [1.00 1.00 1.00]
   *  Accuarcy: 100.00
 
4. For Fold 4:
   *  Confusion matrix :  10  0   0
                          0   9   1
                          0   0  10
   *  Sensitivity:  [1.00 0.90 1.00]
   *  Specificity : [1.00 1.00 0.95]
   *  Precision :   [1.00 1.00 0.90]
   *  Recall :      [1.00 0.90 1.00]
   *  FScore :      [1.00 0.94 0.95]
   *  Accuarcy: 96.66

5. For Fold 5:
   *  Confusion matrix : 13  0   0
                         0   7   4
                         0   0   6
   *  Sensitivity:  [1.00 0.63 1.00]
   *  Specificity : [1.00 1.00 0.83]
   *  Precision :   [1.00 1.00 0.60]
   *  Recall :      [1.00 0.63 1.00]
   *  FScore :      [1.00 0.77 0.75]
   *  Accuarcy: 86.66

   
 
 
 






