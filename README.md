# Classification and Regression

- # Problem 1: Experiment with Gaussian Discriminators

Implement Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA). Implement two functions in Python: ldaLearn and qdaLearn which take a training data set (a feature matrix and labels) and return the means and covariance matrix (or matrices). 
Implement two functions ldaTest and qdaTest which return the true labels for a given test data set and the accuracy using the true labels for the test data. The format of arguments and the outputs is provided in the base code.

- # Problem 2: Experiment with Linear Regression
Implement ordinary least squares method to estimate regression parameters by minimizing the squared loss.

- # Problem 3: Experiment with Ridge Regression
Implement parameter estimation for ridge regression by minimizing the regularized squared loss as follows:

- # Problem 4: Using Gradient Descent for Ridge Regression Learning
Implement the gradient descent procedure for estimating the weights w. Plot the errors on train and test data obtained by using the gradient descent based learning by varying the regularization parameter lambda.

- # Problem 5: Non-linear Regression

Implement the function mapNonLinear which converts a single attribute x into a vector of p attributes,
1; x; x2;....; xp.
<br>
Using the lambda = 0 and the optimal value of lambda found in Problem 3, train ridge regression weights using the
non-linear mapping of the data. Vary p from 0 to 6. Note that p = 0 means using a horizontal line as the
regression line, p = 1 is the same as linear ridge regression. Compute the errors on train and test data.
Compare the results for both values of lambda. What is the optimal value of p in terms of test error in each
setting? Plot the curve for the optimal value of p for both values of lambda and compare.


# Handwritten Digit Classification using Neural Networks
- # Task
<ul>
<li>Implement Neural Network (forward pass and back propagation)</li>
<li>Incorporate regularization on the weights </li>
<li>Use validation set to tune hyper-parameters for Neural Network (number of units in the hidden layer
and lambda).</li>
<li>Run the deep neural network code we provided and compare the results with normal neural network.</li>
<li>Run the convolutional neural network code and print out the results, for example the confusion matrix.</li>
  <li>Write a report to explain the experimental results.</li>
</ul>
