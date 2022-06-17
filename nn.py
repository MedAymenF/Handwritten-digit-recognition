#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def rand_initialize_weights(l_in, l_out):
    epsilon_init = np.sqrt(6) / np.sqrt(l_in + l_out)
    weights = np.random.default_rng().uniform(size=(l_out, l_in + 1))
    weights = weights * 2 * epsilon_init - epsilon_init
    return weights


def feedforward(theta1, theta2, X):
    m = X.shape[0]
    intercept = np.ones((m, 1))
    a1 = np.concatenate((intercept, X), axis=1)
    a2 = sigmoid(a1 @ theta1.T)
    a2 = np.concatenate((intercept, a2), axis=1)
    h = sigmoid(a2 @ theta2.T)
    return a1, a2, h


def nn_cost_function(nn_params, input_layer_size,
                     hidden_layer_size, num_labels, X, y, lambda_):
    # Exctract theta1 and theta2 from nn_params
    theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(
        (hidden_layer_size, input_layer_size + 1))
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(
        (num_labels, hidden_layer_size + 1))

    # Feedforward the neural network
    a1, a2, h = feedforward(theta1, theta2, X)

    # Calculate the cost function
    m = X.shape[0]
    labels_vec = np.eye(num_labels)
    y_matrix = labels_vec[y.ravel(), :]
    J = - (y_matrix * np.log(h) + (1 - y_matrix) * np.log(1 - h)).sum() / m

    # Add regularization term
    J = J + lambda_ / (2 * m) * ((theta1[:, 1:] ** 2).sum()
                                 + (theta2[:, 1:] ** 2).sum())

    # Backpropagation
    delta3 = h - y_matrix
    delta2 = (delta3 @ theta2) * a2 * (1 - a2)
    delta2 = delta2[:, 1:]
    theta2_grad = delta3.T @ a2 / m
    theta1_grad = delta2.T @ a1 / m

    # Add regularization term
    reg_1 = theta1.copy()
    reg_1[:, 0] = 0
    reg_2 = theta2.copy()
    reg_2[:, 0] = 0
    theta1_grad += lambda_ * reg_1 / m
    theta2_grad += lambda_ * reg_2 / m

    # Unroll parameters (weights)
    grad = np.hstack([theta1_grad.ravel(),
                      theta2_grad.ravel()]).reshape(-1)
    return (J, grad)


def compute_numerical_gradient(J, nn_params):
    n = nn_params.shape[0]
    numgrad = np.zeros_like(nn_params)
    perturbation = np.zeros_like(nn_params)
    epsilon = 1e-4
    for i in range(n):
        perturbation[i, 0] = epsilon
        nn_params_plus = nn_params + perturbation
        nn_params_minus = nn_params - perturbation
        (J_plus, _) = J(nn_params_plus)
        (J_minus, _) = J(nn_params_minus)
        numgrad[i, 0] = (J_plus - J_minus) / (2 * epsilon)
        perturbation[i, 0] = 0
    return numgrad


def check_nn_gradients():
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    lambda_ = 3

    theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
    nn_params = np.hstack([theta1.ravel(), theta2.ravel()]).reshape(-1, 1)

    X = rand_initialize_weights(input_layer_size - 1, m)
    y = (np.arange(m) % num_labels).reshape(-1, 1)

    def nn_cost(theta):
        return nn_cost_function(
         theta, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)

    _, grad = nn_cost(nn_params)
    grad = grad.reshape(-1, 1)

    numgrad = compute_numerical_gradient(nn_cost, nn_params)
    diff = np.sqrt(((numgrad - grad) ** 2).sum())\
        / np.sqrt(((numgrad + grad) ** 2).sum())
    print(f'The relative difference between our gradient (analytical)\
 and the numerical gradient is {diff}')
    print(np.hstack([grad, numgrad]))


if __name__ == "__main__":
    # Read dataset
    train_df = pd.read_csv("Dataset/mnist_train.csv")
    test_df = pd.read_csv("Dataset/mnist_test.csv")

    y_train = train_df.pop('label')
    y_test = test_df.pop('label')

    # Convert data to numpy arrays
    x_train = train_df.to_numpy()
    y_train = y_train.to_numpy().reshape(-1, 1)
    x_test = test_df.to_numpy()
    y_test = y_test.to_numpy().reshape(-1, 1)

    # Visualize 10 random images each from the training and test sets
    rng = np.random.default_rng()
    (m_train, n) = x_train.shape
    rand_idx = rng.choice(m_train, 10, replace=False)
    imgs = x_train[rand_idx, :]
    labels = y_train[rand_idx, :]
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        img = imgs[i].reshape(28, 28)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(f'Label = {labels[i, 0]}', fontsize=6)
    m_test = x_test.shape[0]
    rand_idx = rng.choice(m_test, 10, replace=False)
    imgs = x_test[rand_idx, :]
    labels = y_test[rand_idx, :]
    for i in range(10):
        plt.subplot(2, 10, i + 11)
        img = imgs[i].reshape(28, 28)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(f'Label = {labels[i, 0]}', fontsize=6)
    plt.show()

    # Neural network architecture parameters
    input_layer_size = 784
    hidden_layer_size = 25
    num_labels = 10
    lambda_ = 1

    # Gradient checking
    check_nn_gradients()

    # Randomly initialize weights
    init_theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    init_theta2 = rand_initialize_weights(hidden_layer_size, num_labels)

    # Unroll parameters (weights)
    initial_nn_params = np.hstack([init_theta1.ravel(),
                                   init_theta2.ravel()]).reshape(-1)

    # Find the best parameters using scipy.optimize
    result = op.minimize(fun=nn_cost_function, x0=initial_nn_params,
                         args=(input_layer_size, hidden_layer_size,
                               num_labels, x_train, y_train, lambda_),
                         method='CG', jac=True,
                         options={'maxiter': 50, 'disp': True})
    optimal_theta = result.x

    # Exctract theta1 and theta2 from optimal_theta
    theta1 = optimal_theta[:hidden_layer_size * (input_layer_size + 1)]
    theta1 = theta1.reshape((hidden_layer_size, input_layer_size + 1))
    theta2 = optimal_theta[hidden_layer_size * (input_layer_size + 1):]
    theta2 = theta2.reshape((num_labels, hidden_layer_size + 1))

    # Calculate model predictions on the training set
    _, _, h = feedforward(theta1, theta2, x_train)
    predictions_train = np.argmax(h, axis=1).reshape(-1, 1)

    # Calculate training set accuracy
    acc = np.mean(predictions_train == y_train)
    print(f'Training set accuracy = {acc}')

    # Calculate model predictions on the test set
    _, _, h = feedforward(theta1, theta2, x_test)
    predictions_test = np.argmax(h, axis=1).reshape(-1, 1)

    # Calculate test set accuracy
    acc = np.mean(predictions_test == y_test)
    print(f'Test set accuracy = {acc}')

    # Visualize 10 random images each from the training and test sets
    # along with our model's predictions
    rand_idx = rng.choice(m_train, 10, replace=False)
    imgs = x_train[rand_idx, :]
    labels = y_train[rand_idx, :]
    predictions = predictions_train[rand_idx, :]
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        img = imgs[i].reshape(28, 28)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(f'L = {labels[i, 0]} P = {predictions[i, 0]}', fontsize=6)

    rand_idx = rng.choice(m_test, 10, replace=False)
    imgs = x_test[rand_idx, :]
    labels = y_test[rand_idx, :]
    predictions = predictions_test[rand_idx, :]
    for i in range(10):
        plt.subplot(2, 10, i + 11)
        img = imgs[i].reshape(28, 28)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(f'L = {labels[i, 0]} P = {predictions[i, 0]}', fontsize=6)
    plt.show()
