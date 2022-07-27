#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model import MultiLayerPerceptron


def check_nn_gradients():
    input_layer_size = 3
    hidden_layer_1_size = 4
    hidden_layer_2_size = 4
    num_labels = 3
    m = 5
    lambda_ = 2

    architecture = [input_layer_size, hidden_layer_1_size,
                    hidden_layer_2_size, num_labels]
    mlp = MultiLayerPerceptron(architecture,
                               lambda_=lambda_, output_layer='softmax')

    X = np.random.default_rng().uniform(size=(m, input_layer_size))
    y = (np.arange(m) % num_labels).reshape(-1, 1)

    def nn_cost(theta):
        return mlp.nn_cost_function(theta, X, y)

    J, grad = nn_cost(mlp.theta)
    grad = grad.reshape(-1)

    numgrad = mlp.compute_numerical_gradient(nn_cost, mlp.theta)
    diff = np.sqrt(((numgrad - grad) ** 2).sum())\
        / np.sqrt(((numgrad + grad) ** 2).sum())
    print(f'\nThe relative difference between our gradient (analytical)\
 and the numerical gradient is {diff}')
    print(np.hstack([grad.reshape(-1, 1), numgrad.reshape(-1, 1)]))


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
    rng = np.random.default_rng(1337)
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

    # Gradient checking
    check_nn_gradients()

    # Neural network architecture parameters
    input_layer_size = 784
    hidden_layer_size = 50
    num_labels = 10
    lambda_ = 0.1

    # Set the models's architecture (the size of each layer)
    architecture = [input_layer_size, hidden_layer_size,
                    hidden_layer_size, num_labels]
    output_layer = 'softmax'
    mlp = MultiLayerPerceptron(architecture, lambda_=lambda_,
                               output_layer=output_layer)

    # Train the model using gradient descent
    J_train_history, J_test_history = mlp.fit(x_train, y_train, x_test, y_test,
                                              0.2, 500, batch_size=1000)

    # Calculate model predictions on the training and test sets
    predictions_train = mlp.predict(x_train)
    predictions_test = mlp.predict(x_test)

    # Calculate the mean squared error on the training and test sets
    mse_train = mlp.mse(predictions_train, y_train)
    mse_test = mlp.mse(predictions_test, y_test)
    print(f'\n> Training set loss (mean squared error) : {mse_train:.4f}')
    print(f'> Test set loss (mean squared error) : {mse_test:.4f}')

    # Calculate training set accuracy
    acc = np.mean(predictions_train == y_train)
    print(f'> Training set accuracy = {acc:.4f}')

    # Calculate test set accuracy
    acc = np.mean(predictions_test == y_test)
    print(f'> Test set accuracy = {acc:.4f}')

    # Plot the learning curves
    plt.plot(J_train_history, label='Training Loss')
    plt.plot(J_test_history, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

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
