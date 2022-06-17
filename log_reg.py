#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLR


ALPHA = 1e-4
MAX_ITER = 2 ** 7

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

    # Train a logistic regression model using the one-vs-all method
    thetas = []
    for d in range(10):
        # Train a logistic regression model to determine
        # whether an image is of digit d or not
        print(f'Training logistic regression model number {d + 1}')
        mylr = MyLR(np.zeros((n + 1, 1)), alpha=ALPHA,
                    max_iter=MAX_ITER, lambda_=0.1)
        new_y_train = (y_train == d).astype(float)
        mylr.fit_(x_train, new_y_train)
        thetas.append(mylr.thetas)
    all_thetas = np.hstack(thetas)

    # Calculate model predictions on the training set
    X_train = np.hstack([np.ones((m_train, 1)), x_train])
    all_predictions = X_train @ all_thetas
    predictions_train = np.argmax(all_predictions, axis=1).reshape(-1, 1)

    # Calculate accuracy
    acc = np.mean(predictions_train == y_train)
    print(f'Training set accuracy = {acc}')

    # Calculate model predictions on the test set
    X_test = np.hstack([np.ones((m_test, 1)), x_test])
    all_predictions = X_test @ all_thetas
    predictions_test = np.argmax(all_predictions, axis=1).reshape(-1, 1)

    # Calculate accuracy
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
