import numpy as np


class MultiLayerPerceptron():
    """A feedforward neural network"""
    def __init__(self, architecture, lambda_=0, init_theta=None,
                 output_layer='softmax'):
        self.architecture = architecture
        self.depth = len(architecture) - 1
        self.lambda_ = lambda_
        self.output_layer = output_layer

        if isinstance(init_theta, np.ndarray):
            self.theta = init_theta
            self.thetas = self.extract_thetas(self.theta)
        else:
            # Randomly initialize weights
            self.thetas = []
            for i in range(self.depth):
                init_theta = self.rand_initialize_weights(architecture[i],
                                                          architecture[i + 1])
                self.thetas.append(init_theta)

            # Unroll parameters (weights)
            thetas = list(map(lambda x: x.ravel(), self.thetas))
            self.theta = np.hstack(thetas).reshape(-1)

    def rand_initialize_weights(self, l_in, l_out):
        epsilon_init = np.sqrt(6) / np.sqrt(l_in + l_out)
        weights = np.random.default_rng().uniform(size=(l_out, l_in + 1))
        weights = weights * 2 * epsilon_init - epsilon_init
        return weights

    def sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))

    def softmax(self, z):
        exps = np.exp(z - z.max(axis=1, keepdims=True))
        return exps / exps.sum(axis=1, keepdims=True)

    def feedforward(self, thetas, X):
        m = X.shape[0]
        intercept = np.ones((m, 1))
        activations = []
        a = np.concatenate((intercept, X), axis=1)
        activations.append(a)
        for i in range(self.depth - 1):
            a = self.sigmoid(a @ thetas[i].T)
            a = np.concatenate((intercept, a), axis=1)
            activations.append(a)
        if self.output_layer == 'softmax':
            h = self.softmax(a @ thetas[-1].T)
        elif self.output_layer == 'sigmoid':
            h = self.sigmoid(a @ thetas[-1].T)
        activations.append(h)
        return activations

    def extract_thetas(self, theta):
        thetas = []
        index = 0
        for i in range(self.depth):
            in_size = self.architecture[i]
            out_size = self.architecture[i + 1]
            extracted_theta = theta[index:index + out_size * (in_size + 1)]
            extracted_theta = extracted_theta.reshape(out_size, in_size + 1)
            thetas.append(extracted_theta)
            index += out_size * (in_size + 1)
        return thetas

    def crossentropy_loss(self, y_matrix, h, thetas):
        m = h.shape[0]
        if self.output_layer == 'sigmoid':
            J = - (y_matrix * np.log(h)
                   + (1 - y_matrix) * np.log(1 - h)).sum() / m
        elif self.output_layer == 'softmax':
            J = - (y_matrix * np.log(h)).sum() / m

        # Add regularization term
        J = J + self.lambda_ / (2 * m) * sum(
            [(theta[:, 1:] ** 2).sum() for theta in thetas])
        return J

    def nn_cost_function(self, init_theta, X, y):
        # Extract thetas from init_theta
        thetas = self.extract_thetas(init_theta)

        # Feedforward the neural network
        activations = self.feedforward(thetas, X)
        h = activations.pop()

        # One-hot encode the labels
        labels_vec = np.eye(self.architecture[-1])
        y_matrix = labels_vec[y.ravel(), :]

        # Calculate the cost function
        J = self.crossentropy_loss(y_matrix, h, thetas)

        # Backpropagation
        m = X.shape[0]
        thetas_grad = []
        delta = h - y_matrix
        theta_grad = delta.T @ activations[-1] / m
        thetas_grad.append(theta_grad)
        for i in range(self.depth - 1):
            delta = (delta @ thetas[-1 - i])\
                * activations[-1 - i] * (1 - activations[-1 - i])
            delta = delta[:, 1:]
            theta_grad = delta.T @ activations[-2 - i] / m
            thetas_grad.append(theta_grad)
        thetas_grad = thetas_grad[::-1]

        # Add regularization term
        for i in range(self.depth):
            reg = thetas[i].copy()
            reg[:, 0] = 0
            thetas_grad[i] = thetas_grad[i] + self.lambda_ * reg / m

        # Unroll parameters (weights)
        thetas_grad = list(map(lambda x: x.ravel(), thetas_grad))
        grad = np.hstack(thetas_grad).reshape(-1)
        return (J, grad)

    def compute_numerical_gradient(self, J, nn_params):
        n = nn_params.shape[0]
        numgrad = np.zeros_like(nn_params)
        perturbation = np.zeros_like(nn_params)
        epsilon = 1e-4
        for i in range(n):
            perturbation[i] = epsilon
            nn_params_plus = nn_params + perturbation
            nn_params_minus = nn_params - perturbation
            (J_plus, _) = J(nn_params_plus)
            (J_minus, _) = J(nn_params_minus)
            numgrad[i] = (J_plus - J_minus) / (2 * epsilon)
            perturbation[i] = 0
        return numgrad

    def fit(self, x_train, y_train, x_valid, y_valid, alpha, epochs,
            batch_size=-1):
        indices = range(batch_size, x_train.shape[0], batch_size)
        x_batches = np.split(x_train, indices)
        y_batches = np.split(y_train, indices)
        n_batches = len(x_batches)
        J_train_history, J_valid_history = [], []
        for i in range(epochs):
            x_batch = x_batches[i % n_batches]
            y_batch = y_batches[i % n_batches]
            J_train, grad = self.nn_cost_function(self.theta, x_batch, y_batch)
            J_valid, _ = self.nn_cost_function(self.theta, x_valid, y_valid)
            if (i > 1000 and J_valid > J_valid_history[-100]):
                break
            J_train_history.append(J_train)
            J_valid_history.append(J_valid)
            print(f'epoch {i}/{epochs} - loss: {J_train:.4f}\
 - val_loss: {J_valid:.4f}')
            self.theta = self.theta - alpha * grad

        # Extract thetas from optimal theta
        self.thetas = self.extract_thetas(self.theta)
        return J_train_history, J_valid_history

    def predict(self, X, verbose=False, y=None):
        activations = self.feedforward(self.thetas, X)
        h = activations[-1]
        predictions = np.argmax(h, axis=1).reshape(-1, 1)
        if (verbose):
            for _, raw in zip(np.hstack([y, predictions]), h):
                if (_[0] == _[1]):
                    print(f'-> {tuple(_)} - raw{raw}')
                else:
                    print(f'-> {tuple(_)} - raw{raw} <<')
        return predictions

    def mse(self, predictions, y):
        return ((predictions - y) ** 2).sum() / y.shape[0]
