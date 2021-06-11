import numpy

# download dataset from UCI
# inputs: 4 attributes:
# - variance of Wavelet Transformed image
# - skewness of Wavelet transformed image
# - curtosis of wavelet transformed image
# - entropy of image
# output: forged (0) or real bank note (1)
# task: how many hidden neurons to reach 100% accuracy

# logistic activation function
from matplotlib import pyplot


def logistic_activation(a):
    return 1./(1+numpy.exp(-a))  # using numpy here allows us to do it on vector basis

# adds a bias to an input array of dim (N x D)
# returns: matrix of dim(N x D+1)
def add_bias(X):
    # add bias to X x0 dim(n x 1)
    x0 = numpy.ones((X.shape[0], 1))  # creates n X 1 matrix with ones
    # adds 1 as first column to matrix
    return numpy.hstack((x0, X))  # X has now dim(N x d+1)


# Task 2: implement network
# W1: first layer weights: dim (K+1 x D+1) ingore K_0 -> bias for H
# W2: second layer weights: dim (O x K+1)
# X: input with bias: dim (N x D+1)
# returns:
# Y: output dim(O x N)
# Z: logits dim(O x N)
# H: hidden layer activation dim(K+1 x N)
def forward(X, W1, W2):
    A = numpy.dot(W1, X)  # dim: (K+1 x N)
    H = logistic_activation(A)  # dim: (K+1 x N)

    # set first row of H to 1 -> bias for 2nd layer
    H[0] = 1.

    Z = numpy.dot(W2, H)  # logits dim: (O x N)

    Y = logistic_activation(Z)  # dim: (O x N)

    return Y, Z, H

# computes the cross entropy loss for binary classification
# Y: network output: dim (O x N)
# T: training labels: dim(O x N)
# returns loss
def binary_cross_entropy_loss(Y, T):
    return -numpy.sum(T.T * numpy.log(Y) + (1-T).T * numpy.log(1-Y))


# X: inputs dim(D+1xN) (+1 -> bias)
# Y: outputs dim(OxN)
# T: training dim(OxN)
# W1: first layer weights: (K+1xD+1)
# W2: second layer weights: (OxK+1)
# H: hidden layer activation: (K+1xN)
# returns:
# g2: second layer gradient dim(OxK+1)
# g1: first layer gradient dim(k+1xD+1)
def compute_gradient(X, Y, T, W2, H):
    n = Y.shape[1]
    g2 = 2 / n * numpy.dot((Y-T), H.T)

    g1 = 2 / n * numpy.dot(
        (numpy.dot(W2.T, (Y-T)) * H * (1-H))
        , X.T
    )
    return g1, g2


def descent(X, T, W1, W2, eta):
    Y, Z, H = forward(X, W1, W2)
    loss = binary_cross_entropy_loss(Y, T)

    # compute gradients
    G1, G2 = compute_gradient(X, Y, T, W2, H)

    # perform update
    W1 = W1 - eta * G1
    W2 = W2 - eta * G2

    return Y, W1, W2, loss


def train(X, T, hidden_neurons, eta, epochs):
    epoch = 0
    input_dim = X.shape[1]
    output_dim = T.shape[1]
    losses = numpy.array([])
    accuracies = numpy.array([])

    # initialize weights
    # W1 dim(k+1, D+1)
    W1 = numpy.random.rand(hidden_neurons + 1, input_dim) * 2. - 1.  # between [-1.0, 1.0)

    # W2 dim(O x K+1)
    W2 = numpy.random.rand(output_dim, hidden_neurons + 1)

    # transpose X and T so they match DIM of formulas used in lecture
    X = X.T
    T = T.T

    while epoch < epochs:
        Y, W1, W2, loss = descent(X, T, W1, W2, eta)
        losses = numpy.append(losses, loss)
        accuracy = numpy.sum(Y == T)/T.size
        accuracies = numpy.append(accuracies, accuracy)
        print(f"\rEpoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}", end="", flush=True)
        epoch += 1
    return losses, accuracies


# Task 1: Load dataset
data = numpy.genfromtxt('data/data_banknote_authentication.txt', delimiter=",", dtype=float, encoding=None)
input = data[:, [0, 1, 2, 3]]  # dim: (N x D)
output = data[:, [4]]  # dim: (N x O)

X = add_bias(input)  # dim: (N x D+1)

losses, acc = train(X, output, 200, 0.005, 10e2)
pyplot.plot(losses, label="Binary Cross-Entropy Loss")
pyplot.plot(acc, label="Accuracy")
pyplot.xlabel("epochs")
pyplot.ylabel("loss")

pyplot.legend()
pyplot.show()