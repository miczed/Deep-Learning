import math
import numpy
import scipy


# y: prediction (output layer 2), dim: (n x 1)
# t: training, dim: (n x 1)
# w2: weights from layer 2, dim: (k+1 x 1)
# h: first layer output, dim: (k+1 x n)
# x: first layer input, dim: (2 x n)
# returns g1, dim (k, 2)
from matplotlib import pyplot


def first_gradient(y, t, w2, h, x):
    # TODO: understand this fully
    g1 = numpy.dot(
        numpy.outer(w2, y - t) * h * (1 - h),  # dim: (k+1 x n)
        x.T # dim: (n x 2)
    )/len(x)
    # as of now g1 has dim (k+1 x n) but we need (k x n)
    # so we need to discard first row
    # delete 0th element along axis=0 (deletes row)
    g1 = numpy.delete(g1, 0, axis=0)
    return g1


# y: prediction (output layer 2) dim: (n x 1)
# t: training, dim: (n x 1)
# h: first layer output, (4 x n) = (k+1 x n)
# returns g2: vector w/ gradient per hidden node, dim: (k+1 x 1)
def second_gradient(y, t, h):
    # the samples are stored in the ROW
    # thus we cant to calculate the mean per ROW and not per COL
    # setting axis = 1, will calculate the mean per ROW (axis=0 per COL)
    # see: https://www.sharpsightlabs.com/blog/numpy-axes-explained/ for further details
    g2 = 2*numpy.mean((y-t)*h, axis=1)
    return g2


# logistic activation function
def logistic_activation(a):
    return 1./(1+numpy.exp(-a))  # using numpy here allows us to do it on vector basis


# y: output of second layer
# t: training data
def compute_loss(y, t):
    return numpy.mean((y - t) ** 2)/2


def network(x, w1, w2):

    assert x.shape[0] == w1.shape[1], F"dimensions of x {x.shape} and w1 {w1.shape} must match"

    # first layer activation
    a = numpy.dot(w1, x)
    h = logistic_activation(a)

    assert h.min() >= 0 and h.max() <= 1, "logistic function should output values in the range of [0,1]"

    # add bias neuron
    h0 = numpy.ones(h.shape[1])
    h = numpy.vstack((h0, h))

    assert h.shape[0] == w2.shape[0], F"dimensions of h {h.shape} and w2 {ww.shape} must match"

    # second layer activation
    y = logistic_activation(numpy.dot(w2, h))
    return y, h


# x: inputs (without bias!!!)
# t: training labels
# hidden_neurons: number of hidden nodes
# eta: learning rate
# num_epochs: number of epochs
def training(x, t, hidden_neurons, eta=0.01, num_epochs = 1000):

    epoch = 0
    k = hidden_neurons
    y = []
    losses = []

    # ---- Add Bias Neuron to Input ----
    x0 = numpy.ones(x.shape)  # creates n X 1 matrix with ones
    x = numpy.vstack((x0, x))  # adds 1 as first column to matrix

    # ---- Initialize Weights ----

    # w1 dimensionality:
    #  K (nodes in hidden layer) * x + 1 (dim of input + bias)
    # 1st param: row count, 2nd param: col count
    # creates uniform distribution
    w1 = numpy.random.rand(k, 2) * 2. - 1.  # between [-1.0, 1.0)

    # w2 dimensionality:
    # K nodes in hidden layer + 1 x outputs = k+1 x 1
    w2 = numpy.random.random(k+1) * 2. - 1.  # between [-1.0, 1.0)

    # ---- Training ----
    while epoch < num_epochs:  # g2 or scipy.linalg.norm(g2) > 1e-6
        # ---- Forward Pass ----
        y, h = network(x, w1, w2)

        loss = compute_loss(y, t)
        losses.append(loss)

        # ---- Backpropagation ----
        # Gradient Calculation
        # Happens first with old weights prior to learning
        g2 = second_gradient(y, t, h)
        g1 = first_gradient(y, t, w2, h, x)

        # Weight Update -> learning
        w2 = w2 - eta * g2
        w1 = w1 - eta * g1

        print(f"Epoch: {epoch}, Loss: {loss}")

        epoch += 1
    return y, losses

# generates a number of samples in the range
# [start, end)
def generate_samples(start, end, count):
    return numpy.random.random(count) * (end-start) + start


def cosine(x):
    return (numpy.cos(3*x)+1)/2


def gaussian(x):
    return math.e ** (-1/4 * x ** 2)


def polynomial(x):
    return (x**5 + 3 * x**4 - 11 * x**3 - 27 * x**2 + 10 * x + 64)/100


# --- Cosine ----
x = generate_samples(-2., 2., 100)
t = cosine(x)
y, losses = training(x, t, hidden_neurons=5, eta=0.1, num_epochs=10000)

# In General:
# - the more neurons you have, the better the approximation is (the easier it is to approx. the function)
# - if there are jumps (i.e. upticks) in the loss graph, then this means
#   that the learning rate is set too high (will happen more often with more neurons)
# - the amount of training samples also affects how well the model is able to
#   predict the values. -> more training samples -> better predictions

fig, (ax1, ax2) = pyplot.subplots(1, 2)
fig.suptitle('Cosine Approximation')

# plot training data
ax1.plot(x, t, "rx", label="cos()")
ax1.plot(x, y, "ko", label="y")
ax1.legend()

# plot loss
ax2.plot(losses, label="loss")
ax2.legend()
pyplot.show()

