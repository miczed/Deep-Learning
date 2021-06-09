import numpy
from matplotlib import pyplot

# sample 100 x locations in range [-2,2]
X = numpy.random.random(100) * 4 -2
# good values for polynomial in range [-4.5, 3.5]
# X = numpy.ranodm.random(100) * 8 - 4.5


# 1. Cosine waves
cos = numpy.cos(X*3.) / 2. + 0.5

# 2. Gaussian bell curve
sigma = .5
gauss = numpy.exp(-X**2 / sigma**2)

# 3. Polynome
poly = (X**5 + 3*X**4 - 11*X**3 - 27*X**2 + 10*X + 64)/100

# add x_0 to the input
X = numpy.vstack((numpy.ones(X.shape), X))

# 2-layer network with K- hidden nodes
K = 4
eta = 0.1
T = cos

# randomly initialize weights
w1 = numpy.random.random((K+1, 2)) * 2. - 1.
w2 = numpy.random.random(K+1) * 2. - 1.


# logistic activation function
def logistic(x):
    return 1./(1+numpy.exp(-x))


# computes the network output for the given inputs
def network(x):
    a = numpy.dot(w1,x)
    h = logistic(a)
    # this discards the calculations from the first layer in h[0]
    # and sets it to 1 as our bias for the second layer
    # this works because w1 was initialized with k+1 instead of k dims
    h[0] = 1.
    return numpy.dot(w2, h), h


# computes the loss for the whole dataset
def loss():
    Y = network(X)[0]
    # the following line is missing dividing it by two
    # that's just an implementation detail
    # you could also just multiply the learning rate by 2
    return numpy.mean((Y-T)**2)

# computes the gradient, i.e., for both w1 and w2
def gradient():
    # network output and hidden states for all inputs
    Y, H = network(X)

    # gradient for w2
    # we multiply the vector (Y-T) with dim (n) with the matrix H with dim (k+1 x n)
    # the * is NOT a dot product! it's a broadcast multiplication
    # the result of that multiplication is a matrix of dim (k+1 x n)
    # over that matrix we calculate the mean over all samples on axis 1
    # which results in a vector of dim k+1
    g2 = numpy.mean((Y-T) * H, axis=1)

    # gradient for w1
    g1 = numpy.dot(
        numpy.outer(w2, Y-T) * H * (1-H),
        X.T
    )
    return g1 / len(X), g2


# gradient descent
g1, g2 = gradient()

progression = []

for e in range(10000):
    # update weights
    w1 -= eta * g1
    w2 -= eta * g2

    # compute loss
    progression.append(loss())

    # and new gradient
    g1, g2 = gradient()

# plot everything together
# linearly spaced samples
x = numpy.arange(numpy.min(X), numpy.max(X), 0.01)
x = numpy.vstack((numpy.ones(x.shape),x))

pyplot.plot(X[1], T, "rx")
pyplot.plot(x[1], network(x)[0], "k-")
pyplot.savefig("Data.pdf")

# plot loss progression
pyplot.figure()
pyplot.plot(progression, label="Loss")
pyplot.legend()
pyplot.savefig("Loss.pdf")