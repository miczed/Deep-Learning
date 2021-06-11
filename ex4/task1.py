import math
import pandas as pd
from matplotlib import pyplot
import numpy

# ------------------------
# Task 1: Batch Processing
# ------------------------
# Implement a function that returns a batch of given input samples and target values
# with a given batch size
# - should not only give one batch, but should iterate over training set as long as required
# - each batch should have the same size even if X is not dividable by B
# - shuffle X and T after each epoch


# X: inputs dim: N x D [ (x0,x1), (x0, x1) ... ]
# T: training values N x O (if O = output value dim)
# B: size of the batch
def batch(X, T, B, epochs):
    # figure out how many batches there will be
    n = X.shape[0]
    rest = n % B
    batch_count = math.floor(n/B)
    if rest > 0:
        batch_count += 1

    # gets a list of all indices in X
    indices = list(range(X.shape[0]))

    # shuffle indices in place
    numpy.random.shuffle(indices)
    for e in range(epochs):
        if B == n:
            yield X, T, e+1
        else:
            for b in range(batch_count):
                # yield statement pauses the function saving all its states
                # and later continues from there on successive calls.
                start = b*B
                end = (b+1)*B
                if end >= n:
                    rest = end-n
                    partial_batch = indices[start:n]
                    # shuffle samples again
                    numpy.random.shuffle(indices)
                    # print(F"Batching samples [{start},{n}) with rest: [{0},{rest})")
                    whole_batch = numpy.concatenate((partial_batch, indices[0:rest]))
                else:
                    # print(F"Batching samples [{start},{end})")
                    # return rows start..end and all columns ( : stands for all columns)
                    whole_batch = indices[start:end]
                yield numpy.array([X[i] for i in whole_batch]), numpy.array([T[j] for j in whole_batch]), e+1


x = numpy.array(range(100))
t = x*10
epochs = numpy.array([])
for x, t, e in batch(x, t, 33, 3):
    print(F"epoch: {e} - x length: {len(x)}, t length: {len(t)}")

print("foo")

# ------------------------
# Task 2: Multi-target Network
# ------------------------

# logistic activation function
def logistic_activation(a):
    return 1./(1+numpy.exp(-a))  # using numpy here allows us to do it on vector basis


# X: input matrix dim: (D+1 x N)
# W1: first layer weights:  (K+1 x D+1) (ignore k+1 )
# W2: second layer weights: (O x K+1)
# returns:
# Y: output dim (O x N)
# H: hidden layer activation (K+1 x N)
def forward(X, W1, W2):
    # first layer pass
    H = logistic_activation(numpy.dot(W1, X))

    # set first row of H to 1 -> bias for 2nd layer
    H[0] = 1.

    # second layer
    Y = numpy.dot(W2, H)

    return Y, H


# Y: outputs dim(OxN)
# T: training dim(OxN)
# returns: frobenius norm squared
def compute_loss(Y, T):
    return numpy.linalg.norm(Y-T, 'fro')**2 / Y.shape[1]


# X: inputs dim(D+1xN) (+1 -> bias)
# Y: outputs dim(OxN)
# T: training dim(OxN)
# W1: first layer weights: (K+1xD+1)
# W2: second layer weights: (OxK+1)
# H: hidden layer activation: (K+1xN)
# returns:
# g2: second layer gradient dim(OxK+1)
# g1: first layer gradient dim(
def compute_gradient(X, Y, T, W2, H):
    n = Y.shape[1]
    g2 = 2 / n * numpy.dot((Y-T), H.T)

    g1 = 2 / n * numpy.dot(
        (numpy.dot(W2.T, (Y-T)) * H * (1-H))
        , X.T
    )
    return g1, g2


# ------------------------
# Task 3: Gradient Descent Step
# ------------------------
def descent(X, T, W1, W2, eta, mu=None):
    Y, H = forward(X, W1, W2)
    loss = compute_loss(Y, T)

    g1, g2 = compute_gradient(X, Y, T, W2, H)
    old_W1, old_W2 = None, None
    if mu is not None and old_W1 and old_W2:
        # momentum learning
        W1 = W1 - eta * g1 + mu * (W1-old_W1)
        W2 = W2 - eta * g2 + mu * (W2-old_W2)
        old_W1 = W1
        old_W2 = W2
    else:
        W1 = W1 - eta * g1
        W2 = W2 - eta * g2

    return Y, W1, W2, loss

# ------------------------
# Task 4: Data Set Loading
# ------------------------

data = pd.read_csv('data/student-mat.csv', sep=';',header=0).values

#data = numpy.genfromtxt('data/student-mat.csv', delimiter=";", skip_header=1, dtype=None, encoding=None)

def prep_input_data(row):
    return [
            -1 if row[0] == "GP" else 1,  # 1: school -1:GP, 1:MS
            -1 if row[1] == "F" else 1,  # 2: sex -1:F, 1:M
            float(row[2]),  # 3: age integer
            -1 if row[3] == "U" else 1,  # 4: address -1:U, 1:R
            -1 if row[4] == "LE3" else 1,  # 5: famsize -1:LE3, 1:GT3
            -1 if row[5] == "T" else 1,  # 6: Pstatus -1:T, -1:A
            float(row[6]),  # 7: Medu 0-4
            float(row[7]),  # 8: Fedu 0-4
            # skip 9..12
            float(row[12]),  # 13: traveltime numeric 1-4
            float(row[13]),  # 14: studytime numeric: 1-4
            float(row[14]),  # 15: failures numeric: 1-4
            -1 if row[15] == "yes" else 1,  # 16: schoolsup: -1:yes, 1:no
            -1 if row[16] == "yes" else 1,  # 17: famsup: -1:yes, 1:no
            -1 if row[17] == "yes" else 1,  # 18: paid: -1:yes, 1: no
            -1 if row[18] == "yes" else 1,  # 19: activities -1:yes, 1:no
            -1 if row[19] == "yes" else 1,  # 20: nursery: -1:yes, 1:no
            -1 if row[20] == "yes" else 1,  # 21: higher: -1:yes, 1:no
            -1 if row[21] == "yes" else 1,  # 22: internet: -1:yes, 1:no
            -1 if row[22] == "yes" else 1,  # 23: romantic: -1:yes, 1:no
            float(row[23]),  # 24: famrel: 1-5
            float(row[24]),  # 25: freetime: 1-5
            float(row[25]),  # 26: goout: 1-5
            float(row[26]),  # 27: Dalc: 1-5
            float(row[27]),  # 28: Walc: 1-5
            float(row[28]),  # 29: health: 1-5
            float(row[29]),  # 30: absences: 0-93
        ]

def prep_output_data(row):
    return [
        float(row[30]),  # 31: g1 first period grade: 0-20
        float(row[31]),  # 32: g2 second period grade: 0-20
        float(row[32]),  # 33: g3 final grade: 0-20
    ]


input_data = numpy.array(list(map(prep_input_data, data)))
training_data = numpy.array(list(map(prep_output_data, data)))

def add_bias(X):
    # add bias to X x0 dim(n x 1)
    x0 = numpy.ones((X.shape[0], 1))  # creates n X 1 matrix with ones
    # adds 1 as first column to matrix
    return numpy.hstack((x0, X))  # X has now dim(N x d+1)


def gradient_descent(X, T, hidden_neurons, eta=0.01, num_epochs=1000):
    num_samples = X.shape[0]
    input_dim = X.shape[1]
    output_dim = T.shape[1]
    losses = numpy.array([])

    assert num_samples == T.shape[0], "Number of Input and Training samples must match"

    # initialize weights
    # W1 dim(k+1, D+1)
    W1 = numpy.random.rand(hidden_neurons + 1, input_dim) * 2. - 1.  # between [-1.0, 1.0)

    # W2 dim(O x K+1)
    W2 = numpy.random.rand(output_dim, hidden_neurons+1)
    epoch = 0
    while epoch < num_epochs:
        # transpose X and T so they match DIM of formulas used in lecture
        Y, W1, W2, loss = descent(X.T, T.T, W1, W2, eta)
        losses = numpy.append(losses, loss)
        print(f"Epoch: {epoch}, Loss: {loss}")
        epoch += 1

    return losses


def stochastic_gradient_descent(X, T, hidden_neurons, eta=0.01, num_epochs=1000, batch_size=24, mu=None):

    num_samples = X.shape[0]
    input_dim = X.shape[1]
    output_dim = T.shape[1]
    losses = numpy.array([])

    assert num_samples == T.shape[0], "Number of Input and Training samples must match"

    # initialize weights
    # W1 dim(k+1, D+1)
    W1 = numpy.random.rand(hidden_neurons + 1, input_dim) * 2. - 1.  # between [-1.0, 1.0)

    # W2 dim(O x K+1)
    W2 = numpy.random.rand(output_dim, hidden_neurons+1)
    old_epoch = 1
    for x, t, epoch in batch(X, T, batch_size, num_epochs):
        # transpose X and T so they match DIM of formulas used in lecture
        Y, W1, W2, loss = descent(x.T, t.T, W1, W2, eta, mu)
        if epoch != old_epoch:
            old_epoch = epoch
            losses = numpy.append(losses, loss)
            print(f"Epoch: {epoch}, Loss: {loss}")

    return losses


input_data = add_bias(input_data)
gd_losses = gradient_descent(input_data, training_data, 500, 0.005, 100000)
sgd_losses = stochastic_gradient_descent(input_data, training_data, 500, 0.005, 100000, batch_size=64)
mom_losses = stochastic_gradient_descent(input_data, training_data, 500, 0.005, 100000, batch_size=64, mu=0.99)

pyplot.loglog(sgd_losses, label="Stochastic Gradient Descent Loss")
pyplot.loglog(mom_losses, label="Stochastic Gradient Descent Loss + Momentum")
pyplot.loglog(gd_losses, label="Gradient Descent Loss")

pyplot.xlabel("epochs")
pyplot.ylabel("loss")

pyplot.legend()
pyplot.show()
