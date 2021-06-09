import argparse
import csv
import os
from random import random
import numpy
from matplotlib import pyplot


def input_data(types=["mat"]):
    inputs = []
    targets = []
    yn = {"yes":1.,"no":-1.}
    for t in types:
        reader = csv.reader(open(os.path.join(F"data/student-{t}.csv"), 'r'), delimiter=";")
        # skip first line
        next(reader)
        for splits in reader:
            # read input values
            inputs.append([
                1.,                                     # bias
                {"GP": -1., "MS": 1.}[splits[0]],       # school
                {"M": -1., "F": 1.}[splits[1]],         # gender
                float(splits[2]),                       # age
                {"U": -1., "R": 1.}[splits[3]],         # address
                {"LE3": -1., "GT3": 1.}[splits[4]],     # family size
                {"T": -1., "A": 1.}[splits[5]],         # parents living together
                float(splits[6]),                       # mother education
                float(splits[7]),                       # father education
                # skip categorical values
                float(splits[12]),                      # travel time
                float(splits[13]),                      # study time
                float(splits[14]),                      # failures
                yn[splits[15]],                         # extra support
                yn[splits[16]],                         # family support
                yn[splits[17]],                         # paid support
                yn[splits[18]],                         # activities
                yn[splits[19]],                         # nursery school
                yn[splits[20]],                         # higher education
                yn[splits[21]],                         # internet
                yn[splits[22]],                         # romantic
                float(splits[23]),                      # family relation
                float(splits[24]),                      # free time
                float(splits[25]),                      # going out
                float(splits[26]),                      # workday alcohol
                float(splits[27]),                      # weekend alcohol
                float(splits[28]),                      # health
                float(splits[29])                      # absences
            ])

            # read targets values
            targets.append([
                float(splits[30]),  # grade for primary school
                float(splits[31]),  # grade for secondary school
                float(splits[32]),  # grade for tertiary school
            ])

        print(F"Loaded dataset with {len(targets)} samples")
        return numpy.array(inputs), numpy.array(targets)


def logistic(A):
    return 1./(1.+numpy.exp(-A))


def forward(X, W1, W2):
    # compute activation
    H = logistic(numpy.dot(W1,X))
    H[0, :] = 1  # bias

    # compute output
    Y = numpy.dot(W2, H)

    # return both
    return Y, H

def loss(X, T, W1, W2):
    # compute output of network
    Y, H = forward(X, W1, W2)

    # compute loss
    # numpy mean will also implement the frobenius norm
    J = numpy.mean((Y-T)**2)

    return J, Y, H

def gradient(X, T, Y, H, W1, W2):
    # first layer gradient
    G1 = 2./len(X) * (numpy.dot(numpy.dot(W2.T, (Y-T)) * H * (1.-H), X.T))

    # second layer gradient
    G2 = 2./len(X) * (numpy.dot((Y-T), H.T))

    return G1, G2

old_update = None

def descent(X, T, W1, W2, eta, mu=None):
    # compute loss
    J, Y, H = loss(X, T, W1, W2)

    # compute gradient
    G1, G2 = gradient(X, T, Y, H, W1, W2)

    #update weights in place
    W1 -= eta * G1
    W2 -= eta * G2

    if mu is not None:
        global old_update
        if old_update is not None:
            # momentum learning
            W1 += mu * old_update[0]
            W2 += mu * old_update[1]
        old_update = [-eta * G1, -eta * G2]

    return J

def gradient_descent(X, T, W1, W2, eta=0.001, epochs=100000):
    print(F"Performing Gradient Descent for {epochs} epochs")
    losses = []
    for epoch in range(epochs):
        # perform one gradient descent step for full dataset
        J = descent(X.T, T.T, W1, W2, eta)
        losses.append(J)
        # will overwrite the existing line on the cmd line
        print("\repoch: ", epoch+1, "- loss:", J, end="", flush=True)
    print()
    evaluate_all(X, W1, W2)
    return losses

def batch(X, T, B, epochs):
    # get indexes list of all samples
    indexes = list(range(X.shape[0])) # 0..number of samples
    # start with empty batch
    batch = []
    beginning_of_epoch = True
    for epoch in range(epochs):
        # shuffle index before each epoch
        random.shuffle(indexes)
        # iterate over random samples
        for index in indexes:
            # append batch index
            batch.append(index)
            if len(batch) == B:
                # batch is full, yield the samples
                # will return the samples specified by the indices in batch
                yield X[batch], T[batch], beginning_of_epoch
                batch.clear()
                beginning_of_epoch = False
        beginning_of_epoch = True
    # yield the last batch if not empty
    if batch:
        yield X[batch], T[batch], True

def stochastic_gradient_descent(X, T, W1, W2, batch_size=64, eta=0.001,mu=None, epochs=100000):
    print(F"Performing Stochastic Gradient Descent for {epochs} epochs")
    losses = []
    # iterate over batches frawn from the dataset
    # enumerate will return an increasing index -> iteration
    for iteration, (x, t, e) in enumerate(batch(X, T, batch_size, epochs)):
        # perform one gradient step for the current batch
        J = descent(x.T, t.T, W1, W2, eta, mu)
        if e:
            losses.append(J)
        print("\riteration: ", iteration+1, "- loss:", J, end="", flush=True)
    print()
    evaluate_all(X,W1, W2)
    return losses

def evaluate(X, W1, W2, index, name, values=[-1, 1]):
    print(F"Evaluating {name}")
    # correct index; 4 since we skipped 4 categorical values
    if index > 8:
        index -= 4
    # check several different samples
    for value in values:
        # select all input samples with these values
        # will return a boolean array with the indices of all entries in X
        # that match
        matching_values = X[:, index] == value
        x = X[matching_values]
        # forward samples
        y, _ = forward(x.T, W1, W2)
        # compute average output
        mean = numpy.mean(y, axis=1)
        print(F"Average grades for {value} are {mean}")
    print()

def evaluate_all(X, W1, W2):
    # evaluate the influence of four different variables
    evaluate(X, W1, W2, 2, "gender")
    evaluate(X, W1, W2, 4, "address")
    evaluate(X, W1, W2, 14, "studytime", range(1, 5))
    evaluate(X, W1, W2, 18, "paid classes")
    evaluate(X, W1, W2, 22, "internet")
    evaluate(X, W1, W2, 23, "romantic relationship")
    evaluate(X, W1, W2, 25, "free time", range(1, 6))
    evaluate(X, W1, W2, 27, "workday alcohol", range(1, 6))
    evaluate(X, W1, W2, 28, "weekend alcohol", range(1.6))

def command_line_options():
    # create command line parser object
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set some options with default values
    parser.add_argument("-t", "--data-sets", choices=("mat", "por"), nargs="+", default=["mat"], help="Select the datasets to work on")
    parser.add_argument("-K", "--hidden", type=int, default=100, help="Select the number of hidden units")
    parser.add_argument("-e", "--epochs", type=int, default=10000, help="Select the number of epochs for GD")
    parser.add_argument("-B", "--batch-size", type=int, default=32, help="Select the batch size for SGD")
    parser.add_argument("-m", "--momentum", type=float, default=0.999, help="Select the momentum term")
    parser.add_argument("-l", "--learning-rate", type=float, default=1e-4, help="Set the learning rate")
    parser.add_argument("-s", "--seed", type=int, help="If selected, the given random seed is used")
    parser.add_argument("-o", "--plot", default="Losses.pdf", help="Select the filename for the plots")

    return parser.parse_args()


if __name__ == '__main__':
    # get command line arguments
    args = command_line_options()

    # read data
    X, T = input_data(args.data_sets)

    # define number of hidden units
    K = args.hidden

    # initialize weights randomly
    if args.seed is not None:
        numpy.random.seed(args.seed)

    W1 = numpy.random.random((K+1, X.shape[1])) * 2. - 1.
    W2 = numpy.random.random((T.shape[1], K+1)) * 2. - 1.

    # perform gradient descent
    gd = gradient_descent(X, T, W1.copy(), W2.copy(), args.learn_rate, args.epochs)
    sgd = stochastic_gradient_descent(X, T, W1.copy(), W2.copy(), args.batch_size, args.learn_rate, None, args.epochs)
    mom = stochastic_gradient_descent(X, T, W1.copy(), W2.copy(), args.batch_size, args.learn_rate, args.momentum, args.epochs)

    # plot both together in log scale
    pyplot.loglog(sgd, label="Stochastic Gradient Descent")
    pyplot.loglog(mom, label="Stochastic Gradient Descent + Momentum")
    pyplot.loglog(gd, label="Gradient Descent")
    pyplot.xlabel("Epochs")
    pyplot.ylabel("Loss")
    pyplot.legend()
    pyplot.savefig(args.plot)
