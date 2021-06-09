# 1. Create 1D noisy linear training data samples
import numpy

a = 1.5
b = 2

data = numpy.array([])

# applies the linear training function
def training(x):
    return a * x + b + numpy.random.random()

# generates the training data samples
for i in range(100):
    numpy.append(data, [i+1, training(i+1)])

# mean squared loss
def compute_loss(data):
    sum = 0
    for d in data:
        y = d[0]
        t = d[1]
        sum += numpy.square(y-t)
    return sum/len(data)

# linear unit
def linear_unit(x,w):
    y = w * x
    return y

def compute_gradient(x,w):
    sum_0 = 0
    sum_1 = 0
    for xn in x:
        x = xn[0]
        t = xn[1]
        w0 = w[0]
        w1 = w[1]
        sum_0 += (w0 + w1 * x - t)*


print(data)