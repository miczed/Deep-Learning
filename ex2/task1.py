import numpy
import scipy.linalg
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages


# function to compute the loss for the given w1 and w2
def compute_loss(w1, w2):
    return w1**2 + w2**2 + 30 * numpy.sin(w1) * numpy.sin(w2)


def compute_gradient(w1, w2):
    # partial derivative w.r.t w1
    gr1 = 2 * w1 + 30 * numpy.cos(w1) * numpy.sin(w2)
    # partial derivative w.r.t w2
    gr2 = 2 * w2 + 30 * numpy.sin(w1) * numpy.cos(w2)
    return numpy.array([gr1, gr2])


def compute_gradient_vec(w):
    return 2 * w + 30 * numpy.cos(w) * numpy.sin(w[::-1])


def gradient_descent(w, eta):
    # stopping criterion I: limit the number of iterations
    for epoch in range(1000):
        # compute gradient for current w
        gradient = compute_gradient_vec(w)
        # stopping criterion II: if norm of gradient is small
        if scipy.linalg.norm(gradient) < 1e-4:  # scientific notation: 0.0001
            break
        # perform one gradient descent step
        w = w - eta * gradient
    # return optimized weight plus epoch
    return w, epoch+1


# surface plot of the loss function
def plot_surface(alpha=.8):
    # define range of data samples
    w = numpy.arange(-10, 10.001, 0.1)
    w1, w2 = numpy.meshgrid(w, w)

    # compute the loss vor all values of w1 and w2
    J = compute_loss(w1, w2) # will be matrix with same size as w1 and w2

    # initialize 3D plot
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection="3d", azim=-40, elev=50)

    # plot surface with jet colormap
    ax.plot_surface(w1, w2, J, cmap="jet", alpha=alpha)
    return fig, ax


# open pdf file
pdf = PdfPages("surface.pdf")


# start 10 trials with different initial weights
for trails in range(10):
    # create random weights in range [-10, 10]
    w = numpy.random.random(2) * 20 - 10
    # perform gradient descent (copy w to keep original value)
    o, epochs = gradient_descent(w.copy(), 0.04)

    # plot surface
    fig, ax = plot_surface(.5)

    # compute z values for initial and optimal weights
    loss_w = compute_loss(w[0], w[1])
    loss_o = compute_loss(o[0], o[1])

    # plot values, connected with a line
    ax.plot([w[0], o[0]], [w[1], o[1]], [loss_w, loss_o], "kx-")
    pdf.savefig(fig)

    # print the number of iterations, the start and the final
    print(epochs, w, o, loss_o)

# finalize and close pdf file
pdf.close()
