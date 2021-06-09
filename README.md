uzh-deep-learning

# Python Cheatsheet

#### Get Length of Array
```python
x = len(cars)
```

#### Add Bias 1s to Input
````python
import numpy
x = numpy.random.uniform(size=(10,3))
n,m = x.shape # for generality
x0 = numpy.ones((n,1))
x_new = numpy.hstack((x0,x))
````

### Numpy Axis
Just like coordinate systems, NumPy arrays also have axes. In a 2-dimensional NumPy array, the axes are the directions along the rows and columns.
In a NumPy array, axis 0 is the “first” axis. Assuming that we’re talking about multi-dimensional arrays, axis 0 is the axis that runs downward down the rows.