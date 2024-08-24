import numpy as np
import matplotlib.pyplot as plt


# Define activation functions
def linear(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def tanh(x):
    return np.tanh(x)


def bipolar_sigmoid(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))


def softplus(x):
    return np.log(1 + np.exp(x))


# Generate x values
x = np.linspace(-10, 10, 400)

# Create the plot
plt.figure(figsize=(15, 15))

# Plot Linear Activation Function
plt.subplot(4, 2, 1)
plt.plot(x, linear(x))
plt.title('Linear Activation Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

# Plot Sigmoid Activation Function
plt.subplot(4, 2, 2)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid Activation Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

# Plot ReLU Activation Function
plt.subplot(4, 2, 3)
plt.plot(x, relu(x))
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

# Plot Leaky ReLU Activation Function
plt.subplot(4, 2, 4)
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

# Plot Hyperbolic Tangent (tanh) Activation Function
plt.subplot(4, 2, 5)
plt.plot(x, tanh(x))
plt.title('Hyperbolic Tangent (tanh) Activation Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

# Plot Bipolar Sigmoid Activation Function
plt.subplot(4, 2, 6)
plt.plot(x, bipolar_sigmoid(x))
plt.title('Bipolar Sigmoid Activation Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

# Plot Softplus Activation Function
plt.subplot(4, 2, 7)
plt.plot(x, softplus(x))
plt.title('Softplus Activation Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.show()
