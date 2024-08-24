import numpy as np


# Define the activation function (step function for binary classification)
def activation(x):
    return 1 if x >= 0 else 0


# Define the perceptron model
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(input_size + 1)  # Including the bias term

    def predict(self, inputs):
        # Add the bias term
        inputs = np.insert(inputs, 0, 1)
        summation = np.dot(inputs, self.weights)
        return activation(summation)

    def fit(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                inputs = np.insert(X[i], 0, 1)  # Add the bias term
                prediction = self.predict(X[i])
                error = y[i] - prediction
                # Update weights and bias
                self.weights += self.learning_rate * error * inputs

    def evaluate(self, X, y):
        predictions = np.array([self.predict(x) for x in X])
        accuracy = np.mean(predictions == y)
        return accuracy


# Example data: hours of study and sleep per day, and whether the student passed (1) or failed (0)
X = np.array([
    [5, 7],  # Study 5 hours, Sleep 7 hours
    [3, 6],  # Study 3 hours, Sleep 6 hours
    [8, 5],  # Study 8 hours, Sleep 5 hours
    [2, 8],  # Study 2 hours, Sleep 8 hours
    [6, 6],  # Study 6 hours, Sleep 6 hours
])

y = np.array([1, 0, 1, 0, 1])  # 1 = Pass, 0 = Fail

# Create and train the perceptron
model = Perceptron(input_size=2)  # Two input features: study hours and sleep hours
model.fit(X, y)

# Test the model and calculate accuracy
accuracy = model.evaluate(X, y)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Test the model with new data
test_data = np.array([
    [4, 7],  # Study 4 hours, Sleep 7 hours
    [7, 5],  # Study 7 hours, Sleep 5 hours
])

for data in test_data:
    print(f"Input: {data}, Prediction: {model.predict(data)}")






