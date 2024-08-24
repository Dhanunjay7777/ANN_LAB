import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define input and output data
x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([[0], [0], [0], [0], [0], [0], [0], [1]])

# Initialize the model
model = models.Sequential()
model.add(Dense(1, input_dim=3, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x, y, epochs=100, verbose=0)

# Test the model
print("Testing the trained model on training data:")
for i in x:
    prediction = model.predict(np.array([i]))
    print(f"Input: {i} - Predicted Output: {int(round(prediction.flatten()[0]))}")
