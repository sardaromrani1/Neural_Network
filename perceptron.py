# A simple implementation of the perceptron algorithm in Python

import numpy as np

class Perceptron:
    def __init__(self, num_features):
        self.weights = np.zeros(num_features + 1)
    
    def predict(self, features):
        # Add bias term to features
        inputs = np.append(features, 1)
        # Calculate the dot product of weights and inputs
        activation = np.dot(self.weights, inputs)
        # Apply step function
        return 1 if activation >=0 else 0
    
    def train(self, features, target, learning_rate=0.1, epochs=100):
        for _ in range(epochs):
            for x, y in zip(features, target):
                prediction = self.predict(x)
                # Update weights based on prediction error
                self.weights += learning_rate * (y - prediction) * np.append(x, 1)

# Example usage
# Define training data (features and targets)
features = np.array([[0, 0], [0, 1], [1,0], [1, 1]])
targets = np.array([0, 0, 0, 1])

# Create a perceptron with 2 input features
perceptron = Perceptron(num_features=2)

# Train the perceptron
perceptron.train(features, targets)

# Test the perceptron with some example inputs
print(perceptron.predict([0, 0]))   # Output: 0
print(perceptron.predict([0, 1]))   # Output: 0
print(perceptron.predict([1, 0]))   # Output: 0
print(perceptron.predict([1, 1]))   # Output: 1