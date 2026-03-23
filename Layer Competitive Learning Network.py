import numpy as np

# Noisy 2D data
X = np.array([
    [0.1, 0.2], [0.2, 0.1],
    [0.8, 0.9], [0.9, 0.8],
    [0.4, 0.5], [0.45, 0.55]
])

# Parameters
neurons = 3
lr = 0.1
epochs = 50

# Initialize weights
weights = np.random.rand(neurons, 2)

# Training
for _ in range(epochs):
    for x in X:
        distances = np.linalg.norm(weights - x, axis=1)
        winner = np.argmin(distances)
        weights[winner] += lr * (x - weights[winner])

# Prediction
labels = []
for x in X:
    distances = np.linalg.norm(weights - x, axis=1)
    labels.append(np.argmin(distances))

print("Cluster Labels:", labels)
print("Final Weights:\n", weights)