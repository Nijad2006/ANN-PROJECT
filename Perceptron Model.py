import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Dataset (AND gate)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

# Model
model = Perceptron(max_iter=1000, eta0=0.1)
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y, y_pred))