import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

# Model
model = MLPClassifier(hidden_layer_sizes=(4,),
                      activation='relu',
                      max_iter=1000)

model.fit(X, y)

# Prediction
y_pred = model.predict(X)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y, y_pred))