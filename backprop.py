import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output = False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural Network Parameters
# Neural Network Parameters
input_size = X_train.shape[1]
output_size = y_train.shape[1]
hidden_size = 8
learning_rate = 0.01
epochs = 1000

# Initialize random weights
np.random.seed(42)
weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error Loss
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Training the Neural Network
mse_history = []
accuracy_history = []

for epoch in range(epochs):
    # Forward Propagation
    hidden_input = np.dot(X_train, weights_input_hidden)
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output)
    final_output = sigmoid(final_input)

    # Calculate error
    error = mean_squared_error(y_train, final_output)
    mse_history.append(error)

    # Calculate accuracy
    predicted_classes = np.argmax(final_output, axis=1)
    true_classes = np.argmax(y_train, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    accuracy_history.append(accuracy)
    
    # Backpropagation
    output_error = y_train - final_output
    output_delta = output_error * sigmoid_derivative(final_output)

    hidden_layer_error = output_delta.dot(weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_output)

    # Update weights
    weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += X_train.T.dot(hidden_layer_delta) * learning_rate

# Plot Mean Squared Error
plt.plot(mse_history)
plt.title('Mean Squared Error over Iterations')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.show()

# Testing the Neural Network
hidden_input_test = np.dot(X_test, weights_input_hidden)
hidden_output_test = sigmoid(hidden_input_test)
final_input_test = np.dot(hidden_output_test, weights_hidden_output)
final_output_test = sigmoid(final_input_test)

# Convert probabilities to classes
predicted_classes = np.argmax(final_output_test, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_classes)
print(f"Accuracy on Test Data: {accuracy}")
plt.plot(accuracy_history)
plt.title('Accuracy over Iterations')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
