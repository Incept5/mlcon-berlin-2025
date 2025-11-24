
"""
Intelligent Character Recognition (ICR) Demo using TensorFlow and MNIST Dataset

This script demonstrates a basic neural network for handwritten digit recognition.
It uses the MNIST dataset, which contains 70,000 grayscale images of handwritten 
digits (0-9), each 28x28 pixels in size.

The neural network architecture:
- Input layer: 28x28 pixel images (flattened to 784 features)
- Hidden layer: 128 neurons with ReLU activation
- Dropout layer: 20% dropout rate for regularization
- Output layer: 10 neurons (one per digit class 0-9)
"""

# Step 1: Import required libraries
import tensorflow as tf  # Deep learning framework for building and training neural networks
import numpy as np  # Numerical computing library for array operations
import matplotlib.pyplot as plt  # Plotting library for visualizing images and results

# Step 2: Load and preprocess the MNIST dataset
# MNIST (Modified National Institute of Standards and Technology) dataset contains:
# - Training set: 60,000 images
# - Test set: 10,000 images
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values from [0, 255] to [0, 1] range
# This normalization helps the neural network train more efficiently by:
# - Keeping gradients in a reasonable range
# - Preventing any single feature from dominating the learning process
# - Improving convergence speed and model stability
x_train, x_test = x_train / 255.0, x_test / 255.0

# Step 3: Define the neural network model
# Sequential model: layers are stacked linearly, output of one feeds into the next
model = tf.keras.models.Sequential([
    # Input layer: explicitly defines the shape of input data (28x28 pixel images)
    tf.keras.layers.Input(shape=(28, 28)),
    
    # Flatten layer: converts 2D image (28x28) into 1D vector (784 features)
    # This is necessary because Dense layers expect 1D input
    # Transformation: (28, 28) -> (784,)
    tf.keras.layers.Flatten(),
    
    # First Dense (fully connected) layer: 128 neurons with ReLU activation
    # - Each neuron connects to all 784 input features
    # - ReLU (Rectified Linear Unit) activation: f(x) = max(0, x)
    # - ReLU introduces non-linearity, allowing the network to learn complex patterns
    # - Common choice for hidden layers due to computational efficiency
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Dropout layer: randomly sets 20% of input units to 0 during training
    # Purpose: regularization technique to prevent overfitting by:
    # - Forcing the network to learn redundant representations
    # - Preventing co-adaptation of neurons (neurons becoming too dependent on each other)
    # - Creating an ensemble effect (different neurons active each iteration)
    # Note: Dropout is only active during training, not during inference
    tf.keras.layers.Dropout(0.2),
    
    # Output layer: 10 neurons (one per digit class: 0-9)
    # - No activation function here (raw logits output)
    # - Logits will be converted to probabilities later using softmax
    # - Each neuron's output represents the model's confidence for that digit
    tf.keras.layers.Dense(10)
])

# Step 4: Compile and Train the model
# Compilation configures the learning process with three key components:

# Optimizer: 'adam' (Adaptive Moment Estimation)
# - Combines benefits of AdaGrad and RMSProp
# - Adapts learning rate for each parameter automatically
# - Generally performs well without extensive hyperparameter tuning
# - Uses momentum to accelerate convergence

# Loss function: SparseCategoricalCrossentropy
# - Measures difference between predicted probabilities and true labels
# - 'Sparse' means labels are integers (0-9), not one-hot encoded vectors
# - 'from_logits=True' means the model outputs raw scores, not probabilities
# - The loss function will apply softmax internally for numerical stability
# - Cross-entropy penalizes confident wrong predictions more heavily

# Metrics: 'accuracy' tracks the percentage of correct predictions
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model on the training data
# - x_train: input images (60,000 samples)
# - y_train: true labels (60,000 labels)
# - epochs=5: complete passes through the entire training dataset
# Each epoch:
# 1. Forward pass: compute predictions for all training samples
# 2. Calculate loss: measure prediction error
# 3. Backward pass: compute gradients using backpropagation
# 4. Update weights: adjust model parameters to minimize loss
model.fit(x_train, y_train, epochs=5)

# Step 5: Evaluate the model
# Test the model on unseen data (test set) to assess generalization
# - x_test: 10,000 test images the model has never seen
# - y_test: true labels for test images
# - verbose=2: prints one line per epoch
# Returns: test_loss (average loss) and test_acc (accuracy percentage)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("\nTest accuracy:", test_acc)

# Step 6: Make predictions and visualize results
# Create a new model that includes softmax activation
# - Wraps the trained model with a Softmax layer
# - Converts raw logits to probabilities (values sum to 1.0)
# - Each output represents P(digit = i) for i in [0-9]
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# Make predictions on the first test image
# predictions shape: (1, 10) - one probability distribution over 10 classes
predictions = probability_model.predict(x_test[:1])

# Visualize the test image and predicted label
# - cmap='gray': display image in grayscale
# - np.argmax: find the index of highest probability (predicted digit)
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Predicted label: {np.argmax(predictions[0])}")
plt.show()
