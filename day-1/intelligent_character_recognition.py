# Step 1: Import required libraries
# TensorFlow/Keras for building and training the neural network
import tensorflow as tf
# NumPy for numerical operations and array manipulation
import numpy as np
# Matplotlib for visualizing images and predictions
import matplotlib.pyplot as plt

# Step 2: Load and preprocess the MNIST dataset
# MNIST contains 70,000 images of handwritten digits (0-9)
mnist = tf.keras.datasets.mnist
# Load dataset: 60,000 training images and 10,000 test images
# x_train/x_test: grayscale images (28x28 pixels), y_train/y_test: digit labels (0-9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize pixel values from [0, 255] to [0, 1] for better neural network training
x_train, x_test = x_train / 255.0, x_test / 255.0

# Step 3: Define the neural network model
# Sequential model: layers are stacked linearly, one after another
model = tf.keras.models.Sequential([
    # Input layer: expects 28x28 pixel images
    tf.keras.layers.Input(shape=(28, 28)),
    # Flatten layer: converts 2D image (28x28) into 1D array of 784 pixels
    tf.keras.layers.Flatten(),
    # Hidden layer: 128 neurons with ReLU activation (introduces non-linearity)
    tf.keras.layers.Dense(128, activation='relu'),
    # Dropout layer: randomly drops 20% of neurons during training to prevent overfitting
    tf.keras.layers.Dropout(0.2),
    # Output layer: 10 neurons (one for each digit 0-9), produces raw logits (not probabilities)
    tf.keras.layers.Dense(10)
])

# Step 4: Compile and Train the model
# Configure the model with optimizer, loss function, and metrics
model.compile(
    # Adam optimizer: adaptive learning rate optimization algorithm
    optimizer='adam',
    # Sparse Categorical Crossentropy: loss function for multi-class classification
    # from_logits=True: expects raw output from last layer (not probabilities)
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # Track accuracy during training
    metrics=['accuracy']
)
# Train the model for 5 epochs (5 complete passes through the training data)
model.fit(x_train, y_train, epochs=5)

# Step 5: Evaluate the model
# Test the trained model on unseen data to measure its real-world performance
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("\nTest accuracy:", test_acc)

# Step 6: Display a sample image and its prediction
# Create a new model that adds Softmax layer to convert logits to probabilities
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# Predict on the first test image (returns probability distribution over 10 classes)
predictions = probability_model.predict(x_test[:1])
# Display the test image as grayscale
plt.imshow(x_test[0], cmap='gray')
# Show the predicted digit (highest probability class)
plt.title(f"Predicted label: {np.argmax(predictions[0])}")
plt.show()