import os
import cv2
import numpy as np
import seaborn as sn
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize and reshape for Conv2D
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

model_path = "t3_2d_model.h5"

# Build and Train model
if not os.path.exists(model_path):
    model = keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax'),
    ])
    model.summary()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_train_cnn, y_train,
        epochs=30,
        validation_split=0.2,
        batch_size=64,
        callbacks=[early_stop]
    )
    model.save(model_path)
    print(f"Model saved as {model_path}")

    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    y_predicted = model.predict(X_test_cnn)
    y_predicted_labels = [np.argmax(i) for i in y_predicted]

    # Confusion Matrix
    cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
    cm = cm.numpy()
    plt.figure(figsize=(10, 7))
    sn.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot graphs
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

else:
    print("Model already exists. Skipping training.")

# Predict uploaded image
def predict_digit_from_image(image_path):
    if not os.path.exists(model_path):
        print("Trained model not found.")
        return

    model = load_model(model_path)

    # Read and preprocess image for CNN
    img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        print("Could not open image with OpenCV.")
        return

    img_cv = cv2.resize(img_cv, (28, 28), interpolation=cv2.INTER_AREA)
    img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)
    img_cv = cv2.adaptiveThreshold(
        img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    img_array = img_cv.astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    plt.imshow(img_cv, cmap='gray')
    plt.title(f"Predicted Digit: {predicted_class}")
    plt.axis('off')
    plt.show()

    return predicted_class

# select image
def main():
    file_path = input("Enter the path to the digit image (png/jpg/jpeg): ")
    if file_path and os.path.exists(file_path):
        print(f"Selected file: {file_path}")
        img_cv = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img_cv is not None:
            cv2.imshow("Selected Digit Image", img_cv)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        else:
            print("Could not open image with OpenCV.")
        predict_digit_from_image(file_path)
    else:
        print("No file selected or file does not exist.")

# Run
main()
