import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from django.conf import settings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def buildDeapModel():
    json_path = os.path.join(settings.MEDIA_ROOT, 'deap_results.json')
    plot_path_acc = os.path.join(settings.MEDIA_ROOT, 'accuracy_plot.png')
    plot_path_loss = os.path.join(settings.MEDIA_ROOT, 'loss_plot.png')

    # If the JSON file exists, load results and return
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            history = json.load(f)
        print("Loaded training history from JSON.")
        return history

    # Load dataset
    deap = os.path.join(settings.MEDIA_ROOT, 'deapData.csv')
    x = pd.read_csv(deap, low_memory=False)

    y = x.values[:, 0]
    pixels = x.values[:, 1]

    # Convert pixel strings into numpy arrays
    X = np.zeros((len(pixels), 48 * 48))
    for ix in range(len(pixels)):
        X[ix] = np.array(pixels[ix].split(' '), dtype=np.float32)

    # Normalize pixel values
    X = X / 255  
    X_train, X_test = X[:30000], X[30000:32300]
    Y_train, Y_test = y[:30000], y[30000:32300]

    X_train = X_train.reshape((X_train.shape[0], 48, 48, 1))
    X_test = X_test.reshape((X_test.shape[0], 48, 48, 1))
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1)
    datagen.fit(X_train)

    # Model definition
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.22),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['acc'])

    # Train model
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=2, batch_size=64, verbose=2)

    # Convert history to dictionary and save as JSON
    history_dict = {key: [float(i) for i in val] for key, val in history.history.items()}
    with open(json_path, 'w') as f:
        json.dump(history_dict, f)

    print("Training complete. Results saved to JSON.")

    # Generate plots
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['acc'], label='Train Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(plot_path_acc)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plot_path_loss)
    plt.close()

    print("Plots saved as images.")
    return history_dict
