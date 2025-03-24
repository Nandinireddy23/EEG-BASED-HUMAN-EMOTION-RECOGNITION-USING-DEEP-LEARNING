# class InitializeTraining:
#     def start_process(self):
#         print('Training Started')
#         import numpy as np
#         import argparse
#         import matplotlib.pyplot as plt
#         import cv2
#         from tensorflow.keras.models import Sequential
#         from tensorflow.keras.layers import Dense, Dropout, Flatten
#         from tensorflow.keras.layers import Conv2D
#         from tensorflow.keras.optimizers import Adam
#         from tensorflow.keras.layers import MaxPooling2D
#         from tensorflow.keras.preprocessing.image import ImageDataGenerator
#         from django.conf import settings
#         import os
#         os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#         # plots accuracy and loss curves
#         def plot_model_history(model_history):
#             """
#             Plot Accuracy and Loss curves given the model_history
#             """
#             fig, axs = plt.subplots(1, 2, figsize=(15, 5))
#             # summarize history for accuracy
#             axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
#             axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
#             axs[0].set_title('Model Accuracy')
#             axs[0].set_ylabel('Accuracy')
#             axs[0].set_xlabel('Epoch')
#             axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1),
#                               len(model_history.history['accuracy']) / 10)
#             axs[0].legend(['train', 'val'], loc='best')
#             # summarize history for loss
#             axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
#             axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
#             axs[1].set_title('Model Loss')
#             axs[1].set_ylabel('Loss')
#             axs[1].set_xlabel('Epoch')
#             axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1),
#                               len(model_history.history['loss']) / 10)
#             axs[1].legend(['train', 'val'], loc='best')
#             fig.savefig(os.path.join(settings.MEDIA_ROOT,'plot.png'))
#             plt.show()

#         # Define data generators
#         train_dir = os.path.join(settings.MEDIA_ROOT, 'data','train')
#         val_dir = os.path.join(settings.MEDIA_ROOT, 'data','test')

#         num_train = 28709
#         num_val = 7178
#         batch_size = 64
#         num_epoch = 50

#         train_datagen = ImageDataGenerator(rescale=1. / 255)
#         val_datagen = ImageDataGenerator(rescale=1. / 255)

#         train_generator = train_datagen.flow_from_directory(
#             train_dir,
#             target_size=(48, 48),
#             batch_size=batch_size,
#             color_mode="grayscale",
#             class_mode='categorical')

#         validation_generator = val_datagen.flow_from_directory(
#             val_dir,
#             target_size=(48, 48),
#             batch_size=batch_size,
#             color_mode="grayscale",
#             class_mode='categorical')

#         # Create the model
#         model = Sequential()

#         model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
#         model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.25))

#         model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.25))

#         model.add(Flatten())
#         model.add(Dense(1024, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(7, activation='softmax'))
#         model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
#         model_info = model.fit_generator(
#             train_generator,
#             steps_per_epoch=num_train // batch_size,
#             epochs=num_epoch,
#             validation_data=validation_generator,
#             validation_steps=num_val // batch_size)
#         plot_model_history(model_info)
#         model.save_weights('DP_model.h5')


# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import json
# import cv2
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from django.conf import settings

# class InitializeTraining:
#     def start_process(self):
#         print('Training Started')
#         os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#         # Define file paths
#         model_weights_path = os.path.join(settings.MEDIA_ROOT, 'DP_model.h5')
#         results_json_path = os.path.join(settings.MEDIA_ROOT, 'training_result.json')

#         # If results exist, return without training
#         if os.path.exists(results_json_path):
#             print("Training results found. Skipping training.")
#             return

#         # Define data generators
#         train_dir = os.path.join(settings.MEDIA_ROOT, 'data', 'train')
#         val_dir = os.path.join(settings.MEDIA_ROOT, 'data', 'test')

#         num_train = 28709
#         num_val = 7178
#         batch_size = 64
#         num_epoch = 20

#         train_datagen = ImageDataGenerator(rescale=1. / 255)
#         val_datagen = ImageDataGenerator(rescale=1. / 255)

#         train_generator = train_datagen.flow_from_directory(
#             train_dir,
#             target_size=(48, 48),
#             batch_size=batch_size,
#             color_mode="grayscale",
#             class_mode='categorical')

#         validation_generator = val_datagen.flow_from_directory(
#             val_dir,
#             target_size=(48, 48),
#             batch_size=batch_size,
#             color_mode="grayscale",
#             class_mode='categorical')

#         # Create the model
#         model = Sequential([
#             Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
#             Conv2D(64, kernel_size=(3, 3), activation='relu'),
#             MaxPooling2D(pool_size=(2, 2)),
#             Dropout(0.25),

#             Conv2D(128, kernel_size=(3, 3), activation='relu'),
#             MaxPooling2D(pool_size=(2, 2)),
#             Conv2D(128, kernel_size=(3, 3), activation='relu'),
#             MaxPooling2D(pool_size=(2, 2)),
#             Dropout(0.25),

#             Flatten(),
#             Dense(1024, activation='relu'),
#             Dropout(0.5),
#             Dense(7, activation='softmax')
#         ])

#         model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

#         # Load existing weights if available
#         if os.path.exists(model_weights_path):
#             print("Loading existing model weights...")
#             model.load_weights(model_weights_path)

#         # Train the model
#         model_info = model.fit(
#             train_generator,
#             steps_per_epoch=num_train // batch_size,
#             epochs=num_epoch,
#             validation_data=validation_generator,
#             validation_steps=num_val // batch_size)

#         # Save model weights
#         model.save_weights(model_weights_path)

#         # Save the results in a JSON file
#         result = {
#             "accuracy": model_info.history['accuracy'],
#             "val_accuracy": model_info.history['val_accuracy'],
#             "loss": model_info.history['loss'],
#             "val_loss": model_info.history['val_loss']
#         }

#         with open(results_json_path, 'w') as json_file:
#             json.dump(result, json_file)

#         print("Training Completed and Results Saved.")



import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from django.conf import settings

class InitializeTraining:
    def start_process(self):
        print('Training Started')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # Define file paths
        model_weights_path = os.path.join(settings.MEDIA_ROOT, 'DP_model.h5')
        results_json_path = os.path.join(settings.MEDIA_ROOT, 'training_result.json')

        # Load existing results if available
        previous_results = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
        if os.path.exists(results_json_path):
            with open(results_json_path, 'r') as json_file:
                previous_results = json.load(json_file)

        # Define data generators
        train_dir = os.path.join(settings.MEDIA_ROOT, 'data', 'train')
        val_dir = os.path.join(settings.MEDIA_ROOT, 'data', 'test')

        num_train = 28709
        num_val = 7178
        batch_size = 64
        initial_epochs = len(previous_results["accuracy"])  # Detect previously trained epochs
        additional_epochs = 30  # Train for 30 more epochs
        total_epochs = initial_epochs + additional_epochs  # Total epochs after training

        train_datagen = ImageDataGenerator(rescale=1. / 255)
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48, 48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical')

        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48, 48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical')

        # Create the model
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

        # Load existing weights if available
        if os.path.exists(model_weights_path):
            print(f"Loading existing model weights from {model_weights_path}...")
            model.load_weights(model_weights_path)

        print(f"Resuming training from epoch {initial_epochs} to {total_epochs}...")

        # Train for additional epochs
        model_info = model.fit(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=additional_epochs,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)

        # Append new results to previous results
        previous_results["accuracy"].extend(model_info.history['accuracy'])
        previous_results["val_accuracy"].extend(model_info.history['val_accuracy'])
        previous_results["loss"].extend(model_info.history['loss'])
        previous_results["val_loss"].extend(model_info.history['val_loss'])

        # Save model weights
        model.save_weights(model_weights_path)

        # Save updated results in JSON file
        with open(results_json_path, 'w') as json_file:
            json.dump(previous_results, json_file)

        print(f"Training Completed. Model now trained for {total_epochs} epochs.")

