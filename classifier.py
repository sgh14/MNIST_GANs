from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


def get_classifier(input_shape):
    classifier = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    return classifier


def train_classifier(classifier, dataset, batch_size=64, plot=True):
    validation_split = 0.2
    dataset = dataset.batch(batch_size)
    dataset_size = dataset.cardinality().numpy()
    validation_dataset = dataset.take(int(validation_split*dataset_size)) 
    training_dataset = dataset.skip(int(validation_split*dataset_size))

    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    history = classifier.fit(training_dataset, epochs=5, validation_data=validation_dataset, verbose=1)
    if plot:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

    return classifier