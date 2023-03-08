import tensorflow as tf
from matplotlib import pyplot as plt


def train_model(x_train, x_test, y_train, y_test, plot=False):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, input_dim=500, kernel_initializer='normal', activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

    history = model.fit(x_train, y_train, epochs=500, batch_size=16384, verbose=1, validation_data=(x_test, y_test))

    if plot:
        # Show Model's characteristics'
        print("Characteristics of the model:")
        print(model.summary())

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("Model's Training & Validation loss across epochs")
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()

    return model
