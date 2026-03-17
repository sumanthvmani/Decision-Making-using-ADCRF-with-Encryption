import tensorflow as tf
from keras.src.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from Classificaltion_Evaluation import ClassificationEvaluation


# https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058030/1806228006848429/latest.html#:~:text=In%20keras%2C%20it%20is%20easy,o%20the%20sequence%20of%20calling


def Model_DNN(train_data, train_target, test_data, test_target, HN=None, SPE=None, EP=None):
    if SPE is None:
        SPE = 5
    if EP is None:
        EP = 10
    if HN is None:
        HN = 128

    input_shape = (784,)
    num_classes = train_target.shape[-1]

    SIZE = input_shape[0]
    Train_X = np.zeros((train_data.shape[0], SIZE))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (SIZE))
        Train_X[i] = np.reshape(temp, (SIZE))

    Test_X = np.zeros((test_data.shape[0], SIZE))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (SIZE))
        Test_X[i] = np.reshape(temp, (SIZE))

    model = Sequential([
        Dense(256, activation='relu', input_shape=input_shape),  # First dense layer
        Dropout(0.4),  # Dropout layer with rate 0.4
        Dense(128, activation='relu'),  # Second dense layer
        Dropout(0.3),  # Dropout layer with rate 0.3
        Dense(HN, activation='relu'),  # Third dense layer
        Dropout(0.3),  # Dropout layer with rate 0.3
        Dense(num_classes, activation='softmax')  # Output layer with 10 classes
    ])
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])  # rmsprop
    model.fit(Train_X, train_target, epochs=EP, batch_size=32, steps_per_epoch=SPE, validation_data=(Test_X, test_target))
    pred = model.predict(Test_X)
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = ClassificationEvaluation(test_target, pred)
    return Eval, pred