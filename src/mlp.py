import tensorflow as tf
import numpy as np
import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Dropout
import keras_tuner as kt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import resample
from collections import Counter
import tensorflow.keras.backend as K


def build_mlp(input_shape):
    try:
        model = Sequential()
        model.add(Input(shape=(input_shape,)))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        
        def focal_loss(gamma=2.0, alpha=0.25):
            def focal_loss_fixed(y_true, y_pred):
                bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
                bce_exp = K.exp(-bce)
                focal = alpha * K.pow((1 - bce_exp), gamma) * bce
                return focal
            return focal_loss_fixed

        model.compile(optimizer=keras.optimizers.Adam(0.00001),
                    loss=focal_loss(gamma=2.0, alpha=0.25),
                    metrics=[
                        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.AUC(name='auc')
                        ]
                    )
        
        model.summary()
    except Exception as e:
        print("Error building model: ", e)

    return model

def build_tuned_mlp(X_train, y_train, X_val, y_val, directory='tuned_models'):
    try:
        def model_builder(hp):
            model = Sequential()
            model.add(Input(shape=(X_train.shape[1],)))

            for i in range(hp.Int("num_layers", 1, 4)):
                model.add(Dense(hp.Int(f"units_{i}", min_value=16, max_value=128, step=32), activation="relu"))
                model.add(Dropout(hp.Float(f"dropout_{i}", 0, 0.15, step=0.05)))

            model.add(Dense(1, activation="sigmoid"))

            model.compile(optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", values=[0.0002, 0.0001, 0.00005])),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
            return model
        
        tuner = kt.RandomSearch(
            model_builder,
            objective='val_accuracy',
            max_trials=30,
            executions_per_trial=4,
            project_name='mlp_tuning',
            directory=directory
        )

        tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)

        return model

    except Exception as e:
        print("Error building model: ", e)
        return None

    