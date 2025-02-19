import tensorflow as tf
import numpy as np
import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Dropout
import keras_tuner as kt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import resample
from collections import Counter


def build_mlp(input_shape):
    try:
        model = Sequential()
        model.add(Dense(128, activation="relu", input_shape=input_shape))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
        model.summary()
    except Exception as e:
        print("Error building model: ", e)

    return model


def build_tuned_mlp(X, y, num_folds=5, directory='tuned_models', groups=None):
    try:
        def model_builder(hp):
            model = Sequential()
            model.add(Input(shape=(X.shape[1],)))
            model.add(Dense(hp.Int("units_1", min_value=32, max_value=256, step=32), activation="relu"))
            model.add(Dense(hp.Int("units_2", min_value=32, max_value=256, step=32), activation="relu"))
            model.add(Dense(hp.Int("units_3", min_value=32, max_value=256, step=32), activation="relu"))
            model.add(Dense(hp.Int("units_4", min_value=32, max_value=256, step=32), activation="relu"))
            model.add(Dropout(hp.Float("dropout", 0, 0.5, step=0.1)))
            model.add(Dense(1, activation="sigmoid"))

            model.compile(optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", values=[0.001, 0.0001, 0.01])),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
            return model
        
        tuner = kt.RandomSearch(
            model_builder,
            objective='val_accuracy',
            max_trials=10,
            executions_per_trial=4,
            project_name='test',
            directory=directory
        )

        sgkf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
        best_hps = None
        best_avg_val_accuracy = 0  # Track the best hyperparameters based on avg validation accuracy

        for trial in range(tuner.oracle.max_trials):  # Loop over different hyperparameter trials
            fold_accuracies = []
            
            tuner.search(X, y, epochs=10, validation_split=0.2, verbose=1)
            # Generate a fresh set of hyperparameters for this trial
            trial_hps = kt.HyperParameters()
            model_builder(trial_hps)  # Define model with new hyperparameters

            for train_index, val_index in sgkf.split(X, y, groups):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                # Count class distribution before undersampling
                train_counts = Counter(y_train)
                min_class = min(train_counts, key=train_counts.get)
                min_samples = train_counts[min_class]  # Number of samples in the minority class

                # Split majority and minority class
                majority_class = max(train_counts, key=train_counts.get)
                X_majority = X_train[y_train == majority_class]
                y_majority = y_train[y_train == majority_class]
                X_minority = X_train[y_train == min_class]
                y_minority = y_train[y_train == min_class]

                # Undersample majority class
                X_majority_resampled, y_majority_resampled = resample(
                    X_majority, y_majority, replace=False, n_samples=min_samples, random_state=42
                )

                # Combine balanced dataset
                X_train_balanced = np.vstack((X_majority_resampled, X_minority))
                y_train_balanced = np.hstack((y_majority_resampled, y_minority))

                # Build a new model using the current hyperparameters
                model = tuner.hypermodel.build(trial_hps)

                # Train the model on the balanced training set
                model.fit(X_train_balanced, y_train_balanced, epochs=10, validation_data=(X_val, y_val), verbose=1)

                # Evaluate and store accuracy for this fold
                _, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
                fold_accuracies.append(val_accuracy)

            # Compute the average validation accuracy across all folds
            avg_val_accuracy = np.mean(fold_accuracies)

            # Update best hyperparameters if this trial performs better
            if avg_val_accuracy > best_avg_val_accuracy:
                best_avg_val_accuracy = avg_val_accuracy
                best_hps = trial_hps

        # Build and return the best model found across all folds
        best_model = tuner.hypermodel.build(best_hps)
        print("Best hyperparameters: ", best_hps.values)
        print("Best average validation accuracy: ", best_avg_val_accuracy)
        return best_model, best_avg_val_accuracy

    except Exception as e:
        print("Error building model: ", e)
        return None

    