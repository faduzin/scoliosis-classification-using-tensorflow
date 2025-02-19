import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import resample
from collections import Counter
import numpy as np


def class_remapping(data, column_name):
    try:
        data['class'] = (data['Scolio'] > 10).astype(int)
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None


def scale_data(data):
    try:    
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None
    

def extract_individual_n(data):
    try:
        temp_list = []
        for name in data['Name']:
            temp_list.append(name.split('_')[1])
        data['individual_n'] = temp_list
        print('Individual number extracted successfully.')
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None
    

def stratified_group_split(X, y, groups, n_splits=5):
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits)

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

            # Count class distribution after undersampling
            balanced_counts = Counter(y_train_balanced)
            print(f"  Train Distribution After: {balanced_counts}")

            print(f"  Test Distribution (Unchanged): {Counter(y_val)}\n")

            return X_train_balanced, y_train_balanced, X_val, y_val
    
    except Exception as e:
        print(f"Error: {e}")
        return None
