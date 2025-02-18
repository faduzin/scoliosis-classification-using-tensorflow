import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def data_info(data):
    try:   
        print(data.info())
        print("-" * 50)
        print(data.describe())
        print("-" * 50)
        print(data.head(5))
        print("-" * 50)
        print("Data shape: ",data.shape)
        print("Amount of duplicates: ", data.duplicated().sum())
    except Exception as e:
        print(f"Error: {e}")
        return None


def plot_correlations(data):
    try: 
        plt.figure(figsize=(12, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
        return None


def count_classes(y):
    try:    
        unique_labels, counts = np.unique(y, return_counts=True)

        # Display the results
        for label, count in zip(unique_labels, counts):
            print(f"Label {label}: {count} occurrences")
    except Exception as e:
        print(f"Error: {e}")
        return None


def plot_confusion_matrix(y_true, y_pred, labels):
    try:
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
        return None


def plot_boxplot_and_histogram(df, column_name):
    try:
        fig, axes = plt.subplots(1, 
                                2,  
                                figsize=(12,4)) # Inicializa a figura

        sns.boxplot(x=df[column_name], ax=axes[0]) # Plota o boxplot
        axes[0].set_title(f"Box plot de {column_name}") # Adiciona o título

        axes[1].hist(df[column_name],
                    bins=30,
                    edgecolor="Black",
                    alpha=0.7) # Plota o histograma
        axes[1].set_title(f"Histograma de {column_name}") # Adiciona o título
        axes[1].set_xlabel(column_name) # Adiciona o label do eixo x
        axes[1].set_ylabel("Frequência") # Adiciona o label do eixo y

        plt.tight_layout() # Ajusta o layout
        plt.show() # Exibe o gráfico
    except Exception as e:
        print(f"Error: {e}")
        return None