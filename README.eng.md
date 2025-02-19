# Scoliosis Classification using TensorFlow

This project aims to classify scoliosis severity based on patient data. Using TensorFlow, a Multilayer Perceptron (MLP) model is built to predict whether a patient has scoliosis (scoliosis degree >10) or not (scoliosis degree ≤10). The project leverages Keras Tuner for automated hyperparameter optimization, though our experiments revealed that even with extensive tuning, the models were unable to learn meaningful patterns from the dataset.

> **Note:** A master’s thesis applied machine learning to this same dataset and reported a 75% accuracy. However, the thesis did not account for the fact that multiple samples were taken from the same individual. When this aspect is considered, our experiments did not yield consistent or meaningful results. For more details, refer to the [master’s thesis document](https://repositorio.unesp.br/entities/publication/153798ea-a515-4313-8302-a8f59bf1e127).


## Table of Contents

1. [Background](#background)
2. [What is an MLP?](#what-is-an-mlp)
3. [Dataset Information](#dataset-information)
4. [TensorFlow Implementation](#tensorflow-implementation)
5. [Keras Tuner for Hyperparameter Optimization](#keras-tuner-for-hyperparameter-optimization)
6. [The Process](#the-process)
7. [Analysis and Results](#analysis-and-results)
8. [Conclusion](#conclusion)
9. [Contact](#contact)
10. [Repository Structure](#repository-structure)

## Background

Scoliosis is characterized by an abnormal curvature of the spine, with diagnosis typically based on the measured scoliosis degree. In this project, we utilize patient demographic, physical, and biomechanical data—including baropodometer sensor readings—to classify scoliosis severity. While previous work (see the linked master’s thesis) reported promising accuracy, their analysis did not consider that multiple samples could be taken from the same individual. When this factor is accounted for, our models consistently struggled to learn from the data.

## What is an MLP?

A **Multilayer Perceptron (MLP)** is a type of feedforward neural network composed of:
- **Input Layer:** Receives the raw features.
- **Hidden Layers:** Consist of neurons applying a weighted sum and an activation function (typically ReLU) to extract complex patterns.
- **Output Layer:** Produces the final prediction. For binary classification tasks like ours, a sigmoid activation function is used to output probabilities.

MLPs are trained using backpropagation to minimize a loss function (e.g., binary crossentropy) and are particularly well-suited for tabular data.

## Dataset Information

The dataset comprises patient records with a range of features, including:

- **Id:** Unique identifier for each patient.
- **Name:** Anonymized patient identifier.
- **Age:** Patient age (years).
- **Mass:** Patient body mass (kg).
- **Height:** Patient height (m).
- **Gender:** Binary columns indicating gender.
- **Handedness:** Indicators for right-handed or left-handed, though redundant columns existed.
- **CoP_ML:** Center of pressure measurement in the medial-lateral direction.
- **Scolio:** Measured scoliosis degree.
- **Scolio_Class:** Binary label:
  - **0 (No Scoliosis):** Scoliosis degree ≤10
  - **1 (Scoliosis Present):** Scoliosis degree >10

In addition, the dataset includes 120 biomechanical features from baropodometer readings (columns `s0` to `s119`), where:
- `s0` to `s59` are readings from the left foot.
- `s60` to `s119` are readings from the right foot.

Each sensor value represents the mean pressure over a 30–60 second period.

## TensorFlow Implementation

The MLP is implemented using TensorFlow’s Keras API. The high-level steps include:

1. **Data Preparation:**
   - **Cleaning & Feature Selection:**  
     - Imported the data and analyzed all non-sensor features to check for high correlations (correlation >0.8).
     - Removed 12 highly correlated features, retaining only the most representative ones.
     - Removed redundant features (e.g., duplicate columns for handedness and gender) and the unique ID.
   - **Exploratory Data Analysis:**  
     - Conducted basic exploration with boxplots and histograms.
     - Observed that the age distribution is wide, with peaks between 10–20 and 55–65 years.
     - Noted that weight ranges from 20 to 110 kg (most concentrated between 60–80 kg) and heights between 1.6–1.7 meters.
     - The dataset is predominantly female (approximately 120 out of 148 samples) and almost entirely right-handed.
   - **Result:** After cleaning, the dataframe consisted of 137 features.

2. **Building the MLP Model:**
   - **Input Layer:** Receives the preprocessed features.
   - **Hidden Layers:** One or more fully-connected (Dense) layers with ReLU activations.
   - **Dropout Layers:** Included to prevent overfitting.
   - **Output Layer:** A single neuron with a sigmoid activation for binary classification.
   
   Example snippet:
   ```python
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
   ```
   
3. Training:
- The model is trained on the prepared dataset with appropriate batch sizes and epochs.
- Early stopping callbacks are used to halt training when the validation loss plateaus.

## Keras Tuner for Hyperparameter Optimization

To fine-tune the MLP’s architecture, Keras Tuner is employed. This tool automates the search for optimal hyperparameters, such as:

- **Number of Layers & Neurons:** Testing different combinations of hidden layers and neuron counts.
- **Dropout Rates:** Experimenting with various dropout probabilities to balance between underfitting and overfitting.
- **Learning Rate:** Adjusting the learning rate of the optimizer.
- **Epochs:** Optimizing these parameters to improve convergence speed and generalization.

A simplified example using Keras Tuner:

```python
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
```

This approach allows rapid iteration over model architectures, ensuring that the final model is well-optimized for the classification task.

## The Process

### Data Import & Cleaning

- **Imported the raw data.**
- **Explored non-sensor features** to assess correlations.
- **Removed 12 features** with high inter-correlation (keeping only the most representative).
- **Eliminated redundant features** (duplicate handedness and gender columns) and the unique ID.
- **Performed exploratory data analysis** using boxplots and histograms.
  - **Observations:**
    - **Age:** Wide distribution with peaks at 10–20 and 55–65 years.
    - **Weight:** Ranges from 20 to 110 kg, concentrated in the 60–80 kg range.
    - **Height:** Mainly between 1.6 and 1.7 meters.
    - **Gender:** Predominantly female (∼120 of 148 samples).
    - **Handedness:** Almost all individuals are right-handed (one exception).
- **Final cleaned dataframe:** 137 features.

### Data Scaling & Splitting

- **Scaled the processed dataset.**
- **Split the data** using stratified and grouped sampling (to account for multiple samples per individual).

### Model Training & Evaluation

- **Initial results:** The model learned only one class, classifying all inputs into a single category.
- **Approaches to address class imbalance:**
  - **Undersampling:** Did not improve the situation.
  - **PCA-based Feature Selection:** Also led to the same issue.
  - **Loss Function Adjustment:** Switched from binary crossentropy to focal loss to weight classes, but the results remained unchanged.
- **Keras Tuner:** Applied for hyperparameter search; however, due to the model's inability to learn effectively, the tuning outcomes were inconsistent.

## Analysis and Results

Despite extensive preprocessing and multiple strategies to handle data imbalance and redundancy, our models consistently failed to learn meaningful patterns from the dataset. Every approach—undersampling, PCA, and alternative loss functions—resulted in a model that predicted only one class. This suggests that, under the current methodology, there is insufficient evidence to establish a relationship between baropodometer sensor data and scoliosis classification.

## Conclusion

At this stage, we cannot prove any relation between the use of a baropodometer sample to predict scoliosis severity. Although previous work (the master’s thesis) reported 75% accuracy, that analysis did not consider the impact of multiple samples per individual. When accounting for this factor, our models did not yield a meaningful or consistent prediction.


## Contact

If you have any questions or feedback, feel free to reach out:  
[GitHub](https://github.com/faduzin) | [LinkedIn](https://www.linkedin.com/in/ericfadul/) | [eric.fadul@gmail.com](mailto:eric.fadul@gmail.com)

## Repository Structure

```
.
├── assets/           # Contains the images used in the analysis
├── data/             # Stores the datasets required for the project
├── notebooks/        # Includes Jupyter Notebook (.ipynb) files
├── src/              # Package with modules and modularized functions
├── .gitignore        # List of files and directories to be ignored by Git
├── license           # Project's license file
├── readme.eng.md     # English version of the README
└── readme.md         # Main README (in Portuguese), 
```