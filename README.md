# Scoliosis Classification using TensorFlow

This project aims to classify scoliosis severity based on patient data. Using TensorFlow, a Multilayer Perceptron (MLP) model is built to predict whether a patient has scoliosis (scoliosis degree >10) or not (scoliosis degree ≤10). The project leverages Keras Tuner for automated hyperparameter optimization, helping identify the best model topology for the task.

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

Scoliosis, a condition characterized by an abnormal curvature of the spine, is detected by measuring the scoliosis degree. In this project, we use patient demographic, physical, and biomechanical data (such as baropodometer readings) to classify the presence or absence of scoliosis. Our approach centers on a deep learning model—specifically an MLP—to address this binary classification task.

## What is an MLP?

A **Multilayer Perceptron (MLP)** is a type of feedforward artificial neural network that consists of:
- **Input Layer:** Receives the raw data.
- **Hidden Layers:** One or more layers where each neuron applies a weighted sum followed by an activation function (commonly ReLU) to capture complex patterns.
- **Output Layer:** Produces the final prediction. In binary classification, this layer typically uses a sigmoid activation function to output probabilities.

MLPs learn by adjusting the weights using backpropagation, minimizing a loss function (like binary crossentropy) during training. Their fully-connected architecture makes them particularly well-suited for tabular data, as is the case with our scoliosis dataset.

## Dataset Information

The dataset comprises patient records with various attributes, including:

- **Id:** Unique patient identifier.
- **Name:** Anonymized patient identifier.
- **Age:** Patient age in years.
- **Mass:** Patient body mass (kg).
- **Height:** Patient height (m).
- **Gender:** Represented with binary columns (Female/Male).
- **Handedness:** Indicated with binary columns (R_Handed/L_Handed).
- **CoP_ML:** Center of pressure measurement in the medial-lateral direction.
- **Scolio:** The measured scoliosis degree.
- **Scolio_Class:** Binary label derived from the scoliosis degree:
  - **0 (No Scoliosis):** Scoliosis degree ≤10
  - **1 (Scoliosis Present):** Scoliosis degree >10

Additionally, the dataset includes biomechanical features from baropodometer readings (columns `s0` to `s119`), where:
- `s0` to `s59` correspond to left foot sensors.
- `s60` to `s119` correspond to right foot sensors.

Each sensor value reflects the mean pressure recorded over a 30–60 second measurement period, providing insight into postural stability and weight distribution.

## TensorFlow Implementation

The MLP is implemented using TensorFlow’s Keras API. Below is a high-level overview of the implementation steps:

1. **Data Preparation:**
   - **Cleaning & Formatting:** Convert numerical values to the correct format and handle missing values.
   - **Normalization:** Scale numerical features to improve training performance.
   - **Encoding:** Transform categorical features (e.g., gender, handedness) into numerical representations.
   - **Splitting:** Use stratified sampling to divide data into training and testing sets, preserving class balance.

2. **Building the MLP Model:**
   - **Input Layer:** Accepts the preprocessed features.
   - **Hidden Layers:** Multiple dense layers with activation functions (ReLU is commonly used).
   - **Dropout Layers:** Included to mitigate overfitting by randomly disabling a fraction of neurons during training.
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
- **Data Preparation:** Cleaning, normalization, encoding, and splitting.
- **Model Construction:** Building a baseline MLP with TensorFlow’s Keras API.
- **Hyperparameter Tuning:** Utilizing Keras Tuner to iterate over various architectures and parameters.
- **Training & Evaluation:** Training the model with early stopping and evaluating performance using accuracy and other relevant metrics.
- **Optimization:** Refining the model based on validation results to mitigate overfitting and enhance generalization.

## Analysis and Results
After thorough experimentation:

- **Model Performance:** The best-performing model achieved high accuracy on the validation set.
- **Insights:** Keras Tuner helped identify the ideal balance between model complexity and regularization. The inclusion of dropout layers was crucial in reducing overfitting.
- **Future Work:** Further experiments could explore alternative architectures (e.g., CNNs if spatial patterns in sensor data become relevant) or advanced feature engineering techniques.

## Conclusion
This project demonstrates the effective use of an MLP for scoliosis classification using TensorFlow. By carefully preparing the dataset and leveraging Keras Tuner for hyperparameter optimization, we achieved a robust model capable of accurately distinguishing between patients with and without scoliosis. This work not only highlights the power of deep learning in medical applications but also emphasizes the importance of model optimization in achieving reliable performance.

## Contact

If you have any questions or feedback, feel free to reach out:  
[GitHub](https://github.com/faduzin) | [LinkedIn](https://www.linkedin.com/in/ericfadul/) | [eric.fadul@gmail.com](mailto:eric.fadul@gmail.com)
