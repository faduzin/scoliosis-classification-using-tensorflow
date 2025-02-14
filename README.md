## 📊 Dataset Explanation

The dataset used in this project contains patient information relevant to scoliosis classification. It includes a variety of demographic, physical, and biomechanical attributes, along with a scoliosis degree measurement. The primary goal is to classify whether a patient has scoliosis based on their recorded scoliosis degree.

### 🏗 Dataset Structure  
The dataset consists of multiple columns, each representing a different attribute of the patients. The key columns are:

- **Id**: Unique identifier for each patient.  
- **Name**: Anonymized identifier for individuals.  
- **Age**: Age of the patient (in years).  
- **Mass**: Body mass of the patient (in kilograms).  
- **Height**: Height of the patient (in meters).  
- **Female / Male**: Binary columns indicating gender.  
- **Handedness (R_Handed / L_Handed)**: Indicates whether the individual is right-handed or left-handed.  
- **CoP_ML**: Center of pressure measurement in the medial-lateral direction.  
- **Scolio**: The scoliosis degree of the patient, which is the primary feature for classification.  
- **Scolio_Class**: A derived binary classification label based on the scoliosis degree:
  - 🟢 **0 (No Scoliosis)** – If scoliosis degree is **≤10**  
  - 🔴 **1 (Scoliosis Present)** – If scoliosis degree is **>10**  

Additionally, the dataset contains a series of biomechanical features (e.g., `s111, s112, ... s119`) that may be relevant for classification.

### 🎯 Target Variable for Classification  
The target variable for the classification model is **Scolio_Class**, which determines if a patient is classified as having scoliosis or not. This binary classification allows the use of various machine learning models to predict scoliosis based on patient characteristics.

### 🔧 Dataset Preprocessing Steps
- **Data Cleaning**: Formatting numerical values (e.g., converting commas to decimal points).
- **Handling Missing Values**: Ensuring there are no null or incorrect values.
- **Feature Engineering**: Creating the binary classification column (**Scolio_Class**) from the scoliosis degree.

This dataset provides a structured foundation for training machine learning models to predict scoliosis using TensorFlow. 🚀
