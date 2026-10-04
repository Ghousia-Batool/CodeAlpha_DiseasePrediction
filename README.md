# Disease Prediction
A machine learning project that predicts diseases based on symptoms.

## Project Overview
This project uses a machine learning model to predict a possible disease from given symptoms.
The dataset contains 132 symptoms and 41 different diseases.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Random Forest Classifier

## Dataset
The dataset contains symptom information and disease labels.
- Training samples: 4920
- Testing samples: 41
- Number of symptoms: 132
- Number of diseases: 41
The dataset is kept locally and is not uploaded to GitHub.

## Machine Learning Model
A Random Forest Classifier was used for disease prediction.
The disease names were converted into numerical labels using LabelEncoder.

## Result
The model achieved:
**Test Accurcy: 100%**

## How to Run
Install the required libraries:
```bash
pip install pandas scikit-learn