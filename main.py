import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load datasets
train_data = pd.read_csv("dataset/Training.csv")
test_data = pd.read_csv("dataset/Testing.csv")

# Separate symptoms and disease
X_train = train_data.drop("prognosis", axis=1)
y_train = train_data["prognosis"]

X_test = test_data.drop("prognosis", axis=1)
y_test = test_data["prognosis"]

# Convert disease names into numbers
label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Create the Machine Learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train_encoded)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, predictions)

print("Disease Prediction Model")
print("------------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Number of symptoms:", len(X_train.columns))
print("Number of diseases:", len(label_encoder.classes_))
print("Test Accuracy:", accuracy)

# Example prediction
sample = X_test.iloc[0:1]

prediction = model.predict(sample)

predicted_disease = label_encoder.inverse_transform(prediction)

print("Example Predicted Disease:", predicted_disease[0])