import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

# Load dataset

df = pd. read_csv("Social_Network_Ads.csv")


# drop data column
if "User ID" in df.columns:
  df.drop( columns = ["User ID"], inplace=True )


# Target and features
X = df.drop("Purchased",axis=1)
y = df["Purchased"]

numeric_features = X.select_dtypes(include = ['int64','float64']).columns
categorical_features = X.select_dtypes(include = ['object']).columns

# Preprocessing pipeline

num_transformer = Pipeline (
    steps = [
        ('scaler', StandardScaler())
    ]
)

cat_transformer = Pipeline( steps = [
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
] )

preprocessor = ColumnTransformer(
    transformers= [
        ('num',num_transformer,numeric_features),
        ('cat',cat_transformer,categorical_features)
    ]
    )


# Support Vector Machine Model

svc_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    random_state=42
)

# full pipeline
svm_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", svc_model)
])

X_train,X_test, y_train, y_test = train_test_split(
  X,y, test_size = 0.2 , random_state=42
)

svm_pipeline.fit(X_train, y_train)

y_pred = svm_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")

filename = "model.pkl"

with open( filename, "wb" ) as file:
  pickle.dump(svm_pipeline, file )