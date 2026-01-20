import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1. Load the Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# 2. The Logic Function
def predict_Purchased(Gender, Age, EstimatedSalary):
    
    # Pack inputs into a DataFrame
    # The column names must match your CSV file exactly
    input_df = pd.DataFrame([[
        Gender, Age, EstimatedSalary
    ]],
      columns=[
        "Gender", "Age", "EstimatedSalary"
    ])
    
    # Predict
    prediction = model.predict(input_df)[0]

    result = ""

    if prediction == 0:
        result = "No"
    else:
        result = "Yes"
    
    # Return formatted result
    return f"Predicted Purchased Result: {result}"

inputs = [
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Number(label="Age", value=18),
    gr.Number(label="EstimatedSalary", value=20000),
]


app = gr.Interface(
    fn=predict_Purchased,
      inputs=inputs,
        outputs="text", 
        title="Purchased Predictor")

app.launch()