from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("trained_model.pkl")

# Function to preprocess input data
def preprocess_input(entity, year):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({"Entity": [entity], "Year": [year]})
    
    # Perform one-hot encoding for the "Entity" column
    input_data = pd.get_dummies(input_data, columns=["Entity"])
    return input_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input values from the form
    entity = request.form['entity']
    year = int(request.form['year'])
    
    # Preprocess the input data
    input_data = preprocess_input(entity, year)
    
    # Make predictions
    prediction = model.predict(input_data)
    
    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
