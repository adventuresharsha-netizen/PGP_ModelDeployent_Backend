import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
lead_conversion_api = Flask("ExtraaLearn Lead Conversion Predictor")

# Load the trained lead-conversion model (preprocessing + classifier bundled in one Pipeline)
model = joblib.load("extraalearn_lead_conversion_model_v1_0.joblib")

# Define a route for the home page
@lead_conversion_api.get('/')
def home():
    return "Welcome to the ExtraaLearn Lead Conversion Prediction API!"

# Define an endpoint to predict conversion likelihood for a single lead
@lead_conversion_api.post('/v1/lead')
def predict_lead():
    # Get JSON data from the request
    lead_data = request.get_json()

    # Extract the relevant lead features from the input data
    sample = {
        'age': lead_data['age'],
        'current_occupation': lead_data['current_occupation'],
        'first_interaction': lead_data['first_interaction'],
        'profile_completed': lead_data['profile_completed'],
        'website_visits': lead_data['website_visits'],
        'time_spent_on_website': lead_data['time_spent_on_website'],
        'page_views_per_visit': lead_data['page_views_per_visit'],
        'last_activity': lead_data['last_activity'],
        'print_media_type1': lead_data['print_media_type1'],
        'print_media_type2': lead_data['print_media_type2'],
        'digital_media': lead_data['digital_media'],
        'educational_channels': lead_data['educational_channels'],
        'referral': lead_data['referral'],
    }

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Make a prediction (class label) and get the conversion probability
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    # Return the prediction as a JSON response
    return jsonify({
        'Predicted_Status': prediction,
        'Conversion_Probability': round(probability, 4),
        'Label': "Likely to Convert" if prediction == 1 else "Unlikely to Convert"
    })

# Define an endpoint to predict conversion likelihood for a batch of leads
@lead_conversion_api.post('/v1/leadbatch')
def predict_lead_batch():
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the file into a DataFrame
    input_data = pd.read_csv(file)

    # Make predictions and get conversion probabilities for the batch
    predictions = model.predict(input_data).tolist()
    probabilities = model.predict_proba(input_data)[:, 1].tolist()

    # Add predictions and probabilities to the DataFrame
    input_data['Predicted_Status'] = predictions
    input_data['Conversion_Probability'] = [round(p, 4) for p in probabilities]

    # Convert results to a list of dictionaries
    result = input_data.to_dict(orient="records")

    return jsonify(result)

# Run the Flask app in debug mode
if __name__ == '__main__':
    lead_conversion_api.run(debug=True)
