import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(layout="wide", page_title="Loan Approval Predictor")

st.title("Loan Approval Prediction App")
st.write("Enter customer details to predict loan approval status.")

# --- Load the trained model and scaler ---
@st.cache_resource
def load_model_and_scaler():
    try:
        # FIXED: Match your actual file names
        with open('loan_analysiss (1).pkl', 'rb') as file:
            model = pickle.load(file)
        with open('loan_scaler (1).pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model or scaler file not found. Make sure 'loan_analysiss.pkl' and 'loan_scaler.pkl' are in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        st.stop()

model, scaler = load_model_and_scaler()

# --- Define the expected columns after one-hot encoding ---
# This list should match the columns of X_train after preprocessing
expected_columns = [
    'Granted_Loan_Amount', 'FICO_score', 'Monthly_Gross_Income', 'Monthly_Housing_Payment',
    'Ever_Bankrupt_or_Foreclose', 'gross_income_ratio', 'fico_income', 'good_fico_threshold',
    'risk_assessment', 'Employment_Status_part_time', 'Employment_Status_unemployed',
    'Employment_Sector_consumer_discretionary', 'Employment_Sector_consumer_staples',
    'Employment_Sector_energy', 'Employment_Sector_financials', 'Employment_Sector_health_care',
    'Employment_Sector_industrials',
    'Employment_Sector_information_technology',
    'Employment_Sector_materials', 'Employment_Sector_real_estate', 'Employment_Sector_unknown',
    'Employment_Sector_utilities',
    'Lender_B', 'Lender_C', 'Fico_Score_group_fair',
    'Fico_Score_group_good', 'Fico_Score_group_poor', 'Fico_Score_group_very_good'
]

# --- User Input Fields ---
st.header("Customer Information")
col1, col2 = st.columns(2)

with col1:
    granted_loan_amount = st.number_input('Granted Loan Amount', min_value=1000, max_value=200000, value=50000, step=1000)
    fico_score = st.number_input('FICO Score', min_value=300, max_value=850, value=650, step=1)
    monthly_gross_income = st.number_input('Monthly Gross Income', min_value=500.0, max_value=50000.0, value=3500.0, step=100.0)
    monthly_housing_payment = st.number_input('Monthly Housing Payment', min_value=100, max_value=10000, value=800, step=50)

with col2:
    employment_status_options = ['full_time', 'part_time', 'unemployed']
    employment_status = st.selectbox('Employment Status', employment_status_options)

    employment_sector_options = ['consumer_discretionary', 'consumer_staples', 'energy', 'financials', 'health_care',
                                   'industrials', 'information_technology', 'materials', 'real_estate', 'utilities', 'unknown']
    employment_sector = st.selectbox('Employment Sector', employment_sector_options)

    lender_options = ['A', 'B', 'C']
    lender = st.selectbox('Lender', lender_options)

    ever_bankrupt_or_foreclose = st.radio('Ever Bankrupt or Foreclosed?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# --- Feature Engineering (consistent with notebook preprocessing) ---
input_data = {
    'Granted_Loan_Amount': granted_loan_amount,
    'FICO_score': fico_score,
    'Monthly_Gross_Income': monthly_gross_income,
    'Monthly_Housing_Payment': monthly_housing_payment,
    'Ever_Bankrupt_or_Foreclose': ever_bankrupt_or_foreclose,
    'Employment_Status': employment_status,
    'Employment_Sector': employment_sector,
    'Lender': lender,
}

# Calculate derived features
if monthly_gross_income > 0:
    input_data['gross_income_ratio'] = monthly_housing_payment / monthly_gross_income
else:
    input_data['gross_income_ratio'] = 0

input_data['fico_income'] = fico_score * monthly_gross_income
input_data['good_fico_threshold'] = 1 if fico_score > 670 else 0

# Determine Fico_Score_group - Corrected logic
if fico_score >= 800: # Excellent
    fico_group = 'excellent'
elif fico_score >= 740: # Very Good
    fico_group = 'very_good'
elif fico_score >= 670: # Good
    fico_group = 'good'
elif fico_score >= 580: # Fair
    fico_group = 'fair'
else: # Poor
    fico_group = 'poor'

# Calculate risk_assessment - Aligned with notebook's binary logic
input_data['risk_assessment'] = int(
    (input_data['FICO_score'] < 580) and
    (input_data['Employment_Status'] == 'unemployed') and
    (input_data['gross_income_ratio'] > 0.5) and
    (input_data['Ever_Bankrupt_or_Foreclose'] == 1)
)

# Create a DataFrame for prediction
# This DataFrame needs to have the exact same columns as X_train in the same order
processed_input = pd.DataFrame(0, index=[0], columns=expected_columns)

# Populate numerical features
for col in ['Granted_Loan_Amount', 'FICO_score', 'Monthly_Gross_Income', 'Monthly_Housing_Payment',
            'Ever_Bankrupt_or_Foreclose', 'gross_income_ratio', 'fico_income',
            'good_fico_threshold', 'risk_assessment']:
    if col in processed_input.columns:
        processed_input[col] = input_data[col]

# Populate one-hot encoded categorical features
if f'Employment_Status_{employment_status}' in processed_input.columns:
    processed_input[f'Employment_Status_{employment_status}'] = 1

if f'Employment_Sector_{employment_sector}' in processed_input.columns:
    processed_input[f'Employment_Sector_{employment_sector}'] = 1

# Only set Lender_B or Lender_C to 1 if the selected lender is B or C
# The base case (Lender A) is handled by keeping both Lender_B and Lender_C columns as 0.
if lender == 'B' and 'Lender_B' in processed_input.columns:
    processed_input['Lender_B'] = 1
elif lender == 'C' and 'Lender_C' in processed_input.columns:
    processed_input['Lender_C'] = 1

# Set the Fico_Score_group one-hot encoded column (if it exists)
# 'excellent' is likely the dropped reference column
if fico_group != 'excellent' and f'Fico_Score_group_{fico_group}' in processed_input.columns:
    processed_input[f'Fico_Score_group_{fico_group}'] = 1

# --- Scale numerical features ---
features_to_scale = ['Granted_Loan_Amount', 'FICO_score', 'Monthly_Gross_Income', 'Monthly_Housing_Payment', 'gross_income_ratio', 'fico_income']

# Check if features exist before scaling
available_features = [f for f in features_to_scale if f in processed_input.columns]
processed_input[available_features] = scaler.transform(processed_input[available_features])

# --- Debug Info (Remove after testing) ---
with st.expander("Debug Info (Click to expand)"):
    st.write(f"FICO Group: {fico_group}")
    st.write(f"Risk Assessment: {input_data['risk_assessment']}") # Display the single binary value
    st.write(f"Processed input shape: {processed_input.shape}")
    st.write(f"Model expects {getattr(model, 'n_features_in_', 'Unknown')} features")
    st.write("Processed input head:")
    st.dataframe(processed_input.head())

# --- Prediction ---
st.header("Prediction")
if st.button('Predict Loan Approval'):
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)[:, 1]

    # Rule-based overrides are applied *after* model prediction but *before* displaying model result
    final_status_message = ""
    final_probability = prediction_proba[0]

    # Apply business rules as overrides or additional flags
    if fico_score < 500 or monthly_gross_income < monthly_housing_payment or ever_bankrupt_or_foreclose == 1:
        final_status_message = f"**Loan Denied (Business Rule: High Risk)**. (Model Probability: {final_probability:.2%})"
        st.error(final_status_message)
    elif fico_score > 720 and monthly_gross_income > 5000 and ever_bankrupt_or_foreclose == 0:
        final_status_message = f"**Loan Approved (Business Rule: Strong Applicant)**. (Model Probability: {final_probability:.2%})"
        st.success(final_status_message)
    elif prediction[0] == 1:
        final_status_message = f"**Loan Approved!** (Model Probability: {final_probability:.2%})"
        st.success(final_status_message)
    else:
        final_status_message = f"**Loan Denied.** (Model Probability: {final_probability:.2%})"
        st.error(final_status_message)

    st.subheader("Input Data Used for Prediction:")
    st.dataframe(processed_input.T)

