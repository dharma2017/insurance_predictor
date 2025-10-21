import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="💰",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("Health Insurance Premium Predictor")
st.markdown("""
This app predicts health insurance premiums based on personal information.
Please fill in your details below to get an estimate.
""")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    # Age input
    age = st.slider("Age", min_value=18, max_value=100, value=30, help="Select your age")
    
    # Sex input
    sex = st.selectbox("Sex", options=['male', 'female'], help="Select your gender")
    
    # BMI input
    bmi = st.slider("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1, 
                    help="Body Mass Index (weight in kg / height in meters squared)")

with col2:
    # Children input
    children = st.slider("Number of Children", min_value=0, max_value=10, value=0, 
                        help="Number of dependent children")
    
    # Smoker input
    smoker = st.selectbox("Smoker", options=['no', 'yes'], help="Are you a smoker?")
    
    # Region input
    region = st.selectbox("Region", 
                         options=['southwest', 'southeast', 'northwest', 'northeast'],
                         help="Select your region of residence")

# Add a predict button
if st.button("Predict Insurance Premium", help="Click to calculate your estimated premium"):
    try:
        # Load the model and feature order
        model = joblib.load('best_insurance_model.joblib')
        feature_order = pd.read_csv('feature_order.csv').iloc[:,0].tolist()
        scaler = joblib.load('scaler.joblib')  # Load the saved scaler 
        
        # Create input data with one-hot encoding for region
        input_data = {
            'age': [age],
            'sex': [1 if sex == 'female' else 0],  # Encode sex
            'bmi': [bmi],
            'children': [children],
            'smoker': [1 if smoker == 'yes' else 0],  # Encode smoker
            'region_northeast': [1 if region == 'northeast' else 0],
            'region_northwest': [1 if region == 'northwest' else 0],
            'region_southeast': [1 if region == 'southeast' else 0],
            'region_southwest': [1 if region == 'southwest' else 0]
        }
        
        # Create DataFrame with explicit feature order
        input_processed = pd.DataFrame(columns=feature_order)
        for feature in feature_order:
            input_processed[feature] = input_data[feature]
        
        features_to_scale = scaler.feature_names_in_
        input_processed[features_to_scale] = scaler.transform(input_processed[features_to_scale])

        # Make prediction
        log1p_prediction = model.predict(input_processed)[0]  # Get raw prediction (log1p transformed)
        
        # Convert back to original scale using expm1 (inverse of log1p)
        original_premium = np.expm1(log1p_prediction)  # Using expm1 (inverse of log1p)
        yearly_premium = round(original_premium, 2)
        monthly_premium = round(yearly_premium / 12, 2)
        
        # Display results in an expander
        with st.expander("View Prediction Results", expanded=True):
            # Display estimated premium
            st.markdown("### Estimated Insurance Premium:")
            st.markdown(f"<h2 style='color: #4CAF50;'>₹{yearly_premium:,.2f} / year</h2>", 
                       unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #4CAF50;'>₹{monthly_premium:,.2f} / month</h3>",
                       unsafe_allow_html=True)
            
            # Risk Assessment
            st.markdown("### Risk Assessment:")
            risk_factors = []
            if bmi > 30:
                risk_factors.append("High BMI (over 30)")
            if smoker == 'yes':
                risk_factors.append("Smoker")
            if age > 50:
                risk_factors.append("Age over 50")
            
            if risk_factors:
                st.warning("High-risk factors identified: " + ", ".join(risk_factors))
            else:
                st.success("No major risk factors identified")
            
            # Additional Information
            st.info("""
            **Note:** This is an estimate based on the model trained on historical data. 
            Actual premiums may vary based on additional factors and insurance provider policies.
            """)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure all model files are present and inputs are valid.")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<p>Built with Streamlit • Using Machine Learning • Based on Historical Insurance Data</p>
</div>
""", unsafe_allow_html=True)