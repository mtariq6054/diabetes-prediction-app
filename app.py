import streamlit as st
import joblib
import pandas as pd

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvcm0zNzNiYXRjaDE1LTIxNy0wMS5qcGc.jpg");
background-size: 150%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}


[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)



# Load the saved model
model = joblib.load('random_forest_model.pkl')

# Function to predict diabetes
def predict_diabetes(data):
    prediction = model.predict(data)
    return prediction


# Streamlit app
def main():
    # Set page title and color
    st.title("Diabetes Prediction App")
    st.markdown(
        """
        <style>
            .title {
                color: #FCFFE0;
                text-align: center;
                font-size: 36px;
                padding-top: 30px;
                margin-bottom: 50px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Enter the values to predict diabetes:")

    # Input fields for user
    age = st.number_input("**Age**", min_value=0, max_value=120, step=1, value=30)
    hypertension = st.radio("**Hypertension**", options=["No", "Yes"])
    heart_disease = st.radio("**Heart Disease**", options=["No", "Yes"])
    bmi = st.number_input("**BMI**", min_value=10.0, max_value=50.0, step=0.1, value=25.0)
    hba1c_level = st.number_input("**HbA1c Level**", min_value=0.0, max_value=30.0, step=0.1, value=5.0)
    blood_glucose_level = st.number_input("**Blood Glucose Level**", min_value=10.0, max_value=300.0, step=1.0, value=100.0)

    # Convert radio button values to binary
    hypertension = 1 if hypertension == "Yes" else 0
    heart_disease = 1 if heart_disease == "Yes" else 0

    # Create a DataFrame with user input
    input_data = pd.DataFrame({
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'bmi': [bmi],
        'HbA1c_level': [hba1c_level],
        'blood_glucose_level': [blood_glucose_level]
    })

     # Predict button 
    if st.button("Predict", key="predict_button"):
        result = predict_diabetes(input_data)
        if result[0] == 0:
            st.markdown(
                """
                <div style="background-color:#f9f9f9;padding:0px;border-radius:20px;">
                    <h3 style="color:#f63366;text-align:center;">Results:</h3>
                    <p style="text-align:center;font-size:24px;">You are healthy.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif result[0] == 1:
            st.markdown(
                """
                <div style="background-color:#f9f9f9;padding:0px;border-radius:20px;">
                    <h3 style="color:#f63366;text-align:center;">Results:</h3>
                    <p style="text-align:center;font-size:24px;">You are diabetic.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# ******************* 0 and 1 result show ho ga ****************
    # # Predict button 
    # if st.button("Predict", key="predict_button"):
    #     Results = predict_diabetes(input_data)
    #     st.markdown(
    #         f"""
           
    #  <div style="background-color:#f9f9f9;padding:0px;border-radius:20px;">
    #             <h3 style="color:#f63366;text-align:center;">Results:</h3>
    #             <p style="text-align:center;font-size:24px;">{Results[0]}</p>
    #         </div>
    #         """,
    #         unsafe_allow_html=True
    #     )


if __name__ == '__main__':
    # Set background image
    main()
    st.write ("Made by Anaiza")
