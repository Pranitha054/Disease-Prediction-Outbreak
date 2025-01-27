import os
import pickle
import joblib  # Import joblib for loading models saved with joblib
import streamlit as st
from streamlit_option_menu import option_menu

# Set the page configuration for Streamlit
st.set_page_config(page_title="Prediction of Disease Outbreaks", layout="wide", page_icon="")

# Define the working directory where your models are stored
working_dir = "C:/Users/Pranitha/OneDrive/Desktop/jupyter notes"

# Function to load models safely with os.path.join for handling paths
def load_model(model_name, use_joblib=False):
    try:
        model_path = os.path.join(working_dir, 'saved_models', model_name)
        if use_joblib:
            return joblib.load(model_path)  # Use joblib to load the model
        else:
            with open(model_path, 'rb') as file:
                return pickle.load(file)  # Use pickle to load the model
    except Exception as e:
        st.error(f"Error loading {model_name}: {str(e)}")
        return None

# Load the models (use joblib for Parkinson's model, pickle for others)
diabetes_model = load_model('diabetes_model.sav')
heart_disease_model = load_model('heart_disease_model.sav')
parkinsons_model = load_model('parkinsons_model.sav', use_joblib=True)  # Load Parkinson's model using joblib

# Streamlit sidebar menu for navigation
with st.sidebar:
    selected = option_menu(
        'Prediction of Disease Outbreaks System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# Diabetes Prediction
if selected == "Diabetes Prediction":
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0, step=1)
    with col3:
        BloodPressure = st.number_input('Blood Pressure Value', min_value=0, step=1)
    with col1:
        SkinThickness = st.number_input('Skin Thickness Value', min_value=0, step=1)
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0, step=1)
    with col3:
        BMI = st.number_input('BMI Value', min_value=0.0, step=0.1)
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function Value', min_value=0.0, step=0.01)
    with col2:
        Age = st.number_input('Age of the Person', min_value=0, step=1)

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        diab_prediction = diabetes_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'
    st.success(diab_diagnosis)

# Heart Disease Prediction
elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', min_value=0, step=1)
    with col2:
        sex = st.number_input('Sex (1=Male, 0=Female)', min_value=0, max_value=1, step=1)
    with col3:
        cp = st.number_input('Chest Pain Types (0-3)', min_value=0, max_value=3, step=1)
    with col1:
        testbps = st.number_input('Resting Blood Pressure', min_value=0, step=1)
    with col2:
        cho = st.number_input('Serum Cholesterol (mg/dl)', min_value=0, step=1)
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl (1=Yes, 0=No)', min_value=0, max_value=1, step=1)
    with col1:
        restecg = st.number_input('Resting Electrocardiographic Results (0,1,2)', min_value=0, max_value=2, step=1)
    with col2:
        thalach = st.number_input('Maximum Heart Rate', min_value=0, step=1)
    with col3:
        exang = st.number_input('Exercise Induced Angina (1=Yes, 0=No)', min_value=0, max_value=1, step=1)
    with col1:
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, step=0.1)
    with col2:
        slope = st.number_input('Slope of Peak Exercise ST Segment', min_value=0, max_value=2, step=1)
    with col3:
        ca = st.number_input('Major Vessels Colored by Fluoroscopy (0-3)', min_value=0, max_value=3, step=1)
    with col1:
        thal = st.number_input('Thal: 0=Normal, 1=Fixed Defect, 2=Reversible Defect', min_value=0, max_value=2, step=1)

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, testbps, cho, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        heart_prediction = heart_disease_model.predict([user_input])
        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have heart disease'
    st.success(heart_diagnosis)

# Parkinson's Disease Prediction
elif selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, step=0.1)
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, step=0.1)
    with col3:
        flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, step=0.1)
    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, step=0.1)
    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, step=0.1)
    with col1:
        RAP = st.number_input('MDVP:RAP', min_value=0.0, step=0.1)
    with col2:
        PPQ = st.number_input('MDVP:PPQ', min_value=0.0, step=0.1)
    with col3:
        DDP = st.number_input('Jitter:DDP', min_value=0.0, step=0.1)
    with col4:
        Shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, step=0.1)
    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, step=0.1)
    with col1:
        APQ3 = st.number_input('Shimmer:APQ3', min_value=0.0, step=0.1)
    with col2:
        APQ5 = st.number_input('Shimmer:APQ5', min_value=0.0, step=0.1)
    with col3:
        APQ = st.number_input('MDVP:APQ', min_value=0.0, step=0.1)
    with col4:
        DDA = st.number_input('Shimmer:DDA', min_value=0.0, step=0.1)
    with col5:
        NHR = st.number_input('NHR', min_value=0.0, step=0.1)
    with col1:
        HNR = st.number_input('HNR', min_value=0.0, step=0.1)
    with col2:
        RPDE = st.number_input('RPDE', min_value=0.0, step=0.1)
    with col3:
        DFA = st.number_input('DFA', min_value=0.0, step=0.1)
    with col4:
        spread1 = st.number_input('spread1', min_value=0.0, step=0.1)
    with col5:
        spread2 = st.number_input('spread2', min_value=0.0, step=0.1)
    with col1:
        D2 = st.number_input('D2', min_value=0.0, step=0.1)
    with col2:
        PPE = st.number_input('PPE', min_value=0.0, step=0.1)

    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, RPDE, DFA, spread1, spread2, D2, PPE]
        parkinsons_prediction = parkinsons_model.predict([user_input])
        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
    st.success(parkinsons_diagnosis)
