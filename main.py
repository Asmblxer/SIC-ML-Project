import streamlit as st
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('startup_data.csv')
    return data

data = load_data()

# Prepare the data
X = data.drop(['status', 'Unnamed: 0', 'state_code', 'zip_code', 'id', 'city', 'Unnamed: 6', 'name', 'founded_at', 'closed_at', 'first_funding_at', 'last_funding_at', 'state_code.1', 'category_code', 'object_id'], axis=1)
y = (data['status'] == 'acquired').astype(int)  # 1 if acquired, 0 otherwise

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Streamlit app
st.title('Startup Success Predictor')

# Input fields
st.header('Enter Startup Information')
company_name = st.text_input('Company Name')
foundation_date = st.date_input('Foundation Date')
age_first_funding_year = st.number_input('Age at First Funding (years)', min_value=0.0)
age_last_funding_year = st.number_input('Age at Last Funding (years)', min_value=0.0)
relationships = st.number_input('Number of Relationships', min_value=0)
funding_rounds = st.number_input('Number of Funding Rounds', min_value=0)
funding_total_usd = st.number_input('Total Funding (USD)', min_value=0)
milestones = st.number_input('Number of Milestones', min_value=0)

state_options = ['CA', 'NY', 'MA', 'TX', 'Other']
state = st.selectbox('State', options=state_options)

category_options = ['Software', 'Web', 'Mobile', 'Enterprise', 'Advertising', 'Games/Video', 'E-commerce', 'Biotech', 'Consulting', 'Other']
category = st.selectbox('Category', options=category_options)

has_VC = st.checkbox('Has Venture Capital Funding')
has_angel = st.checkbox('Has Angel Funding')
has_roundA = st.checkbox('Has Round A Funding')
has_roundB = st.checkbox('Has Round B Funding')
has_roundC = st.checkbox('Has Round C Funding')
has_roundD = st.checkbox('Has Round D Funding')

avg_participants = st.number_input('Average Number of Funding Participants', min_value=0.0)
is_top500 = st.checkbox('Is in Top 500')

# Make prediction
if st.button('Predict Success'):
    # Prepare input data
    input_data = pd.DataFrame({
        'age_first_funding_year': [age_first_funding_year],
        'age_last_funding_year': [age_last_funding_year],
        'relationships': [relationships],
        'funding_rounds': [funding_rounds],
        'funding_total_usd': [funding_total_usd],
        'milestones': [milestones],
        'is_CA': [1 if state == 'CA' else 0],
        'is_NY': [1 if state == 'NY' else 0],
        'is_MA': [1 if state == 'MA' else 0],
        'is_TX': [1 if state == 'TX' else 0],
        'is_otherstate': [1 if state == 'Other' else 0],
        'is_software': [1 if category == 'Software' else 0],
        'is_web': [1 if category == 'Web' else 0],
        'is_mobile': [1 if category == 'Mobile' else 0],
        'is_enterprise': [1 if category == 'Enterprise' else 0],
        'is_advertising': [1 if category == 'Advertising' else 0],
        'is_gamesvideo': [1 if category == 'Games/Video' else 0],
        'is_ecommerce': [1 if category == 'E-commerce' else 0],
        'is_biotech': [1 if category == 'Biotech' else 0],
        'is_consulting': [1 if category == 'Consulting' else 0],
        'is_othercategory': [1 if category == 'Other' else 0],
        'has_VC': [1 if has_VC else 0],
        'has_angel': [1 if has_angel else 0],
        'has_roundA': [1 if has_roundA else 0],
        'has_roundB': [1 if has_roundB else 0],
        'has_roundC': [1 if has_roundC else 0],
        'has_roundD': [1 if has_roundD else 0],
        'avg_participants': [avg_participants],
        'is_top500': [1 if is_top500 else 0]
    })
    
    # Ensure all columns from training data are present
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Reorder columns to match training data
    input_data = input_data[X.columns]
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    # Display result
    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.success('The startup is predicted to be acquired!')
    else:
        st.error('The startup is predicted to fail or remain operating.')
    
    st.write(f'Probability of being acquired: {probability[0][1]:.2f}')

st.sidebar.info('This app predicts the success (acquisition) of a startup based on input parameters. Please note that this is a simplified model and should not be used for actual investment decisions.')
