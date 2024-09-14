import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
import numpy as np

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('startup_data.csv')
    return data

data = load_data()

def modeling(df):
    df = df.drop(['Unnamed: 0', 'Unnamed: 6', 'latitude', 'longitude', 'zip_code', 'id', 'name', 'object_id'], axis=1)
    
    df['State'] = 'other'
    for state in ['CA', 'NY', 'MA', 'TX', 'WA']:
        df.loc[df['state_code'] == state, 'State'] = state
    
    df = df.drop(['state_code', 'state_code.1', 'is_CA', 'is_NY', 'is_MA', 'is_TX', 'is_otherstate'], axis=1)
    
    df['category'] = 'other'
    categories = ['software', 'web', 'mobile', 'enterprise', 'advertising', 'games_video', 'semiconductor', 'network_hosting', 'biotech', 'hardware', 'ecommerce', 'public_relations']
    for category in categories:
        df.loc[df['category_code'] == category, 'category'] = category
    
    df['City'] = 'other'
    cities = ['San Francisco', 'New York', 'Mountain View', 'Palo Alto', 'Santa Clara']
    for city in cities:
        df.loc[df['city'] == city, 'City'] = city
    
    df = df.drop(['city', 'labels', 'category_code', 'is_software', 'is_web', 'is_mobile', 'is_enterprise', 'is_advertising', 'is_gamesvideo', 'is_ecommerce', 'is_biotech', 'is_consulting', 'is_othercategory'], axis=1)
    
    df['founded_at'] = pd.to_datetime(df['founded_at'])
    df['founded_year'] = df['founded_at'].dt.year
    
    df['closed_at'] = df['closed_at'].fillna('0')
    drop = df.loc[(df['closed_at'] != '0') & (df['status'] == 'acquired')]
    df = df.drop(drop.index).reset_index(drop=True)
    
    df = df.drop(['founded_at', 'closed_at'], axis=1)
    
    for col in ['age_first_funding_year', 'age_last_funding_year', 'age_first_milestone_year', 'age_last_milestone_year']:
        df[col] = np.where(df[col] < 0, 0, df[col])
        if col.startswith('age_first_milestone') or col.startswith('age_last_milestone'):
            df[col] = df[col].fillna(df[col].mode()[0])
    
    df = df.drop(['first_funding_at', 'last_funding_at'], axis=1)
    
    df['has_RoundABCD'] = np.where((df['has_roundA'] == 1) | (df['has_roundB'] == 1) | (df['has_roundC'] == 1) | (df['has_roundD'] == 1), 1, 0)
    df['has_Investor'] = np.where((df['has_VC'] == 1) | (df['has_angel'] == 1), 1, 0)
    df['has_Seed'] = np.where((df['has_RoundABCD'] == 0) & (df['has_Investor'] == 1), 1, 0)
    df['invalid_startup'] = np.where((df['has_RoundABCD'] == 0) & (df['has_VC'] == 0) & (df['has_angel'] == 0), 1, 0)
    
    df = pd.get_dummies(df, columns=['State', 'category', 'City'])
    
    # Convert 'status' to binary
    df['status'] = np.where(df['status'] == 'acquired', 1, 0)
    
    X = df.drop(['status'], axis=1)
    y = df['status']
    
    return X, y

X, y = modeling(data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)
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

state_options = ['CA', 'NY', 'MA', 'TX', 'WA', 'other']
state = st.selectbox('State', options=state_options)

category_options = ['software', 'web', 'mobile', 'enterprise', 'advertising', 'games_video', 'semiconductor', 'network_hosting', 'biotech', 'hardware', 'ecommerce', 'public_relations', 'other']
category = st.selectbox('Category', options=category_options)

city_options = ['San Francisco', 'New York', 'Mountain View', 'Palo Alto', 'Santa Clara', 'other']
city = st.selectbox('City', options=city_options)

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
        'has_VC': [1 if has_VC else 0],
        'has_angel': [1 if has_angel else 0],
        'has_roundA': [1 if has_roundA else 0],
        'has_roundB': [1 if has_roundB else 0],
        'has_roundC': [1 if has_roundC else 0],
        'has_roundD': [1 if has_roundD else 0],
        'avg_participants': [avg_participants],
        'is_top500': [1 if is_top500 else 0]
    })
    
    # Apply the same preprocessing as in modling function
    input_data['has_RoundABCD'] = np.where((input_data['has_roundA'] == 1) | (input_data['has_roundB'] == 1) | (input_data['has_roundC'] == 1) | (input_data['has_roundD'] == 1), 1, 0)
    input_data['has_Investor'] = np.where((input_data['has_VC'] == 1) | (input_data['has_angel'] == 1), 1, 0)
    input_data['has_Seed'] = np.where((input_data['has_RoundABCD'] == 0) & (input_data['has_Investor'] == 1), 1, 0)
    input_data['invalid_startup'] = np.where((input_data['has_RoundABCD'] == 0) & (input_data['has_VC'] == 0) & (input_data['has_angel'] == 0), 1, 0)
    
    # One-hot encode categorical variables
    state_cols = [f'State_{s}' for s in state_options]
    category_cols = [f'category_{c}' for c in category_options]
    city_cols = [f'City_{c}' for c in city_options]

    for col in state_cols + category_cols + city_cols:
        input_data[col] = 0

    input_data[f'State_{state}'] = 1
    input_data[f'category_{category}'] = 1
    input_data[f'City_{city}'] = 1

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
    st.success(f'The Probability of success :  {100 * probability[0][1]:.2f}%')
