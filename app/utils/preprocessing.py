import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(data):
    data.columns = data.columns.str.strip()

    if 'Type_of_attack' not in data.columns:
        print("❌ Error: 'Type_of_attack' column not found in the DataFrame. Please check your CSV file.")
        return None, None 

    y_raw = data['Type_of_attack']

    columns_to_drop = ['No.', 'Time', 'Type'] 
    df_features = data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1)

    if 'Type_of_attack' in df_features.columns:
        df_features = df_features.drop('Type_of_attack', axis=1)

    categorical_columns_X = df_features.select_dtypes(include=['object']).columns
    for col in categorical_columns_X:
        le = LabelEncoder()
        df_features[col] = le.fit_transform(df_features[col])

    attack_type_mapping = {
        "No Attack": 0, "Non-Attack": 0, "DoS Attack": 1, "ARP Spoofing": 2,
        "Nmap Port Scan": 3, "Port Scan": 3, "Smurf Attack": 4
    }

    y_encoded = y_raw.map(attack_type_mapping).fillna(0).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    return X_scaled, y_encoded