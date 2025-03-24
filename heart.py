import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data(filepath, target_column):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)  # Remove missing values

    # Encode categorical variables if necessary
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print(report)

def main():
    datasets = {
        'heart_disease': '/heart_disease_data.csv'
    }

    models = {}

    for disease, filepath in datasets.items():
        print(f'Training model for {disease}...')
        X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath, target_column='target')
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        models[disease] = model

    return models
    for disease, filepath in datasets.items(lung_disease):
        print(f'Training model for {disease}...')
        X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath, target_column='target')
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        models[disease] = model

    return models



if __name__ == "__main__":
    trained_models = main()
