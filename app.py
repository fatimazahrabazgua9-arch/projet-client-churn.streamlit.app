import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# --- CONFIGURATION ---
st.set_page_config(page_title="Expert Churn Dash", layout="wide")

# --- CACHE DES DONNÉES ET MODÈLES (Indispensable pour éviter le blocage 403) ---
@st.cache_data
def load_data(secteur_choisi):
    if secteur_choisi == "Banque":
        df = pd.read_csv('bank_customer_churn_10k.csv').drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
        df['ServiceCount'] = df['NumOfProducts']
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
        df_encoded = pd.get_dummies(df, drop_first=True)
        target = 'Exited'
    else:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv').drop('customerID', axis=1)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        serv_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        df['ServiceCount'] = (df[serv_cols] == 'Yes').sum(axis=1)
        df['ChargeTenureRatio'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        df_encoded = pd.get_dummies(df, drop_first=True)
        target = 'Churn'
    return df_encoded, target

@st.cache_resource
def train_model(X_train, y_train, algo_name):
    if algo_name == "XGBoost":
        m = XGBClassifier()
    elif algo_name == "Random Forest":
        m = RandomForestClassifier()
    else:
        m = LogisticRegression()
    m.fit(X_train, y_train)
    return m

# --- INTERFACE ---
st.sidebar.title("🏢 Business Unit")
secteur = st.sidebar.selectbox("Secteur :", ["Banque", "Télécommunications"])

try:
    df_encoded, target_col = load_data(secteur)
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    st.title(f"🚀 Plateforme Prédictive : {secteur}")
    tab1, tab2 = st.tabs(["🎯 Prédiction Expert", "📊 Dashboard Performance"])

    with tab1:
        col_inp, col_out = st.columns([1, 2])
        with col_inp:
            st.subheader("Paramètres Client")
            inputs = {col: 0 for col in X.columns}
            if secteur == "Banque":
                inputs['Age'] = st.slider("Âge", 18, 92, 40)
                inputs['tenure'] = st.slider("Ancienneté (Années)", 0, 10, 5)
                inputs['NumOfProducts'] = st.selectbox("Produits Bancaires", [1, 2, 3, 4])
                inputs['ServiceCount'] = inputs['NumOfProducts']
                inputs['Balance'] = st.number_input("Solde ($)", 0, 250000, 15000)
            else:
                inputs['tenure'] = st.slider("Ancienneté (Mois)", 0, 72, 12)
                inputs['MonthlyCharges'] = st.number_input("Frais Mensuels ($)", 0.0, 150.0, 75.0)
                inputs['ServiceCount'] = st.slider("Services souscrits", 0, 8, 3)
                inputs['ChargeTenureRatio'] = inputs['MonthlyCharges'] / (inputs['tenure'] + 1)
            
            user_df = pd.DataFrame(inputs, index=[0])[X.columns]
            user_scaled = scaler.transform(user_df)

        with col_out:
            algo = st.selectbox("Intelligence Artificielle", ["XGBoost", "Random Forest", "Logistic Regression"])
            model = train_model(X_train_s, y_train, algo)
            prob = model.predict_proba(user_scaled)[0][1]

            st.subheader("Indicateurs de Risque")
            k1, k2, k3 = st.columns(3)
            k1.metric("Risque Churn", f"{round(prob*100, 1)}%")
            k2.metric("Nb Services", inputs['ServiceCount'])
            k3.metric("Statut", "Alerte" if prob > 0.5 else "Stable")

            if prob > 0.5: st.error("🚨 HAUT RISQUE")
            else: st.success("✅ FIDÈLE")

    with tab2:
        if st.button("Lancer la comparaison"):
            st.write("Calcul des scores en cours...")
            # Ici on compare sans tout recréer
            acc = accuracy_score(y_test, model.predict(X_test_s))
            st.info(f"Précision du modèle {algo} : {round(acc*100,2)}%")

except Exception as e:
    st.error(f"Erreur système : {e}")