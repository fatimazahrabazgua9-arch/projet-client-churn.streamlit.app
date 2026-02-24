import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# --- CONFIGURATION ---
st.set_page_config(page_title="Churn IA Fix", layout="wide")

# Vider le cache si on change de mode pour éviter les conflits de colonnes
if 'last_mode' not in st.session_state:
    st.session_state.last_mode = None

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_and_clean(mode):
    # Chargement
    df_b = pd.read_csv('bank_customer_churn_10k.csv')
    df_t = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    if mode == "Banque":
        df = df_b.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
        df_final = pd.get_dummies(df, drop_first=True)
        return df_final, 'Exited'
    
    elif mode == "Télécom":
        df = df_t.drop('customerID', axis=1)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        df_final = pd.get_dummies(df, drop_first=True)
        return df_final, 'Churn'
    
    else: # FUSION
        b = pd.DataFrame({'tenure': df_b['Tenure']*12, 'target': df_b['Exited'], 'secteur': 1})
        t = pd.DataFrame({'tenure': df_t['tenure'], 'target': df_t['Churn'].map({'Yes':1,'No':0}), 'secteur': 0})
        return pd.concat([b, t], axis=0).reset_index(drop=True), 'target'

# --- INTERFACE ---
mode = st.sidebar.selectbox("Choisir le mode", ["Banque", "Télécom", "Fusion"])

# Reset automatique du cache interne si on change de mode
if st.session_state.last_mode != mode:
    st.cache_data.clear()
    st.session_state.last_mode = mode

df, target = load_and_clean(mode)
X = df.drop(target, axis=1)
y = df[target]

# Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Entrainement du modèle
model = XGBClassifier(n_estimators=100).fit(X_train_s, y_train)

st.title(f"Analyse : {mode}")

# --- FORMULAIRE DE PRÉDICTION ---
st.subheader("Faire une prédiction")
col1, col2 = st.columns(2)

with col1:
    # Créer un dictionnaire de base avec des zéros pour TOUTES les colonnes attendues
    user_input = {col: 0 for col in X.columns}
    
    if mode == "Banque":
        user_input['CreditScore'] = st.number_input("Credit Score", 300, 850, 600)
        user_input['Age'] = st.number_input("Age", 18, 100, 40)
        user_input['Tenure'] = st.number_input("Tenure (Années)", 0, 10, 5)
        user_input['Balance'] = st.number_input("Balance", 0, 200000, 50000)
    elif mode == "Télécom":
        user_input['tenure'] = st.number_input("Tenure (Mois)", 0, 72, 12)
        user_input['MonthlyCharges'] = st.number_input("Monthly Charges", 0, 150, 70)
    else:
        user_input['tenure'] = st.number_input("Tenure (Mois)", 0, 120, 24)

    # ÉTAPE CRUCIALE : On transforme le dictionnaire en DataFrame 
    # et on force l'ordre des colonnes pour qu'il soit identique à X
    final_input_df = pd.DataFrame([user_input])[X.columns]
    
    # Scaling
    final_input_scaled = scaler.transform(final_input_df)

with col2:
    pred = model.predict_proba(final_input_scaled)[0][1]
    st.metric("Risque de départ", f"{round(pred*100, 2)}%")
    if pred > 0.5: st.error("Alerte Churn")
    else: st.success("Client Fidèle")

# --- PERFORMANCE ---
st.divider()
acc = accuracy_score(y_test, model.predict(X_test_s))
st.write(f"Précision du modèle actuel : **{round(acc*100, 2)}%**")