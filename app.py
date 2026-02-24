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
st.set_page_config(page_title="IA Churn Pro", layout="wide")

# --- PROTECTION DU CACHE ---
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = None

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_and_process(mode):
    try:
        df_b = pd.read_csv('bank_customer_churn_10k.csv')
        df_t = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    except Exception as e:
        return None, f"Erreur de fichier : {e}"

    if mode == "Banque":
        df = df_b.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
        df_final = pd.get_dummies(df, drop_first=True)
        return df_final, 'Exited'
    
    elif mode == "Télécom":
        df = df_t.drop('customerID', axis=1)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        serv_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        df['ServiceCount'] = (df[serv_cols] == 'Yes').sum(axis=1)
        df_final = pd.get_dummies(df, drop_first=True)
        return df_final, 'Churn'
    
    else: # MODE FUSION
        b = pd.DataFrame({'tenure': df_b['Tenure']*12, 'services': df_b['NumOfProducts'], 'target': df_b['Exited'], 'is_bank': 1})
        t_services = (df_t[['PhoneService', 'MultipleLines', 'OnlineSecurity']] == 'Yes').sum(axis=1)
        t = pd.DataFrame({'tenure': df_t['tenure'], 'services': t_services, 'target': df_t['Churn'].map({'Yes':1,'No':0}), 'is_bank': 0})
        return pd.concat([b, t], axis=0).reset_index(drop=True), 'target'

# --- INTERFACE ---
st.sidebar.title("🛠️ Menu Expert")
mode = st.sidebar.selectbox("Mode d'analyse", ["Banque", "Télécom", "Fusion (Recherche)"])

# Reset si changement de mode
if st.session_state.current_mode != mode:
    st.cache_data.clear()
    st.session_state.current_mode = mode

res = load_and_process(mode)
df, target_col = res[0], res[1]

if df is not None:
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = XGBClassifier().fit(X_train_s, y_train)

    st.title(f"🚀 Dashboard Churn : {mode}")
    
    t1, t2, t3 = st.tabs(["🎯 Prédiction", "📊 Performance", "🔬 Recherche"])

    with t1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Saisie Client")
            user_data = {col: 0 for col in X.columns}
            
            if mode == "Banque":
                user_data['CreditScore'] = st.number_input("Score Crédit", 300, 850, 600)
                user_data['Age'] = st.number_input("Âge", 18, 90, 40)
                user_data['Tenure'] = st.slider("Ancienneté (Années)", 0, 10, 5)
            elif mode == "Télécom":
                user_data['tenure'] = st.slider("Ancienneté (Mois)", 0, 72, 12)
                user_data['MonthlyCharges'] = st.number_input("Frais ($)", 0, 150, 70)
            else:
                user_data['tenure'] = st.slider("Ancienneté Globale (Mois)", 0, 120, 24)
                user_data['services'] = st.slider("Nombre de services", 1, 8, 2)

            final_df = pd.DataFrame([user_data])[X.columns]
            final_scaled = scaler.transform(final_df)

        with col2:
            prob = model.predict_proba(final_scaled)[0][1]
            st.metric("Risque estimé", f"{round(prob*100, 2)}%")
            if prob > 0.5:
                st.error("⚠️ Risque de départ détecté")
            else:
                st.success("✅ Client Fidèle")

    with t2:
        acc = accuracy_score(y_test, model.predict(X_test_s))
        st.write(f"Précision de l'IA : **{round(acc*100, 2)}%**")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.heatmap(confusion_matrix(y_test, model.predict(X_test_s)), annot=True, fmt='d', cmap='viridis')
        st.pyplot(fig)

    with t3:
        st.header("Étude Transversale")
        st.line_chart(df.groupby('tenure' if 'tenure' in df.columns else 'tenure_mois')[target_col].mean())

else:
    st.error("Fichiers CSV manquants ou erreur de chargement.")