import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Algorithmes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# --- CONFIGURATION ---
st.set_page_config(page_title="Expert Churn Dash", layout="wide")

# --- MOTEUR DE TRAITEMENT (FEATURE ENGINEERING INCLUS) ---
@st.cache_data
def load_data(secteur_choisi):
    if secteur_choisi == "Banque":
        df = pd.read_csv('bank_customer_churn_10k.csv').drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
        # Feature Engineering Banque
        df['ServiceCount'] = df['NumOfProducts']
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
        df_encoded = pd.get_dummies(df, drop_first=True)
        target = 'Exited'
    else:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv').drop('customerID', axis=1)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        
        # 1. Calcul du Nombre de Services (Telco)
        serv_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        df['ServiceCount'] = (df[serv_cols] == 'Yes').sum(axis=1)
        
        # 2. Calcul du Ratio Charges/Ancienneté
        df['ChargeTenureRatio'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        df_encoded = pd.get_dummies(df, drop_first=True)
        target = 'Churn'
    
    return df_encoded, target

# --- BARRE LATÉRALE ---
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
                tenure = st.slider("Ancienneté (Années)", 0, 10, 5)
                inputs['tenure'] = tenure
                inputs['NumOfProducts'] = st.selectbox("Produits Bancaires", [1, 2, 3, 4])
                inputs['ServiceCount'] = inputs['NumOfProducts']
                inputs['Balance'] = st.number_input("Solde ($)", 0, 250000, 15000)
                geo = st.selectbox("Pays", ["France", "Germany", "Spain"])
                if f"Geography_{geo}" in inputs: inputs[f"Geography_{geo}"] = 1
            else:
                tenure = st.slider("Ancienneté (Mois)", 0, 72, 12)
                monthly = st.number_input("Frais Mensuels ($)", 0.0, 150.0, 75.0)
                services = st.slider("Services souscrits (Internet, TV, etc.)", 0, 8, 3)
                
                inputs['tenure'] = tenure
                inputs['MonthlyCharges'] = monthly
                inputs['ServiceCount'] = services
                inputs['ChargeTenureRatio'] = monthly / (tenure + 1)
                
            user_df = pd.DataFrame(inputs, index=[0])
            user_scaled = scaler.transform(user_df[X.columns])

        with col_out:
            algo = st.selectbox("Intelligence Artificielle", ["XGBoost", "Random Forest", "Logistic Regression"])
            model = XGBClassifier() if algo == "XGBoost" else (RandomForestClassifier() if algo == "Random Forest" else LogisticRegression())
            model.fit(X_train_s, y_train)
            prob = model.predict_proba(user_scaled)[0][1]

            # Affichage des KPI calculés
            st.subheader("Indicateurs Clés de Risque")
            k1, k2, k3 = st.columns(3)
            k1.metric("Probabilité Churn", f"{round(prob*100, 1)}%")
            
            if secteur == "Télécommunications":
                k2.metric("Ratio $/Mois", f"{round(inputs['ChargeTenureRatio'], 2)}")
                k3.metric("Nb Services", inputs['ServiceCount'])
            else:
                k2.metric("Engagement", f"{inputs['ServiceCount']} Prod.")
                k3.metric("Type Client", "Senior" if inputs['Age'] > 50 else "Standard")

            if prob > 0.5:
                st.error("🚨 HAUT RISQUE : Client sur le départ.")
            else:
                st.success("✅ FIDÈLE : Engagement client solide.")

            # Rapport et export
            rapport = f"Analyse {secteur}\nRisque : {round(prob*100,2)}%\nServices : {inputs['ServiceCount']}\nModèle : {algo}"
            st.download_button("📥 Télécharger Rapport d'Analyse", rapport, file_name="expert_churn.txt")

    with tab2:
        st.header("Analyse Statistique & Duel d'Algorithmes")
        if st.button("Lancer la comparaison"):
            results = []
            for name, m in {"XGBoost": XGBClassifier(), "Random Forest": RandomForestClassifier(), "LR": LogisticRegression()}.items():
                m.fit(X_train_s, y_train)
                acc = accuracy_score(y_test, m.predict(X_test_s))
                results.append({"Algorithme": name, "Précision": acc})
            
            st.table(pd.DataFrame(results))
            
            # Graphique d'importance des variables (pour montrer que nos calculs servent !)
            if hasattr(model, 'feature_importances_'):
                st.subheader("Variables les plus importantes (Feature Importance)")
                feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
                fig, ax = plt.subplots()
                feat_imp.plot(kind='bar', ax=ax, color='skyblue')
                st.pyplot(fig)

except Exception as e:
    st.error(f"Erreur système : {e}")