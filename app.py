import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# --- CONFIGURATION ---
st.set_page_config(page_title="IA Churn Universelle", layout="wide")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def get_data(mode):
    try:
        df_bank_raw = pd.read_csv('bank_customer_churn_10k.csv')
        df_telco_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    except:
        st.error("⚠️ Erreur : Assurez-vous que les deux fichiers CSV sont sur GitHub.")
        return None, None

    if mode == "Banque":
        df = df_bank_raw.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
        df['ServiceCount'] = df['NumOfProducts']
        df_encoded = pd.get_dummies(df, drop_first=True)
        return df_encoded, 'Exited'
    
    elif mode == "Télécom":
        df = df_telco_raw.drop('customerID', axis=1)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        serv_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        df['ServiceCount'] = (df[serv_cols] == 'Yes').sum(axis=1)
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        df_encoded = pd.get_dummies(df, drop_first=True)
        return df_encoded, 'Churn'
    
    else: # MODE FUSION
        bank_univ = pd.DataFrame({
            'tenure_mois': df_bank_raw['Tenure'] * 12,
            'nb_services': df_bank_raw['NumOfProducts'],
            'secteur_banque': 1,
            'target': df_bank_raw['Exited']
        })
        telco_univ = pd.DataFrame({
            'tenure_mois': df_telco_raw['tenure'],
            'nb_services': (df_telco_raw[['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1),
            'secteur_banque': 0,
            'target': df_telco_raw['Churn'].map({'Yes': 1, 'No': 0})
        })
        return pd.concat([bank_univ, telco_univ], axis=0).reset_index(drop=True), 'target'

# --- INTERFACE ---
st.sidebar.title("🧬 Intelligence Artificielle")
mode_select = st.sidebar.selectbox("Mode d'analyse :", ["Banque", "Télécom", "Fusion (Recherche)"])

df, target = get_data(mode_select)

if df is not None:
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Séparation et Scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.title(f"🚀 Plateforme Churn : Mode {mode_select}")
    tab1, tab2, tab3 = st.tabs(["🎯 Prédiction", "📊 Performance", "🔬 Recherche"])

    with tab1:
        if mode_select != "Fusion (Recherche)":
            col_in, col_out = st.columns([1, 2])
            with col_in:
                st.subheader("Profil Client")
                # Créer un dictionnaire vide basé sur les colonnes d'entraînement
                input_data = {col: 0 for col in X.columns}
                
                if mode_select == "Banque":
                    input_data['CreditScore'] = st.slider("Score de Crédit", 300, 850, 600)
                    input_data['Age'] = st.slider("Âge", 18, 92, 40)
                    input_data['Tenure'] = st.slider("Ancienneté (Années)", 0, 10, 5)
                    input_data['Balance'] = st.number_input("Solde", 0, 200000, 10000)
                    input_data['NumOfProducts'] = st.slider("Produits", 1, 4, 1)
                    input_data['ServiceCount'] = input_data['NumOfProducts']
                else:
                    input_data['tenure'] = st.slider("Ancienneté (Mois)", 0, 72, 12)
                    input_data['MonthlyCharges'] = st.number_input("Mensualité ($)", 0.0, 150.0, 70.0)
                    input_data['ServiceCount'] = st.slider("Services", 0, 8, 3)

                # Convertir en DataFrame avec l'ordre EXACT des colonnes X
                user_df = pd.DataFrame([input_data])[X.columns]
                user_scaled = scaler.transform(user_df)

            with col_out:
                # Entraînement rapide pour la démo
                model = XGBClassifier().fit(X_train_scaled, y_train)
                prob = model.predict_proba(user_scaled)[0][1]
                
                st.metric("Risque de Churn", f"{round(prob*100, 2)}%")
                if prob > 0.5: st.error("🚨 Risque élevé")
                else: st.success("✅ Client Fidèle")
        else:
            st.info("Passez à l'onglet Recherche pour le mode Fusion.")

    with tab2:
        st.header("Analyse Technique")
        model_rf = RandomForestClassifier().fit(X_train_scaled, y_train)
        y_pred = model_rf.predict(X_test_scaled)
        st.write(f"Précision : **{round(accuracy_score(y_test, y_pred)*100, 2)}%**")
        
        # Matrice de confusion
        fig, ax = plt.subplots(figsize=(5,3))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='RdYlGn')
        st.pyplot(fig)

    with tab3:
        st.header("🔬 Fusion des Données (Banque + Telco)")
        df_f, _ = get_data("Fusion")
        st.write(f"Total des données : {df_f.shape[0]} clients analysés.")
        
        # Graphique de comparaison
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='secteur_banque', y='tenure_mois', data=df_f, ax=ax2)
        ax2.set_xticklabels(['Télécom', 'Banque'])
        st.pyplot(fig2)

# --- CONCLUSION ---
st.divider()
st.info("💡 **Conseil Expert** : Les erreurs de type 'Feature Mismatch' sont réglées ici en forçant l'ordre des colonnes `user_df[X.columns]`.")