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
st.set_page_config(page_title="Churn Multi-Secteurs & Recherche", layout="wide")

# --- FONCTION DE CHARGEMENT ---
@st.cache_data
def get_data(mode):
    # Chargement des fichiers bruts
    df_bank_raw = pd.read_csv('bank_customer_churn_10k.csv')
    df_telco_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
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
        df['ChargeTenureRatio'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        df_encoded = pd.get_dummies(df, drop_first=True)
        return df_encoded, 'Churn'
    
    else: # MODE FUSION (RECHERCHE)
        # Standardisation Banque
        bank_univ = pd.DataFrame({
            'tenure_mois': df_bank_raw['Tenure'] * 12,
            'nb_services': df_bank_raw['NumOfProducts'],
            'secteur_banque': 1,
            'target': df_bank_raw['Exited']
        })
        # Standardisation Telco
        serv_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        telco_univ = pd.DataFrame({
            'tenure_mois': df_telco_raw['tenure'],
            'nb_services': (df_telco_raw[serv_cols] == 'Yes').sum(axis=1),
            'secteur_banque': 0,
            'target': df_telco_raw['Churn'].map({'Yes': 1, 'No': 0})
        })
        return pd.concat([bank_univ, telco_univ], axis=0), 'target'

# --- INTERFACE ---
st.sidebar.title("🧬 Intelligence Artificielle")
mode_select = st.sidebar.selectbox("Mode d'analyse :", ["Banque", "Télécom", "Fusion (Recherche)"])

try:
    df, target = get_data(mode_select)
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.title(f"🚀 Plateforme Churn : Mode {mode_select}")

    tab1, tab2, tab3 = st.tabs(["🎯 Prédiction Client", "📊 Performance", "🔬 Recherche Universelle"])

    with tab1:
        if mode_select != "Fusion (Recherche)":
            col_in, col_out = st.columns([1, 2])
            with col_in:
                st.subheader("Profil")
                inputs = {col: 0 for col in X.columns}
                if mode_select == "Banque":
                    inputs['Age'] = st.slider("Âge", 18, 92, 40)
                    inputs['tenure'] = st.slider("Ancienneté (Années)", 0, 10, 5)
                    inputs['NumOfProducts'] = st.selectbox("Produits", [1, 2, 3, 4])
                else:
                    inputs['tenure'] = st.slider("Ancienneté (Mois)", 0, 72, 24)
                    inputs['MonthlyCharges'] = st.number_input("Charges ($)", 0.0, 150.0, 60.0)
                
                user_df = pd.DataFrame(inputs, index=[0])
            
            with col_out:
                model = XGBClassifier().fit(X_train, y_train)
                prob = model.predict_proba(user_df)[0][1]
                st.metric("Risque", f"{round(prob*100, 2)}%")
                if prob > 0.5: st.error("⚠️ Risque de départ")
                else: st.success("✅ Client fidèle")
        else:
            st.info("Utilisez l'onglet 'Recherche Universelle' pour ce mode.")

    with tab2:
        st.header("Analyse des Modèles")
        if st.button("Evaluer la précision"):
            model_rf = RandomForestClassifier().fit(X_train, y_train)
            acc = accuracy_score(y_test, model_rf.predict(X_test))
            st.write(f"Précision Random Forest : **{round(acc*100, 2)}%**")
            fig, ax = plt.subplots(figsize=(5,3))
            sns.heatmap(confusion_matrix(y_test, model_rf.predict(X_test)), annot=True, fmt='d', cmap='Blues')
            st.pyplot(fig)

    with tab3:
        st.header("🔬 Étude de la Fusion des Données")
        df_f, t_f = get_data("Fusion")
        st.write(f"Nombre total de données fusionnées : **{df_f.shape[0]} lignes**")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Moyenne d'ancienneté par secteur (en mois)**")
            st.bar_chart(df_f.groupby('secteur_banque')['tenure_mois'].mean())
        with c2:
            st.write("**Moyenne de services par secteur**")
            st.bar_chart(df_f.groupby('secteur_banque')['nb_services'].mean())

        

        st.divider()
        st.write("### 🧠 Prédiction Transversale")
        st.write("Ce modèle apprend des deux secteurs simultanément.")
        model_univ = XGBClassifier().fit(df_f.drop('target', axis=1), df_f['target'])
        
        test_tenure = st.slider("Ancienneté client (Mois)", 0, 120, 24, key="univ_ten")
        test_serv = st.slider("Services possédés", 1, 8, 2, key="univ_serv")
        test_sect = st.radio("Secteur cible", ["Télécom (0)", "Banque (1)"])
        sect_val = 1 if "Banque" in test_sect else 0
        
        res_univ = model_univ.predict_proba([[test_tenure, test_serv, sect_val]])[0][1]
        st.subheader(f"Probabilité de Churn Universelle : {round(res_univ*100, 2)}%")

except Exception as e:
    st.error(f"Veuillez vérifier vos fichiers CSV : {e}")