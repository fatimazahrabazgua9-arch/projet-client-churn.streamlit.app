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
st.set_page_config(page_title="Churn Bank Dashboard", layout="wide")
st.title("🏦 Dashboard de Rétention Client")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv('bank_customer_churn_10k.csv').drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
    df = pd.get_dummies(df, drop_first=True)
    return df

try:
    df = load_and_preprocess()
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- BARRE LATÉRALE : FORMULAIRE ---
    st.sidebar.header("📋 Profil du Nouveau Client")
    def get_user_input():
        data = {
            'CreditScore': st.sidebar.slider("Score de Crédit", 300, 850, 600),
            'Gender': 1 if st.sidebar.selectbox("Genre", ["Homme", "Femme"]) == "Homme" else 0,
            'Age': st.sidebar.slider("Âge", 18, 92, 40),
            'Tenure': st.sidebar.slider("Ancienneté", 0, 10, 5),
            'Balance': st.sidebar.number_input("Solde ($)", 0, 250000, 50000),
            'NumOfProducts': st.sidebar.selectbox("Nombre de produits", [1, 2, 3, 4]),
            'HasCrCard': 1 if st.sidebar.checkbox("Carte de Crédit", True) else 0,
            'IsActiveMember': 1 if st.sidebar.checkbox("Membre Actif", True) else 0,
            'EstimatedSalary': st.sidebar.number_input("Salaire ($)", 0, 200000, 50000),
            'Geo': st.sidebar.selectbox("Pays", ["France", "Germany", "Spain"])
        }
        # Mapping pour l'encodage One-Hot
        data['Geography_Germany'] = 1 if data['Geo'] == "Germany" else 0
        data['Geography_Spain'] = 1 if data['Geo'] == "Spain" else 0
        del data['Geo']
        return pd.DataFrame(data, index=[0])

    input_df = get_user_input()
    input_s = scaler.transform(input_df)

    # --- SELECTION DU MODÈLE ---
    st.write("### 🤖 Analyse par Intelligence Artificielle")
    algo_name = st.selectbox("Sélectionnez l'algorithme :", 
                             ["Random Forest", "XGBoost", "Régression Logistique", "KNN", "SVM"])

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "Régression Logistique": LogisticRegression(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True)
    }

    model = models[algo_name]
    model.fit(X_train_s, y_train)
    
    # --- RÉSULTATS ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prédiction")
        prob = model.predict_proba(input_s)[0][1]
        st.metric("Probabilité de Départ", f"{round(prob * 100, 2)}%")
        
        if prob > 0.5:
            st.error("⚠️ HAUT RISQUE : Le client risque de quitter la banque.")
        else:
            st.success("✅ FIDÈLE : Le client semble vouloir rester.")

    with col2:
        st.subheader("Performance Technique")
        y_pred = model.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Précision du modèle ({algo_name}) : **{round(acc*100, 2)}%**")
        
        # Matrice de Confusion
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax_cm)
        ax_cm.set_xlabel('Prédictions')
        ax_cm.set_ylabel('Réalité')
        st.pyplot(fig_cm)

    # --- IMPORTANCE (Onglet spécifique) ---
    if algo_name in ["Random Forest", "XGBoost"]:
        st.divider()
        st.subheader("📊 Quels facteurs pèsent le plus ?")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
        fig_imp, ax_imp = plt.subplots()
        importances.plot(kind='barh', color='orange', ax=ax_imp)
        st.pyplot(fig_imp)

except FileNotFoundError:
    st.error("Fichier CSV introuvable. Assurez-vous qu'il est dans le même dossier.")