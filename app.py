import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# --- CONFIGURATION INITIALE ---
st.set_page_config(page_title="IA Churn Pro", layout="wide")

# --- GESTION DU CACHE ET DES ERREURS MÉMOIRE ---
if 'last_sector' not in st.session_state:
    st.session_state