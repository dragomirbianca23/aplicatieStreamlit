import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#set page config
st.set_page_config(
    page_title="HR Attrition Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# incarcare date
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
    if not os.path.exists(file_path):
        st.error(f"Eroare: Fișierul nu a fost găsit la {file_path}.")
        return None
        
    df = pd.read_csv(file_path)
    return df

# sidebar navigare
st.sidebar.title("Navigare proiect")
st.sidebar.markdown('---')

pages = [
    "1. Overview & dataset",
    "2. Tratare valori lipsă & extreme",
    "3. Codificare & scalare",
    "4. Statistici & grupare",
    "5. Machine Learning: clustering",
    "6. Modelare: regresie logistică",
    "7. Modelare: regresie multiplă"
]

selection = st.sidebar.radio("Mergi la:", pages)

st.sidebar.markdown('---')
st.sidebar.info("Proiect Pachete Software Python & SAS")

# incarcare date
df_raw = load_data()

if df_raw is None:
    st.stop()

# curatare date
if 'Unnamed: 0' in df_raw.columns:
    df_raw = df_raw.drop(columns=['Unnamed: 0']) # Apare când CSV-ul a fost salvat cu indexul rândului inclus. Nu e o coloană reală din date.
if 'EmployeeCount' in df_raw.columns:
    df_raw = df_raw.drop(columns=['EmployeeCount']) # Toți angajații sunt Full-Time (1). Nu oferă informație.
if 'Over18' in df_raw.columns:
    df_raw = df_raw.drop(columns=['Over18']) # Toți angajații sunt 'Y'. Nu oferă informație.
if 'StandardHours' in df_raw.columns:
    df_raw = df_raw.drop(columns=['StandardHours']) # Toți angajații au 80 de ore standard. Nu oferă informație.

# functii utilitare
def get_clean_data(df):
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=np.number).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    return df_clean

def encode_categorical(df):
    df_enc = df.copy()
    le = LabelEncoder()
    # Transformam toate coloanele text in numere 
    cat_cols = df_enc.select_dtypes(include='object').columns
    for col in cat_cols:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    return df_enc


# PAGE 1
if selection == "1. Overview & dataset":
    st.title("Analiza Attrition în Resurse Umane")
    st.markdown(""" Această aplicație analizează factorii care contribuie la plecarea angajaților (**Attrition**) folosind pachetele Python: 
    `pandas`, `scikit-learn`, `statsmodels`, `plotly` și metode `streamlit`.
    """)
    
    # METODE STREAMLIT (st.metric)
    st.header("Metrici generale")
    
    col1, col2, col3, col4 = st.columns(4)
    total_employees = len(df_raw)
    attrition_rate = (len(df_raw[df_raw['Attrition'] == 'Yes']) / total_employees) * 100
    avg_age = df_raw['Age'].mean()
    avg_income = df_raw['MonthlyIncome'].mean()
    
    col1.metric("Total angajați", f"{total_employees:,}")
    col2.metric("Rată attrition", f"{attrition_rate:.1f}%")
    col3.metric("Vârstă medie", f"{avg_age:.1f} ani" if pd.notnull(avg_age) else "N/A")
    col4.metric("Venit mediu lunar", f"${avg_income:,.0f}" if pd.notnull(avg_income) else "N/A")
    
    st.markdown("---")
    
    st.header("Previzualizare date")
    
    # Checkbox pentru filtrare
    show_attrition_only = st.checkbox("Arată doar angajații care au plecat (Attrition = Yes)")
    
    if show_attrition_only:
        st.dataframe(df_raw[df_raw['Attrition'] == 'Yes'].head(50), use_container_width=True)
    else:
        st.dataframe(df_raw.head(50), use_container_width=True)
        
    # Info expander
    with st.expander("Detalii set de date"):
        st.write(f"Număr de rânduri: {df_raw.shape[0]}")
        st.write(f"Număr de coloane: {df_raw.shape[1]}")
        st.write("Coloane categoriale: ", list(df_raw.select_dtypes(include='object').columns))
        st.write("Coloane numerice: ", list(df_raw.select_dtypes(include=np.number).columns))
        
    st.subheader("Distribuția vârstei pe departamente")
    fig_age_dept = px.box(df_raw, x="Department", y="Age", color="Attrition", 
                          title="Vârsta vs Departament")
    st.plotly_chart(fig_age_dept, use_container_width=True)


# PAGE 2
elif selection == "2. Tratare valori lipsă & extreme":
    st.title("Tratarea valorilor lipsă și extreme")
    
    st.header("A. Detectarea valorilor lipsă")
    # Calcul valori lipsă
    missing_data = df_raw.isnull().sum()
    missing_data = missing_data[missing_data > 0].reset_index()
    missing_data.columns = ['Coloană', 'Valori Lipsă (Număr)']
    missing_data['Valori Lipsă (%)'] = (missing_data['Valori Lipsă (Număr)'] / len(df_raw)) * 100
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("Tabel sumar:")
        st.dataframe(missing_data)
        
    with col2:
        if not missing_data.empty:
            fig_missing = px.bar(missing_data, x='Coloană', y='Valori Lipsă (Număr)', 
                                 title="Numărul de valori lipsă pe coloane",
                                 text='Valori Lipsă (Număr)', color='Valori Lipsă (%)')
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("Setul de date nu conține valori lipsă!")

    st.markdown("---")
    
    st.header("B. Imputarea valorilor lipsă")
    col_to_impute = st.selectbox("Selectează coloana pentru imputare:", missing_data['Coloană'].tolist() if not missing_data.empty else [])
    
    if col_to_impute:
        df_imputed = df_raw.copy()
        is_numeric = pd.api.types.is_numeric_dtype(df_raw[col_to_impute])
        
        if is_numeric:
            method = st.radio("Metoda de imputare (Numerică):", ["Medie (Mean)", "Mediană (Median)", "Zero"])
            original_mean = df_imputed[col_to_impute].mean()
            
            if method == "Medie (Mean)":
                val = df_imputed[col_to_impute].mean()
            elif method == "Mediană (Median)":
                val = df_imputed[col_to_impute].median()
            else:
                val = 0
                
            df_imputed[col_to_impute] = df_imputed[col_to_impute].fillna(val)
            new_mean = df_imputed[col_to_impute].mean()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Valoare folosită pentru imputare", f"{val:.2f}")
            c2.metric("Media înainte de imputare", f"{original_mean:.2f}")
            c3.metric(f"Media după imputare ({method})", f"{new_mean:.2f}", delta=f"{new_mean - original_mean:.2f}")
            
        else:
            method = st.radio("Metoda de imputare (Categorială):", ["Modul (Cea mai frecventă)", "Valoare Nouă ('Necunoscut')"])
            original_mode = df_imputed[col_to_impute].mode()[0]
            
            if method == "Modul (Cea mai frecventă)":
                val = original_mode
            else:
                val = "Necunoscut"
                
            df_imputed[col_to_impute] = df_imputed[col_to_impute].fillna(val)
            
            c1, c2 = st.columns(2)
            c1.metric("Valoare folosită pentru imputare", str(val))
            c2.metric("Cea mai frecventă categorie originară", str(original_mode))
        
    st.markdown("---")
    
    st.header("C. Detectarea și tratarea valorilor extreme (Outliers)")
    st.write("Metoda utilizată: **IQR (Interquartile Range)** pentru a detecta outlierii (+ IQR Boxplot)")
    
    numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    outlier_col = st.selectbox("Alege variabila (ex. MonthlyIncome, YearsAtCompany):", numeric_cols, index=numeric_cols.index('MonthlyIncome') if 'MonthlyIncome' in numeric_cols else 0)
    
    df_clean = get_clean_data(df_raw) # need clean data for quantiles
    
    Q1 = df_clean[outlier_col].quantile(0.25)
    Q3 = df_clean[outlier_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_clean[(df_clean[outlier_col] < lower_bound) | (df_clean[outlier_col] > upper_bound)]
    
    st.write(f"S-au găsit **{len(outliers)}** outlieri pentru `{outlier_col}` (Limita superioară: {upper_bound:.2f}).")
    
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.subheader("Înainte de tratare (Cu Outlieri)")
        fig_box = px.box(df_clean, y=outlier_col, title=f"Boxplot {outlier_col}")
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col_plot2:
        st.subheader("După plafonare (Capping / Winsorizare)")
        # Capping
        df_capped = df_clean.copy()
        df_capped.loc[df_capped[outlier_col] > upper_bound, outlier_col] = upper_bound
        df_capped.loc[df_capped[outlier_col] < lower_bound, outlier_col] = lower_bound
        
        fig_box2 = px.box(df_capped, y=outlier_col, title=f"Boxplot {outlier_col} (Plafonat)")
        st.plotly_chart(fig_box2, use_container_width=True)
        

# PAGE 3
elif selection == "3. Codificare & scalare":
    st.title("Metode de codificare și scalare")
    
    df_clean = get_clean_data(df_raw)
    
    st.header("A. Codificarea datelor categoriale")
    st.write("Demonstrăm `Label Encoding` și `One-Hot Encoding`.")
    
    cat_columns = df_clean.select_dtypes(include='object').columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Label Encoding")
        st.write("Transformă categorii în numere (0, 1, 2...). Folosit pentru variabile ordinale (ex. Attrition, BusinessTravel).")
        le_col = st.selectbox("Alege coloana", ['Attrition', 'BusinessTravel'])
        
        le = LabelEncoder()
        df_le = pd.DataFrame({
            'Categorie Originală': df_clean[le_col].head(10),
            'Codificare Label': le.fit_transform(df_clean[le_col].head(10))
        })
        st.dataframe(df_le)
        
    with col2:
        st.subheader("2. One-Hot Encoding (Dummies)")
        st.write("Creează o coloană binară pentru fiecare categorie. (ex. Departament)")
        ohe_col = st.selectbox("Alege coloana", ['Department', 'Gender'])
        
        df_ohe = pd.get_dummies(df_clean[[ohe_col]].head(10), drop_first=False)
        st.dataframe(df_ohe)

    st.markdown("---")
    
    st.header("B. Metode de scalare")
    st.write("Scalarea caracteristicilor continue folosind `StandardScaler` (Z-score) și `MinMaxScaler` (0-1).")
    
    scale_col = st.selectbox("Alege variabila pentru demonstrare:", ['MonthlyIncome', 'Age', 'TotalWorkingYears'])
    
    # Original
    orig_data = df_clean[[scale_col]].copy()
    
    # Standard Scaler
    ss = StandardScaler()
    ss_data = ss.fit_transform(orig_data) # Indică la câte deviații standard față de medie se află o valoare.
    
    # MinMax Scaler
    mm = MinMaxScaler()
    mm_data = mm.fit_transform(orig_data) # Scalează datele între 0 și 1, menținând forma distribuției.
    
    df_compare = pd.DataFrame({
        'Original': orig_data[scale_col],
        'StandardScaler (Z-score)': ss_data.flatten(),
        'MinMaxScaler (0-1)': mm_data.flatten()
    }).head(15)
    
    st.dataframe(df_compare.style.format("{:.2f}"))
    
    # Histogram comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df_compare['Original'], ax=axes[0], color='blue', kde=True)
    axes[0].set_title('Original')
    
    sns.histplot(df_compare['StandardScaler (Z-score)'], ax=axes[1], color='red', kde=True)
    axes[1].set_title('Standard Scaler')
    
    sns.histplot(df_compare['MinMaxScaler (0-1)'], ax=axes[2], color='green', kde=True)
    axes[2].set_title('MinMax Scaler')
    
    st.pyplot(fig)


# PAGE 4
elif selection == "4. Statistici & grupare":
    st.title("Prelucrări statistice, grupare și agregare (Pandas)")
    
    df_clean = get_clean_data(df_raw)
    st.markdown("Folosim funcțiile `groupby()`, `agg()` și `apply()` din biblioteca Pandas.")
    
    # 1. Groupby cerință 
    st.header("1. Grupare (Groupby) cu Agregări Multiple")
    
    group_col = st.selectbox("Selectează coloana pentru grupare:", ['Department', 'JobRole', 'EducationField'])
    agg_col = st.selectbox("Statistici asupra coloanei numerice:", ['MonthlyIncome', 'Age', 'YearsAtCompany'])
    
    # Folosim agg() pentru funcții de grup
    grouped_df = df_clean.groupby(group_col).agg(
        Număr_Angajați=(agg_col, 'count'),
        Medie=(agg_col, 'mean'),
        Min=(agg_col, 'min'),
        Max=(agg_col, 'max'),
        Standard_Dev=(agg_col, 'std')
    ).round(2).reset_index()
    
    st.dataframe(grouped_df, use_container_width=True)
    
    # Bar chart
    fig_grp = px.bar(grouped_df, x=group_col, y='Medie', 
                     title=f"Media `{agg_col}` per `{group_col}`",
                     color='Număr_Angajați', color_continuous_scale='Blues')
    st.plotly_chart(fig_grp, use_container_width=True)
    
    st.markdown("---")
    
    st.header("2. Funcții de grup: Transform & Apply")
    
    st.write("Demonstrăm crearea unei noi coloane standardizate în interiorul unui grup (Departament). Exemplu: *Venitul lunar normalizat față de media departamentului (Z-score intragrup)*.")
    
    # Funcție customizată aplicată cu transform
    def group_z_score(x):
        return (x - x.mean()) / x.std()
        
    df_preview = df_clean[['Department', 'JobRole', 'MonthlyIncome']].copy()
    df_preview['Dept_Mean_Income'] = df_preview.groupby('Department')['MonthlyIncome'].transform('mean').round(2)
    df_preview['Income_Z_Score_in_Dept'] = df_preview.groupby('Department')['MonthlyIncome'].transform(group_z_score).round(3)
    
    st.write("Observați cum media departamentului este mapată pentru fiecare individ, iar apoi Z-score-ul este calculat:")
    st.dataframe(df_preview.head(15), use_container_width=True)


# PAGE 5
elif selection == "5. Machine Learning: clustering":
    st.title("Scikit-Learn: Clusterizare (K-Means)")
    
    st.markdown("""
    Căutăm **segmente naturale de angajați** bazat pe venitul lunar și anii petrecuți în companie.
    Acest algoritm de clusterizare nesupervizat ne ajută să descoperim profiluri de angajați.
    """)
    
    df_clean = get_clean_data(df_raw)
    
    # Select features
    features = st.multiselect("Selectează caracteristicile pentru clusterizare (K-Means):", 
                              ['MonthlyIncome', 'YearsAtCompany', 'Age', 'TotalWorkingYears'],
                              default=['MonthlyIncome', 'Age'])
    
    if len(features) >= 2:
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean[features])
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("A. Determinarea K optim (Metoda Cotului / Elbow)")
            
            inertias = []
            k_range = range(1, 11)
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_scaled)
                inertias.append(km.inertia_)
                
            fig_elbow = go.Figure(data=go.Scatter(x=list(k_range), y=inertias, mode='lines+markers'))
            fig_elbow.update_layout(title='Elbow Method', xaxis_title='Număr Clustere (K)', yaxis_title='Inerție')
            st.plotly_chart(fig_elbow, use_container_width=True)
            
            k_selected = st.slider("Alege numărul de clustere K pentru rularea modelului:", 2, 6, 3)
            
        with col2:
            st.subheader(f"B. Rezultatul Clusterizării (K={k_selected})")
            
            # Run model
            kmeans = KMeans(n_clusters=k_selected, random_state=42, n_init=10)
            df_clean['Cluster'] = kmeans.fit_predict(X_scaled)
            df_clean['Cluster'] = df_clean['Cluster'].astype(str)
            
            # Plot
            if len(features) >= 2:
                fig_cluster = px.scatter(df_clean, x=features[0], y=features[1], 
                                         color='Cluster', hover_data=['JobRole', 'Department'],
                                         title=f"Clustere: {features[0]} vs {features[1]}")
                st.plotly_chart(fig_cluster, use_container_width=True)
                
        st.subheader("C. Profilarea Clusterelor (Media variabilelor pe cluster)")
        cluster_profile = df_clean.groupby('Cluster')[features].mean().round(2).reset_index()
        st.dataframe(cluster_profile, use_container_width=True)
        
    else:
        st.warning("Vă rugăm să selectați cel puțin 2 trăsături numerice pentru clusterizare.")


# PAGE 6
elif selection == "6. Modelare: regresie logistică":
    st.title("Regresie logistică")
    
    st.markdown("""
    Putem prezice dacă un angajat va pleca (`Attrition = Yes`) folosind variabilele din setul de date?
    Acesta este un model de antrenare supervizată de clasificare.
    """)
    
    df_clean = get_clean_data(df_raw)
    
    # Pregatire date explicit
    numeric_features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome', 
                        'JobSatisfaction', 'EnvironmentSatisfaction']
    
    st.write("**Caracteristici predictorate alese pentru model:**", numeric_features)
    
    X = df_clean[numeric_features]
    y = (df_clean['Attrition'] == 'Yes').astype(int) # 1 daca pleaca, 0 daca nu
    
    # Scalare obligatorie pentru LogisticRegression - standardizam datele
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # impartire set de date
    test_size = st.slider("Proporție set de test:", 0.1, 0.4, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    
    st.write(f"- Set de antrenare: {len(X_train)} mostre")
    st.write(f"- Set de testare: {len(X_test)} mostre")
    
    if st.button("Antrenează Modelul Logistic"):
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)  # ghicim daca pleaca sau nu
        y_prob = model.predict_proba(X_test)[:, 1] # probabilitatea sa plece
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Matricea de confuzie")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, 
                               x=['Predict rămâne (0)', 'Predict pleacă (1)'], 
                               y=['Actual rămâne (0)', 'Actual pleacă (1)'],
                               color_continuous_scale="Blues_r")
            st.plotly_chart(fig_cm)
            
        with col2:
            st.subheader("2. Curba ROC")
            roc_auc = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC-AUC Score = {roc_auc:.3f}', mode='lines'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random Guess (0.5)', line=dict(dash='dash'), mode='lines'))
            fig_roc.update_layout(xaxis_title='Rată Fals Pozitive', yaxis_title='Rată Adevărat Pozitive')
            st.plotly_chart(fig_roc)
            
        st.subheader("3. Importanța Caracteristicilor (Coeficienți B)")
        coef_df = pd.DataFrame({
            'Caracteristică': numeric_features,
            'Coeficient': model.coef_[0]
        })
        coef_df['Impact Absolut'] = coef_df['Coeficient'].abs()
        coef_df = coef_df.sort_values(by='Impact Absolut', ascending=False)
        
        fig_coef = px.bar(coef_df, x='Caracteristică', y='Coeficient',
                          color='Coeficient', color_continuous_scale='RdBu_r',
                          title="Efectul caracteristicilor asupra șanselor de Attrition")
        st.plotly_chart(fig_coef)


# Page 7
elif selection == "7. Modelare: regresie multiplă":
    st.title("Statsmodels: Regresie Liniară Multiplă")
    
    st.markdown("""
    **Problema:** Dorim să explicăm variația Venitului Lunar (`MonthlyIncome`) folosind variabile independente. Pentru a obține p-values și statistici detaliate, folosim pachetul `statsmodels.api` (metoda OLS - Ordinary Least Squares).
    """)
    
    df_clean = get_clean_data(df_raw)
    
    target_var = 'MonthlyIncome'
    
    st.write("**Alege factorii (variabile independente X):**")
    cols_available = df_clean.select_dtypes(include=np.number).columns.tolist()
    if target_var in cols_available:
        cols_available.remove(target_var)
        
    default_predictors = ['JobLevel', 'TotalWorkingYears', 'YearsAtCompany', 'Age']
    predictors = st.multiselect("Predictori:", cols_available, default=default_predictors)
    
    if len(predictors) > 0:
        # Cast explicit la float (pentru a evita erorile ValueError generate de Statsmodels pe coloane 'object')
        X = df_clean[predictors].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float64')
        y = pd.to_numeric(df_clean[target_var], errors='coerce').fillna(0).astype('float64')
        
        # adaugam constanta
        X_with_const = sm.add_constant(X)
        
        # modelul de regresie
        model = sm.OLS(y, X_with_const)
        results = model.fit()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("R-squared (Varianta Explicată)", f"{results.rsquared:.3f}")
            st.metric("Adjusted R-squared", f"{results.rsquared_adj:.3f}")
            
        with col2:
            st.metric("Prob (F-statistic)", f"{results.f_pvalue:.4e}")
            
        st.subheader("Sumar Model (OLS Summary)")
        st.text(results.summary().as_text())
        
        # grafic pred vs real
        st.subheader("Grafic: Preziceri vs Valori Reale")
        df_pred = pd.DataFrame({
            'Actual': y,
            'Predicted': results.predict(X_with_const)
        })
        
        fig_scatter = px.scatter(df_pred, x='Actual', y='Predicted', opacity=0.5,
                                 title="Predictii Model Liniar vs Venit Lunar Real")
        fig_scatter.add_shape(type="line", line=dict(dash='dash'), x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Vă rugăm să selectați cel puțin un predictor.")
