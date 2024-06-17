import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# URL do arquivo CSV no GitHub
url = 'https://raw.githubusercontent.com/David1r20/Produ-o-de-energia-el-trica/main/Power_data.csv'

data = pd.read_csv(url)

def main():
    st.title("Previsão de Produção de Energia Elétrica")

    try:
        # Mostrar os primeiros registros do dataframe
        st.subheader("Dados do Dataset")
        st.write(data.head())

        # Definir variáveis independentes (X) e dependente (y)
        X = data[['Avg temperature', 'Exhaust vacuum', 'Ambient pressure', 'Relative humidity']]
        y = data['Net hourly electrical energy output']

        # Normalizar os dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Dividir os dados em conjuntos de treino e teste (80% treino, 20% teste)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.08, random_state=45)

        # Grid de parâmetros para Lasso e Ridge com sequência logarítmica
        alpha_values = np.logspace(-4, 1, 10)  # Gera 10 valores logaritmicamente espaçados entre 0.0001 e 10

        param_grid_lasso = {'alpha': alpha_values}
        param_grid_ridge = {'alpha': alpha_values}

        # Configurar GridSearchCV para Lasso
        lasso = Lasso()
        grid_search_lasso = GridSearchCV(lasso, param_grid_lasso, cv=5, scoring='neg_mean_squared_error')
        grid_search_lasso.fit(X_train, y_train)
        best_lasso = grid_search_lasso.best_estimator_

        # Configurar GridSearchCV para Ridge
        ridge = Ridge()
        grid_search_ridge = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='neg_mean_squared_error')
        grid_search_ridge.fit(X_train, y_train)
        best_ridge = grid_search_ridge.best_estimator_

        # Avaliar os modelos ajustados
        y_pred_lr = grid_search_lasso.predict(X_test)
        y_pred_ridge = grid_search_ridge.predict(X_test)
        
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)

        r2_lr = r2_score(y_test, y_pred_lr)
        r2_ridge = r2_score(y_test, y_pred_ridge)

        st.subheader("Avaliação dos Modelos Ajustados")
        st.write(f"**Modelo Lasso Regression (Melhor Alpha: {grid_search_lasso.best_params_['alpha']}):**")
        st.write(f"   Mean Squared Error (MSE): {mse_lr:.2f}")
        st.write(f"   R-squared (R2): {r2_lr:.2f}")

        st.write(f"**Modelo Ridge Regression (Melhor Alpha: {grid_search_ridge.best_params_['alpha']}):**")
        st.write(f"   Mean Squared Error (MSE): {mse_ridge:.2f}")
        st.write(f"   R-squared (R2): {r2_ridge:.2f}")

        # Widgets para entrada de parâmetros de previsão
        st.sidebar.header("Parâmetros de Previsão")
        temperature = st.sidebar.slider("Temperatura Média (°C)", min_value=0, max_value=40, value=25)
        vacuum = st.sidebar.slider("Pressão de Vácuo (cm Hg)", min_value=25, max_value=80, value=55)
        pressure = st.sidebar.slider("Pressão Ambiente (mbar)", min_value=900, max_value=1100, value=1010)
        humidity = st.sidebar.slider("Umidade Relativa (%)", min_value=0, max_value=100, value=50)

        # Criar um dataframe com os dados de entrada
        input_data = pd.DataFrame({
            'Avg temperature': [temperature],
            'Exhaust vacuum': [vacuum],
            'Ambient pressure': [pressure],
            'Relative humidity': [humidity]
        })

        # Normalizar os dados de entrada
        input_data_scaled = scaler.transform(input_data)

        # Fazer previsões com os modelos ajustados
        predicted_energy_output_lasso = best_lasso.predict(input_data_scaled)[0]
        predicted_energy_output_ridge = best_ridge.predict(input_data_scaled)[0]

        st.subheader("Previsões de Produção de Energia")
        st.markdown(f"### **Modelo Lasso Regression**", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color: red;'>Previsão de Produção de Energia Elétrica: {predicted_energy_output_lasso:.2f} MW</h1>", unsafe_allow_html=True)
        
        st.markdown(f"### **Modelo Ridge Regression**", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color: red;'>Previsão de Produção de Energia Elétrica: {predicted_energy_output_ridge:.2f} MW</h1>", unsafe_allow_html=True)

        # Exemplos de previsões com gráficos
        st.subheader("Exemplos de Previsões")
        examples = [
            {"Avg temperature": 20, "Exhaust vacuum": 40, "Ambient pressure": 1000, "Relative humidity": 60},
            {"Avg temperature": 30, "Exhaust vacuum": 50, "Ambient pressure": 950, "Relative humidity": 70},
            {"Avg temperature": 25, "Exhaust vacuum": 55, "Ambient pressure": 1010, "Relative humidity": 50},
            {"Avg temperature": 15, "Exhaust vacuum": 35, "Ambient pressure": 980, "Relative humidity": 80}
        ]

        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        for i, (example, ax_row) in enumerate(zip(examples, ax.flatten()), start=1):
            input_data_example = pd.DataFrame([example])
            input_data_scaled_example = scaler.transform(input_data_example)
            pred_lasso_example = best_lasso.predict(input_data_scaled_example)[0]
            pred_ridge_example = best_ridge.predict(input_data_scaled_example)[0]
            
            ax_row.plot([0, 1], [pred_lasso_example, pred_ridge_example], marker='o')
            ax_row.set_title(f"Exemplo {i}")
            ax_row.set_xticks([0, 1])
            ax_row.set_xticklabels(['Lasso', 'Ridge'])
            ax_row.set_ylabel("Produção de Energia (MW)")
            ax_row.set_ylim([0, max(pred_lasso_example, pred_ridge_example) + 10])
            ax_row.grid(True)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro ao carregar o arquivo CSV: {e}")

if __name__ == "__main__":
    main()
