import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
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

        # Definir valores fixos para alpha
        alpha_lasso = 0.01
        alpha_ridge = 1.0
        alpha_elastic = 0.1
        l1_ratio_elastic = 0.5

        # Configurar modelos Lasso, Ridge, ElasticNet e Decision Tree
        lasso = Lasso(alpha=alpha_lasso)
        ridge = Ridge(alpha=alpha_ridge)
        elastic = ElasticNet(alpha=alpha_elastic, l1_ratio=l1_ratio_elastic)
        tree = DecisionTreeRegressor(random_state=45)

        # Ajustar os modelos
        lasso.fit(X_train, y_train)
        ridge.fit(X_train, y_train)
        elastic.fit(X_train, y_train)
        tree.fit(X_train, y_train)

        # Avaliar os modelos ajustados
        y_pred_lr = lasso.predict(X_test)
        y_pred_ridge = ridge.predict(X_test)
        y_pred_elastic = elastic.predict(X_test)
        y_pred_tree = tree.predict(X_test)

        mse_lr = mean_squared_error(y_test, y_pred_lr)
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)
        mse_elastic = mean_squared_error(y_test, y_pred_elastic)
        mse_tree = mean_squared_error(y_test, y_pred_tree)

        r2_lr = r2_score(y_test, y_pred_lr)
        r2_ridge = r2_score(y_test, y_pred_ridge)
        r2_elastic = r2_score(y_test, y_pred_elastic)
        r2_tree = r2_score(y_test, y_pred_tree)

        st.subheader("Avaliação dos Modelos Ajustados")
        st.write(f"**Modelo Lasso Regression (Alpha: {alpha_lasso}):**")
        st.write(f"   Mean Squared Error (MSE): {mse_lr:.2f}")
        st.write(f"   R-squared (R2): {r2_lr:.2f}")

        st.write(f"**Modelo Ridge Regression (Alpha: {alpha_ridge}):**")
        st.write(f"   Mean Squared Error (MSE): {mse_ridge:.2f}")
        st.write(f"   R-squared (R2): {r2_ridge:.2f}")

        st.write(f"**Modelo Elastic Net Regression (Alpha: {alpha_elastic}, L1 Ratio: {l1_ratio_elastic}):**")
        st.write(f"   Mean Squared Error (MSE): {mse_elastic:.2f}")
        st.write(f"   R-squared (R2): {r2_elastic:.2f}")

        st.write(f"**Modelo Decision Tree Regression**")
        st.write(f"   Mean Squared Error (MSE): {mse_tree:.2f}")
        st.write(f"   R-squared (R2): {r2_tree:.2f}")

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
        predicted_energy_output_lasso = lasso.predict(input_data_scaled)[0]
        predicted_energy_output_ridge = ridge.predict(input_data_scaled)[0]
        predicted_energy_output_elastic = elastic.predict(input_data_scaled)[0]
        predicted_energy_output_tree = tree.predict(input_data_scaled)[0]

        st.subheader("Previsões de Produção de Energia")
        st.markdown(f"### **Modelo Lasso Regression**", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>Previsão de Produção de Energia Elétrica: <span style='color: red;'>{predicted_energy_output_lasso:.2f} MW</span></h1>", unsafe_allow_html=True)
        
        st.markdown(f"### **Modelo Ridge Regression**", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>Previsão de Produção de Energia Elétrica: <span style='color: red;'>{predicted_energy_output_ridge:.2f} MW</span></h1>", unsafe_allow_html=True)

        st.markdown(f"### **Modelo Elastic Net Regression**", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>Previsão de Produção de Energia Elétrica: <span style='color: red;'>{predicted_energy_output_elastic:.2f} MW</span></h1>", unsafe_allow_html=True)

        st.markdown(f"### **Modelo Decision Tree Regression**", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>Previsão de Produção de Energia Elétrica: <span style='color: red;'>{predicted_energy_output_tree:.2f} MW</span></h1>", unsafe_allow_html=True)

        # Previsão mensal com valores aleatórios
        st.subheader("Previsão Mensal com Valores Aleatórios")
        days = np.arange(1, 31)
        temperatures = np.random.uniform(temperature - 5, temperature + 5, size=30)
        vacuums = np.random.uniform(vacuum - 10, vacuum + 10, size=30)
        pressures = np.random.uniform(pressure - 20, pressure + 20, size=30)
        humidities = np.random.uniform(humidity - 20, humidity + 20, size=30)

        monthly_data = pd.DataFrame({
            'Avg temperature': temperatures,
            'Exhaust vacuum': vacuums,
            'Ambient pressure': pressures,
            'Relative humidity': humidities
        })

        monthly_data_scaled = scaler.transform(monthly_data)

        predictions_lasso = lasso.predict(monthly_data_scaled)
        predictions_ridge = ridge.predict(monthly_data_scaled)
        predictions_elastic = elastic.predict(monthly_data_scaled)
        predictions_tree = tree.predict(monthly_data_scaled)

        fig, ax = plt.subplots()
        ax.plot(days, predictions_lasso, label='Lasso Regression', color='blue')
        ax.plot(days, predictions_ridge, label='Ridge Regression', color='red')
        ax.plot(days, predictions_elastic, label='Elastic Net Regression', color='purple')
        ax.plot(days, predictions_tree, label='Decision Tree Regression', color='green')
        ax.set_xlabel('Dia do Mês')
        ax.set_ylabel('Produção de Energia (MW)')
        ax.set_title('Previsão de Produção de Energia Elétrica Mensal')
        ax.legend()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo CSV: {e}")

if __name__ == "__main__":
    main()
