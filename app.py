import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# URL do arquivo CSV no GitHub
url = 'https://raw.githubusercontent.com/David1r20/Produ-o-de-energia-el-trica/main/Power_data.csv'
data = pd.read_csv(url)

def main():
    st.title("Previsão de Produção de Energia Elétrica")

    st.write("""
    Este conjunto de dados contém dados operacionais de uma usina de energia, detalhando vários fatores ambientais e operacionais,
    juntamente com a produção líquida de energia elétrica por hora. Será analisado a influência das condições ambientais no
    desempenho da usina e pode ser usado para modelagem preditiva e estudos de otimização.
    
    **Características:**
    - Temperatura média: Temperatura ambiente média (em Celsius).
    - Vácuo de exaustão: Pressão de vácuo do vapor que sai da turbina (em cm Hg).
    - Pressão ambiente: Pressão ambiente (em milibares).
    - Umidade relativa: Umidade relativa (%).
    - Produção líquida de energia elétrica horária: Produção líquida de energia elétrica horária (em MW).
    
    **Uso:**
    Este conjunto de dados será usado para análise de regressão Lasso e Ridge.
    """)

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

        # Configurar modelos Lasso e Ridge
        lasso = Lasso(alpha=alpha_lasso)
        ridge = Ridge(alpha=alpha_ridge)

        # Ajustar os modelos
        lasso.fit(X_train, y_train)
        ridge.fit(X_train, y_train)

        # Avaliar os modelos ajustados
        y_pred_lr = lasso.predict(X_test)
        y_pred_ridge = ridge.predict(X_test)

        mse_lr = mean_squared_error(y_test, y_pred_lr)
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)

        r2_lr = r2_score(y_test, y_pred_lr)
        r2_ridge = r2_score(y_test, y_pred_ridge)

        st.subheader("Avaliação dos Modelos Ajustados")
        st.write(f"**Modelo Lasso Regression (Alpha: {alpha_lasso}):**")
        st.write(f"   Mean Squared Error (MSE): {mse_lr:.2f}")
        st.write(f"   R-squared (R2): {r2_lr:.2f}")

        st.write(f"**Modelo Ridge Regression (Alpha: {alpha_ridge}):**")
        st.write(f"   Mean Squared Error (MSE): {mse_ridge:.2f}")
        st.write(f"   R-squared (R2): {r2_ridge:.2f}")

        # Widgets para entrada de parâmetros de previsão mensal
        st.sidebar.header("Parâmetros de Previsão Mensal")
        temperature_range = st.sidebar.slider("Temperatura Média (°C)", min_value=0, max_value=40, value=(20, 30))
        vacuum_range = st.sidebar.slider("Pressão de Vácuo (cm Hg)", min_value=25, max_value=80, value=(40, 60))
        pressure_range = st.sidebar.slider("Pressão Ambiente (mbar)", min_value=900, max_value=1100, value=(950, 1050))
        humidity_range = st.sidebar.slider("Umidade Relativa (%)", min_value=0, max_value=100, value=(40, 60))

        # Gerar previsões para um mês (30 dias)
        days = np.arange(1, 31)
        temperatures = np.random.uniform(temperature_range[0], temperature_range[1], size=30)
        vacuums = np.random.uniform(vacuum_range[0], vacuum_range[1], size=30)
        pressures = np.random.uniform(pressure_range[0], pressure_range[1], size=30)
        humidities = np.random.uniform(humidity_range[0], humidity_range[1], size=30)

        monthly_data = pd.DataFrame({
            'Avg temperature': temperatures,
            'Exhaust vacuum': vacuums,
            'Ambient pressure': pressures,
            'Relative humidity': humidities
        })

        monthly_data_scaled = scaler.transform(monthly_data)

        # Previsões
        predictions_lasso = lasso.predict(monthly_data_scaled)
        predictions_ridge = ridge.predict(monthly_data_scaled)

        # Plotar resultados
        fig, ax = plt.subplots()
        ax.plot(days, predictions_lasso, label='Lasso Regression', color='blue')
        ax.plot(days, predictions_ridge, label='Ridge Regression', color='red')
        ax.set_xlabel('Dia do Mês')
        ax.set_ylabel('Produção de Energia (MW)')
        ax.set_title('Previsão de Produção de Energia Elétrica Mensal')
        ax.legend()
        st.pyplot(fig)

        # Mostrar previsões reais
        st.subheader("Previsões para Conjunto de Teste")
        st.write(f"**Modelo Lasso Regression (Alpha: {alpha_lasso}):**")
        st.write(f"   Previsão Média: {np.mean(y_pred_lr):.2f} MW")
        st.write(f"**Modelo Ridge Regression (Alpha: {alpha_ridge}):**")
        st.write(f"   Previsão Média: {np.mean(y_pred_ridge):.2f} MW")
        
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo CSV: {e}")

if __name__ == "__main__":
    main()
