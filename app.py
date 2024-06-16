import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

        # Dividir os dados em conjuntos de treino e teste (80% treino, 20% teste)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Criar e treinar o modelo de regressão linear
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Fazer previsões no conjunto de teste
        y_pred = model.predict(X_test)

        # Avaliar o modelo
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Avaliação do Modelo")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R-squared (R2):** {r2:.2f}")
        st.write(f"**Coeficientes do Modelo:** {model.coef_}")
        st.write(f"**Intercepto do Modelo:** {model.intercept_}")

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

        # Fazer previsão com os dados de entrada
        predicted_energy_output = model.predict(input_data)[0]

        st.subheader("Previsão de Produção de Energia")
        st.write(f"**A previsão de produção de energia elétrica é:** {predicted_energy_output:.2f} MW")
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo CSV: {e}")

if __name__ == "__main__":
    main()
