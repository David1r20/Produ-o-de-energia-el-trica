import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Função principal do Streamlit
@st.cache_data
def main():
    st.title("Previsão de Produção de Energia Elétrica")

    # URL do arquivo CSV no GitHub (Use a URL raw)
    url = 'https://raw.githubusercontent.com/David1r20/repository/main/Power_data.csv'  # Substitua pela URL correta

    
    if uploaded_file is not None:
        data = pd.read_csv(url)

        # Mostrar os primeiros registros do dataframe
        st.write("Primeiros registros do dataframe:")
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

        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"R-squared (R2): {r2}")
        st.write(f"Coeficientes do modelo: {model.coef_}")
        st.write(f"Intercepto do modelo: {model.intercept_}")

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

        # Mostrar a previsão
        st.write(f"A previsão de produção de energia elétrica é: {predicted_energy_output:.2f} MW")

if __name__ == "__main__":
    main()
