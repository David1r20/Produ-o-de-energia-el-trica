import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# URL do arquivo CSV no GitHub
url = 'https://raw.githubusercontent.com/David1r20/Produ-o-de-energia-el-trica/main/Power_data.csv'
data = pd.read_csv(url)

def main():
    st.title("Previsão de Produção de Energia Elétrica")
    
    # Introdução ao Streamlit
    st.header("Visão Geral da Aplicação")
    st.write("""
    Este conjunto de dados contém informações detalhadas sobre as condições operacionais de uma usina de energia, 
    incluindo aspectos ambientais como temperatura média, pressão de vácuo, pressão ambiente e umidade relativa, 
    além da produção líquida de energia elétrica por hora. A análise desses dados é crucial para entender como 
    cada variável afeta a geração de energia e para desenvolver modelos preditivos que possam ajudar na 
    otimização do desempenho da usina.
    
    ### Características do Conjunto de Dados:
    - **Temperatura Média:** Temperatura ambiente média (em Celsius).
    - **Vácuo de Exaustão:** Pressão de vácuo do vapor que sai da turbina (em cm Hg).
    - **Pressão Ambiente:** Pressão ambiente (em milibares).
    - **Umidade Relativa:** Umidade relativa (%).
    - **Produção Líquida de Energia Elétrica Horária:** Produção líquida de energia elétrica horária (em MW).
    
    ### Aplicação do Código:
    Este código realiza uma análise de regressão para prever a produção 
    de energia elétrica com base nas condições ambientais e operacionais. O modelo é treinado com dados normalizados, 
    e os usuários podem ajustar os parâmetros para prever a produção de energia em diferentes cenários. 
    Além disso, o código gera previsões mensais com valores aleatórios, proporcionando uma visão abrangente das 
    possíveis variações na produção de energia ao longo do tempo.
    """)
    
    try:
        st.subheader("Dados do Dataset")
        st.write(data.head())
    
        # Visualização das distribuições das variáveis
        st.subheader("Distribuição das Variáveis")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        sns.histplot(data['Avg temperature'], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Distribuição da Temperatura Média')
        sns.histplot(data['Exhaust vacuum'], kde=True, ax=axes[0, 1])
        axes[0, 1].set_title('Distribuição do Vácuo de Exaustão')
        sns.histplot(data['Ambient pressure'], kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Distribuição da Pressão Ambiente')
        sns.histplot(data['Relative humidity'], kde=True, ax=axes[1, 1])
        axes[1, 1].set_title('Distribuição da Umidade Relativa')
        st.pyplot(fig)
    
        # Matriz de correlação com mapa de calor
        st.subheader("Matriz de Correlação")
        corr_matrix = data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Mapa de Calor da Matriz de Correlação')
        st.pyplot(fig)
        
        
        # Definir variáveis independentes (X) e dependente (y)
        X = data[['Avg temperature', 'Exhaust vacuum', 'Ambient pressure', 'Relative humidity']]
        y = data['Net hourly electrical energy output']
    
        # Normalizar os dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
        # Dividir os dados em conjuntos de treino e teste (80% treino, 20% teste)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=45)
    
        # Definir valores fixos para alpha e parâmetros dos modelos
        alpha_lasso = 0.01
        alpha_ridge = 0.2
        alpha_elastic = 0.01
        l1_ratio_elastic = 0.0
    
        # Configurar modelos Lasso, Ridge e ElasticNet
        lasso = Lasso(alpha=alpha_lasso)
        ridge = Ridge(alpha=alpha_ridge)
        elastic = ElasticNet(alpha=alpha_elastic, l1_ratio=l1_ratio_elastic)
    
        # Ajustar os modelos
        lasso.fit(X_train, y_train)
        ridge.fit(X_train, y_train)
        elastic.fit(X_train, y_train)
    
        # Avaliar os modelos ajustados
        y_pred_lr = lasso.predict(X_test)
        y_pred_ridge = ridge.predict(X_test)
        y_pred_elastic = elastic.predict(X_test)
    
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)
        mse_elastic = mean_squared_error(y_test, y_pred_elastic)
    
        r2_lr = r2_score(y_test, y_pred_lr)
        r2_ridge = r2_score(y_test, y_pred_ridge)
        r2_elastic = r2_score(y_test, y_pred_elastic)

            # Gráfico de Previsão vs Real
        st.subheader("Previsão vs Real")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot para Lasso
        ax.scatter(y_test, y_pred_lr, color='blue', label='Lasso', alpha=0.6)
        
        # Scatter plot para Ridge
        ax.scatter(y_test, y_pred_ridge, color='red', label='Ridge', alpha=0.6)
        
        # Scatter plot para ElasticNet
        ax.scatter(y_test, y_pred_elastic, color='green', label='ElasticNet', alpha=0.6)
        
        # Adicionar linhas de referência
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
        
        # Configurações do gráfico
        ax.set_xlabel('Produção Real (MW)')
        ax.set_ylabel('Produção Prevista (MW)')
        ax.set_title('Comparação entre Produção Real e Produção Prevista')
        ax.legend()
        st.pyplot(fig)
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
    
        st.write("""
        Os modelos Lasso, Ridge e Elastic Net apresentaram um desempenho excelente na previsão da produção de 
        energia elétrica, com valores de Mean Squared Error (MSE) próximos, variando de 21.10 a 21.18, e um R-squared (R2) 
        de 0.93, indicando que eles conseguem explicar aproximadamente 93% da variabilidade nos dados. O modelo Lasso e o Ridge mostraram 
        um desempenho idêntico, enquanto o Elastic Net, com uma leve ênfase em regularização L2, teve um MSE ligeiramente maior. Em resumo, 
        todos os modelos são eficazes e bem ajustados, mostrando alta precisão e capacidade de previsão para as condições ambientais e operacionais analisadas.
        """)
    
        # Widgets para entrada de parâmetros de previsão
        st.sidebar.header("Parâmetros de Previsão")
        temperature = st.sidebar.slider("Temperatura Média (°C)", min_value=0, max_value=60, value=25)
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
    
        st.subheader("**Previsões de Produção de Energia em Tempo Real**")
        st.markdown(f"### **Modelo Lasso Regression**", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>Previsão: <span style='color: red;'>{predicted_energy_output_lasso:.2f} MWh</span></h1>", unsafe_allow_html=True)
        st.markdown(f"### **Modelo Ridge Regression**", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>Previsão: <span style='color: red;'>{predicted_energy_output_ridge:.2f} MWh</span></h1>", unsafe_allow_html=True)
    
        st.markdown(f"### **Modelo Elastic Net Regression**", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>Previsão: <span style='color: red;'>{predicted_energy_output_elastic:.2f} MWh</span></h1>", unsafe_allow_html=True)
        
        # Seção de previsão mensal
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
    
        fig, ax = plt.subplots()
        ax.plot(days, predictions_lasso, label='Lasso Regression', color='blue')
        ax.plot(days, predictions_ridge, label='Ridge Regression', color='red')
        ax.plot(days, predictions_elastic, label='Elastic Net Regression', color='grey')
        ax.set_xlabel('Dia do Mês')
        ax.set_ylabel('Produção de Energia (MW)')
        ax.set_title('Previsão de Produção de Energia Elétrica Mensal')
        ax.legend()
        
        # Mostrar gráfico no Streamlit
        st.pyplot(fig)
        # Conclusão
        st.subheader("Conclusão")
        st.write("""
        Após realizar a análise e treinamento dos modelos de regressão Lasso, Ridge e ElasticNet, observamos que todos os modelos 
        apresentaram um desempenho satisfatório na previsão da produção de energia elétrica. As métricas de avaliação, como o Mean 
        Squared Error (MSE) e o R-squared (R2), indicam que os modelos são capazes de explicar aproximadamente 93% da variabilidade 
        nos dados, com valores de MSE próximos entre 21.10 e 21.18. Isso demonstra uma alta precisão e capacidade de previsão para 
        as condições ambientais e operacionais analisadas.
        
        Adicionalmente, a visualização das distribuições das variáveis e a matriz de correlação ajudaram a entender melhor a 
        relação entre as variáveis, contribuindo para a escolha e ajuste dos modelos. A análise de previsão mensal com valores 
        aleatórios forneceu uma perspectiva sobre possíveis variações na produção de energia ao longo do tempo, enquanto o gráfico 
        de previsão vs real mostrou a acuracidade das previsões em comparação com os valores reais.
        
        Em resumo, os modelos Lasso, Ridge e ElasticNet são eficazes e bem ajustados, mostrando-se adequados para aplicações 
        práticas na previsão da produção de energia elétrica com base em variáveis ambientais e operacionais.
        """)
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo CSV: {e}")

if __name__ == "__main__":
    main()
        
