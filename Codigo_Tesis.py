# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:45:19 2024

@author: Leohdezj
"""


import pandas as pd
import numpy as np
import yfinance as yf
import time

# Preprocesado y modelado
# ==============================================================================
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#----------------------             FUNCIONES              -------------------------------
def tasa_libre_riesgo(start_date, end_date):
    # Símbolo del bono del Tesoro a 10 años en Estados Unidos
    symbol = "^TNX"
    
    # Crear un objeto Ticker con el símbolo especificado
    tnx = yf.Ticker(symbol)
    
    # Obtener el historial de precios de cierre ajustados dentro del rango de fechas
    historial_precios = tnx.history(start=start_date, end=end_date)
    
    # Último precio de cierre ajustado (rendimiento)
    rateRiskFree = historial_precios['Close'].iloc[-1]
    return rateRiskFree

def tasa_ret_anual_precio_act(lista, start_date, end_date):
    if not isinstance(lista, list):
        lista = [lista]  # Convertirlo en una lista
    
    # Crear un DataFrame vacío para almacenar los resultados
    df = pd.DataFrame(columns=['symbol', 'S0', 'mu'])
    
    for i in lista:
        # Crear el objeto Ticker con el símbolo correspondiente
        ticker = yf.Ticker(i)
        
        # Obtener el precio actual (último precio ajustado)
        price = ticker.history(period='1d')['Close'].iloc[0]
        
        # Obtener el historial de precios en el rango de fechas especificado
        data = ticker.history(start=start_date, end=end_date)
        data = data.reset_index()
        
        # Crear un DataFrame con las fechas y los precios ajustados
        retorn = pd.DataFrame({'Date': data['Date'], 'Price': data['Close']})
        
        # Calcular los retornos logarítmicos
        retorn['Log_Returns'] = np.log(retorn['Price'] / retorn['Price'].shift(1))
        
        # Calcular la deriva anualizada (drift)
        drift = retorn['Log_Returns'].mean() * 252  # Aproximadamente 252 días de mercado por año
        
        # Agregar los resultados al DataFrame
        temp = [i, price, drift]
        df.loc[len(df)] = temp

    return df


     
def obtener_opciones(portfolio, start_date, end_date):
    if not isinstance(portfolio, list):
        portfolio = [portfolio]  # Convertirlo en una lista
    
    # Crear una lista para almacenar los DataFrames de las opciones
    df_list = []
    
    for symbol in portfolio:
        ticker = yf.Ticker(symbol)
        options = ticker.option_chain()  # Obtiene las opciones (calls y puts)
        df_calls = options.calls
        df_calls['optionType'] = 'calls'
        
        df_puts = options.puts
        df_puts['optionType'] = 'puts'
        
        # Unificamos los 'calls' y 'puts' en un único DataFrame
        df = pd.concat([df_calls, df_puts], ignore_index=True)
        
        # Filtrar por las fechas de negociación
        df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate'])
        df = df[(df['lastTradeDate'] >= start_date) & (df['lastTradeDate'] <= end_date)]
        
        # Filtramos las opciones 'inTheMoney'
        df['inTheMoney'] = df.apply(
            lambda row: (row['optionType'] == 'calls' and row['lastPrice'] > row['strike']) or
                        (row['optionType'] == 'puts' and row['lastPrice'] < row['strike']),
            axis=1
        )
        
        df=df[(df['inTheMoney'] == True) & (df['impliedVolatility'] <3)]
        
        # Añadir la columna del símbolo
        df['symbol'] = symbol
        
        # Crear una columna con la fecha corta
        df['shortDate'] = df['lastTradeDate'].dt.strftime('%Y-%m-%d')
        
        # Añadir el DataFrame a la lista
        df_list.append(df)
    
    # Concatenamos todos los DataFrames obtenidos
    final_df = pd.concat(df_list, ignore_index=True)
    final_df = final_df.reset_index(drop=True)
    return final_df
 


def condicional(symbol,option_type, strike):
    x=inf_portfolio.loc[inf_portfolio['symbol']==symbol].iloc[0]['S0']
    if option_type == "calls":
        return max(round(x - strike,2),0)
    else:
        return max(-round(x - strike,2),0) 
    

def is_itm(row):
    if row['optionType']== 'calls':
        return 'ITM' if row['lastPrice']>row['strike'] else 'OTM'
    elif row['optionType']== 'puts':
        return 'ITM' if row['lastPrice']<row['strike'] else 'OTM'
    

#-----------------------------------------------------------------------------------------
def simulate_gbm(mu, sigma, s0, T, dt, num_paths):
    np.random.seed(99)
    num_steps = int(T / dt) + 1
 
    times = np.linspace(0, T, num_steps)
    paths = np.zeros(( num_paths,num_steps))

    for i in range(num_paths):
        # Generate random normal increments
        dW = np.random.normal(0, np.sqrt(dt), num_steps - 1)
        # Calculate the cumulative sum of increments
        cumulative_dW = np.cumsum(dW)
        # Calculate the stock price path using the GBM formula
        paths[i,1:] = s0 * np.exp((mu - 0.5 * sigma**2) * times[1:] + sigma * cumulative_dW)
    return paths


def longstaff_schwartz(paths, strike, r,option_type):
    cash_flows = np.zeros_like(paths)
    
    
    if option_type == "calls":
        for i in range(0,cash_flows.shape[0]):
            cash_flows[i] = [max(round(x - strike,2),0) for x in paths[i]]
    else:
        for i in range(0,cash_flows.shape[0]):
            cash_flows[i] = [max(-round(x - strike,2),0) for x in paths[i]]
    

    
    
    T = cash_flows.shape[1]-1

    for t in range(T,0,-1):
        
        # Look at time t+1
        # Create index to only look at in the money paths at time t
        if option_type == "calls":
            in_the_money =paths[:,t-1] > strike
        else:
            in_the_money =paths[:,t-1] < strike
            
        if len(paths[in_the_money,t-1])>0:
            # Run Regression
            X = (paths[in_the_money,t-1])
            X=X.reshape(-1,1)
            Y = cash_flows[in_the_money,t]  * np.exp(-r)
            model_sklearn = SVR(kernel="rbf", C=1e1, gamma=0.1)
            model = model_sklearn.fit(X, Y)
            conditional_exp = model.predict(X)
            continuations = np.zeros_like(paths[:,t])
            continuations[in_the_money] = conditional_exp
        else:
            continuations = np.zeros_like(paths[:,t])
        # # First rule: If continuation is greater in t =0, then cash flow in t=1 is zero
        cash_flows[:,t] = np.where(continuations> cash_flows[:,t], 0, cash_flows[:,t])
    
        # 2nd rule: If stopped ahead of time, subsequent cashflows = 0
        exercised_early = continuations < cash_flows[:,t]
        cash_flows[:,0:t][exercised_early,:] = 0
    

    decision=pd.DataFrame(cash_flows)  
    option_valuation = decision.idxmax(axis=1)
    option_valuation = option_valuation.tolist()
    option_valuation= pd.DataFrame( option_valuation )
    
    vector_valuation = option_valuation.T 

    option_valuation = option_valuation[option_valuation[0] != 0]
    # Calcular la moda de los valores diferentes de 0
    
    option_valuation = option_valuation.mode()[0][0] if  not option_valuation.empty else 0
    
    final_cfs = np.zeros((cash_flows.shape[0], 1), dtype=float)
    for i,row in enumerate(final_cfs):
        final_cfs[i] = max(cash_flows[i,:])
        option_price = np.mean(final_cfs)
    return option_price,option_valuation,vector_valuation




    
def price_value_op(symbol,optionType,strike,impliedVolatility,lastPrice):
    #S0=inf_portfolio.loc[inf_portfolio['symbol']==symbol].iloc[0]['S0']
    mu=inf_portfolio.loc[inf_portfolio['symbol']==symbol].iloc[0]['mu']
    
    paths = simulate_gbm(mu, impliedVolatility, lastPrice, T, dt, num_paths)
    option_price,option_valuation,vector_valuation=longstaff_schwartz(paths = paths, strike =strike, r = rateRiskFree, option_type=optionType)
    
    return option_price,option_valuation,vector_valuation

def obtain_option_values(row):
    # Llamar a la función 'price_value_op' para obtener los tres valores
    option_price, option_valuation, vector_valuation = price_value_op(
        row['symbol'], row['optionType'], row['strike'], row['impliedVolatility'], row['lastPrice']
    )
    
    # Devolver los tres valores como un pandas.Series
    # Esto retornará 'optionPrice', 'optionValuation', y las columnas del 'vector_valuation'
    result = pd.Series([option_price, option_valuation] + vector_valuation.iloc[0].tolist())
    
    return result



def convertir_a_string(valor):
    return str(valor)

# Aplica la función a cada elemento de la columna 'B' usando map

#------------------------------------Variables Globales------------------------------------------------#


T = 1  # Total time period (in years)
dt = 1/1000 # Time increment (daily simulation)
num_paths = 100  # Number of simulation paths
start_date, end_date="2023-01-01", "2025-12-01"
rateRiskFree  = tasa_libre_riesgo(start_date, end_date)




#----------------------------------------Prueba con una sola accion -----------------------------------------------------------------#

portfolio ="AAPL"
inf_portfolio=tasa_ret_anual_precio_act(portfolio,start_date, end_date)
# Símbolo de un activo (por ejemplo, AAPL)
df= obtener_opciones(portfolio,start_date, end_date)


df=df[df['optionType']=='puts']
df = df.reset_index()

df=df[(df['inTheMoney'] == True) & (df['impliedVolatility'] <3)]

df=df.tail(1)
df = df.reset_index()

impliedVolatility=df.loc[0, 'impliedVolatility']
r=rateRiskFree
S0=	df.loc[0, 'lastPrice']
mu=inf_portfolio.loc[inf_portfolio['symbol']== 'AAPL'].iloc[0]['mu']

paths = simulate_gbm(mu, impliedVolatility, S0, T, dt, num_paths)
strike=	df.loc[0, 'strike']


optionType=df.loc[0, 'optionType']

cash_flows = np.zeros_like(paths)



rateRiskFree  = tasa_libre_riesgo(start_date, end_date)



if optionType == "calls":
    for i in range(0,cash_flows.shape[0]):
        cash_flows[i] = [max(round(x - strike,2),0) for x in paths[i]]
else:
    for i in range(0,cash_flows.shape[0]):
        cash_flows[i] = [max(-round(x - strike,2),0) for x in paths[i]]

discounted_cash_flows = np.zeros_like(cash_flows)


T_num = cash_flows.shape[1]-1
     
for t in range(T_num,0,-1):

    # Look at time t
    # Create index to only look at in the money paths at time t-1
    if optionType == "calls":
        in_the_money =paths[:,t-1] > strike
    else:
        in_the_money =paths[:,t-1] < strike
        
    if len(paths[in_the_money,t-1])>0:
        # Run Regression
        X = (paths[in_the_money,t-1])
        X=X.reshape(-1,1)
        Y = cash_flows[in_the_money,t]  * np.exp(-r)
        model_sklearn = SVR(kernel="rbf", C=1e1, gamma=0.1)
        model = model_sklearn.fit(X, Y)
        conditional_exp = model.predict(X)
        continuations = np.zeros_like(paths[:,t])
        continuations[in_the_money] = conditional_exp
    else:
        continuations = np.zeros_like(paths[:,t])
    # # First rule: If continuation is greater in t =0, then cash flow in t=1 is zero
    cash_flows[:,t] = np.where(continuations> cash_flows[:,t], 0, cash_flows[:,t])

    # 2nd rule: If stopped ahead of time, subsequent cashflows = 0
    exercised_early = continuations < cash_flows[:,t]
    cash_flows[:,0:t][exercised_early,:] = 0



decision=pd.DataFrame(cash_flows)  
option_valuation = decision.idxmax(axis=1)
option_valuation = option_valuation.tolist()
option_valuation= pd.DataFrame( option_valuation )

vector_valuation = option_valuation.T

option_valuation = option_valuation[option_valuation[0] != 0]
# Calcular la moda de los valores diferentes de 0

option_valuation = option_valuation.mode()[0][0] if  not option_valuation.empty else 0

final_cfs = np.zeros((cash_flows.shape[0], 1), dtype=float)
for i,row in enumerate(final_cfs):
    final_cfs[i] = max(cash_flows[i,:])
    option_price = np.mean(final_cfs)



df['symbol'] = df['symbol'].map(convertir_a_string)

df['cashFlows'] = list(map(condicional, df['symbol'], df['optionType'], df['strike']))
# Crear un DataFrame temporal con todos los valores
new_cols = df.apply(obtain_option_values, axis=1, result_type="expand")

# Crear las nuevas columnas con los nombres correspondientes
new_cols.columns = ['optionPrice', 'optionValuation'] + [f'vector_valuation_{i}' for i in range(num_paths)]

# Concatenar las nuevas columnas con el DataFrame original
df = pd.concat([df, new_cols], axis=1)
valores_distintos = df['optionValuation'].unique()

#----------------------------------------Vector de acciones -----------------------------------------------------------------#


portfolio = ['META', 'AMZN', 'AAPL', 'NFLX','GOOG' ]


inf_portfolio=tasa_ret_anual_precio_act(portfolio,start_date, end_date)

df= obtener_opciones(portfolio,start_date, end_date)

df['cashFlows'] = list(map(condicional, df['symbol'], df['optionType'], df['strike']))
# Crear un DataFrame temporal con todos los valores
new_cols = df.apply(obtain_option_values, axis=1, result_type="expand")

# Crear las nuevas columnas con los nombres correspondientes
new_cols.columns = ['optionPrice', 'optionValuation'] + [f'vector_valuation_{i}' for i in range(num_paths)]

# Concatenar las nuevas columnas con el DataFrame original
df = pd.concat([df, new_cols], axis=1)
valores_distintos = df['optionValuation'].unique()


#----------------------------------------Prueba con una accion para graficar -----------------------------------------------------------------#

df1=df[df['symbol']=='AMZN'].copy()


import matplotlib.pyplot as plt

x= range (1, len(df1)+1)
plt.plot(x, df1['optionValuation'], marker ='o',color='b', linestyle='-')

plt.title('Grafico de linea')
plt.xlabel('indice')
plt.ylabel('valor')

plt.show()


#------------------------------------Mensaje nuevo ------------------------------------------------#