# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:45:19 2024

@author: Leohdezj
"""


import pandas as pd
import numpy as np
from yahooquery import Ticker
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
def tasa_libre_riesgo():
    # Símbolo del bono del Tesoro a 10 años en Estados Unidos
    symbol = "^TNX"
    # Crear un objeto Ticker con el símbolo especificado
    tnx = Ticker(symbol)
    # Obtener el historial de precios de cierre
    historial_precios = tnx.history(period='1y')
     # Último precio de cierre (rendimiento)
    rateRiskFree  = historial_precios['adjclose'].iloc[-1]
    return rateRiskFree

def tasa_ret_anual_precio_act(lista):
    
    df=pd.DataFrame(columns=['symbol','S0','mu'])
    for i in lista:
        ticker = Ticker(i)
        price=ticker.history(period='1d')['adjclose'][0]
        
        data = ticker.history(period='1y')
        data = data.reset_index()
        retorn = pd.DataFrame({'Date': data['date'], 'Price': data['adjclose']})
    
        # Calcular los retornos logarítmicos
        retorn['Log_Returns'] = np.log(retorn['Price'] / retorn['Price'].shift(1))
    
        # Calcular la deriva (drift)
        drift = retorn['Log_Returns'].mean() / retorn.shape[0]
        
        temp=[i, price, drift ]
        df.loc[len(df)] = temp

    return df
     

 


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
    

    N = cash_flows.shape[1]-1
         
    for t in range(N,0,-1):
        # Look at time t+1
        # Create index to only look at in the money paths at time t
        if option_type == "calls":
            in_the_money =paths[:,t] > strike
        else:
            in_the_money =paths[:,t] < strike
            
        if len(paths[in_the_money,t])>0:
            # Run Regression
            X = (paths[in_the_money,t])
            X=X.reshape(-1,1)
            Y = cash_flows[in_the_money,t-1]  * np.exp(-r)
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
    
    
    option_valuation = option_valuation[option_valuation[0] != 0]
    # Calcular la moda de los valores diferentes de 0
    
    option_valuation = option_valuation.mode()[0][0] if  not option_valuation.empty else 0
    
    final_cfs = np.zeros((cash_flows.shape[0], 1), dtype=float)
    for i,row in enumerate(final_cfs):
        final_cfs[i] = max(cash_flows[i,:])
        option_price = np.mean(final_cfs)
    
    return option_price,option_valuation



    
def price_value_op(symbol,optionType,strike,impliedVolatility,lastPrice):
    #S0=inf_portfolio.loc[inf_portfolio['symbol']==symbol].iloc[0]['S0']
    mu=inf_portfolio.loc[inf_portfolio['symbol']==symbol].iloc[0]['mu']
    
    paths = simulate_gbm(mu, impliedVolatility, lastPrice, T, dt, num_paths)
    option_price,option_valuation=longstaff_schwartz(paths = paths, strike =strike, r = rateRiskFree, option_type=optionType)
    
    return option_price,option_valuation



def convertir_a_string(valor):
    return str(valor)

# Aplica la función a cada elemento de la columna 'B' usando map

#------------------------------------Variables Globales------------------------------------------------#


T = 1  # Total time period (in years)
dt = 1/10 # Time increment (daily simulation)
num_paths = 10  # Number of simulation paths

rateRiskFree  = tasa_libre_riesgo()




#----------------------------------------Prueba con una sola accion -----------------------------------------------------------------#

portfolio = [ 'AAPL' ]
inf_portfolio=tasa_ret_anual_precio_act(portfolio)

t = Ticker(portfolio, asynchronous=True)
df = pd.DataFrame(t.option_chain)
df=df[(df['inTheMoney'] == True) & (df['impliedVolatility'] <3) & (df['impliedVolatility'] > 0.0001)][['contractSymbol','strike','lastPrice','currency','impliedVolatility','inTheMoney']]
df = df.reset_index()

df['inTheMoney'] = df.apply(
    lambda row: (row['optionType'] == 'calls' and row['lastPrice'] > row['strike']) or
                (row['optionType'] == 'puts' and row['lastPrice'] < row['strike']),
    axis=1
)


df=df[df['inTheMoney']==True]
df=df[df['optionType']=='calls']
df = df.reset_index()


df=df.head(1)

impliedVolatility=df.loc[0, 'impliedVolatility']
r=rateRiskFree
S0=	df.loc[0, 'lastPrice']
mu=inf_portfolio.loc[inf_portfolio['symbol']== 'AAPL'].iloc[0]['mu']
T = 1  # Total time period (in years)
dt = 1/10 # Time increment (daily simulation)
num_paths = 10  # Number of simulation paths
paths = simulate_gbm(mu, impliedVolatility, S0, T, dt, num_paths)
strike=	df.loc[0, 'strike']


optionType=df.loc[0, 'optionType']

cash_flows = np.zeros_like(paths)



rateRiskFree  = tasa_libre_riesgo()


if optionType == "calls":
    for i in range(0,cash_flows.shape[0]):
        cash_flows[i] = [max(round(x - strike,2),0) for x in paths[i]]
else:
    for i in range(0,cash_flows.shape[0]):
        cash_flows[i] = [max(-round(x - strike,2),0) for x in paths[i]]



N = cash_flows.shape[1]-1
     
for t in range(N,0,-1):
    # Look at time t+1
    # Create index to only look at in the money paths at time t
    if optionType == "calls":
        in_the_money =paths[:,t] > strike
    else:
        in_the_money =paths[:,t] < strike
        
    if len(paths[in_the_money,t])>0:
        # Run Regression
        X = (paths[in_the_money,t])
        X=X.reshape(-1,1)
        Y = cash_flows[in_the_money,t-1]  * np.exp(-r)
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


option_valuation = option_valuation[option_valuation[0] != 0]
# Calcular la moda de los valores diferentes de 0

option_valuation = option_valuation.mode()[0][0] if  not option_valuation.empty else 0

final_cfs = np.zeros((cash_flows.shape[0], 1), dtype=float)
for i,row in enumerate(final_cfs):
    final_cfs[i] = max(cash_flows[i,:])
    option_price = np.mean(final_cfs)



df['symbol'] = df['symbol'].map(convertir_a_string)

df['cashFlows'] = list(map(condicional, df['symbol'], df['optionType'], df['strike']))
df[['optionPrice','optionValuation']]=pd.DataFrame(list(map(price_value_op, df['symbol'], df['optionType'], df['strike'], df['impliedVolatility'], df['lastPrice'])))
valores_distintos = df['optionValuation'].unique()

#----------------------------------------Vector de acciones -----------------------------------------------------------------#


portfolio = ['META', 'AMZN', 'AAPL', 'NFLX','GOOG' ]
inf_portfolio=tasa_ret_anual_precio_act(portfolio)

t = Ticker(portfolio, asynchronous=True)
df = pd.DataFrame(t.option_chain)
df=df[(df['inTheMoney'] == True) & (df['impliedVolatility'] <3& (df['impliedVolatility'] > 0.0001))][['contractSymbol','strike','lastPrice','currency','impliedVolatility','inTheMoney']]
df = df.reset_index()


df['inTheMoney'] = df.apply(
    lambda row: (row['optionType'] == 'calls' and row['lastPrice'] > row['strike']) or
                (row['optionType'] == 'puts' and row['lastPrice'] < row['strike']),
    axis=1
)


df=df[df['inTheMoney']==True]
df = df.reset_index()

df['symbol'] = df['symbol'].map(convertir_a_string)

df['cashFlows'] = list(map(condicional, df['symbol'], df['optionType'], df['strike']))
df[['optionPrice','optionValuation']]=pd.DataFrame(list(map(price_value_op, df['symbol'], df['optionType'], df['strike'], df['impliedVolatility'], df['lastPrice'])))
valores_distintos = df['optionValuation'].unique()


#----------------------------------------Prueba con una accion para graficar -----------------------------------------------------------------#

df1=df[df['symbol']=='AAPL'].copy()


import matplotlib.pyplot as plt

x= range (1, len(df1)+1)
plt.plot(x, df1['optionValuation'], marker ='o',color='b', linestyle='-')

plt.title('Grafico de linea')
plt.xlabel('indice')
plt.ylabel('valor')

plt.show()


#------------------------------------Mensaje nuevo ------------------------------------------------#