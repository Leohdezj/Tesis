# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 22:37:31 2024

@author: LeoHdezJ
"""

import numpy as np
# Create matrices
strike = 1.10
r = 0.06
option_type = "Put"
# Stock Price Paths from Longstaff-Schwartz 
# (Otherwise created w/ Monte-Carlo Simulations of Brownian Motion)
t3 = [1.34,1.54,1.03,.92,1.52,.9,1.01,1.34]
t2 = [1.08,1.26,1.07,.97,1.56,.77,.84,1.22]
t1 = [1.09,1.16,1.22,.93,1.11,.76,.92,.88]
t0 = [1,1,1,1,1,1,1,1]
paths = np.array([t3,t2,t1,t0])
paths

cash_flows = np.zeros_like(paths)

if option_type == "Call":
    for i in range(0,cash_flows.shape[0]):
        cash_flows[i] = [max(round(x - strike,2),0) for x in paths[i]]
else:
    for i in range(0,cash_flows.shape[0]):
        cash_flows[i] = [max(-round(x - strike,2),0) for x in paths[i]]

discounted_cash_flows = np.zeros_like(cash_flows)
cash_flows

from sklearn.linear_model import LinearRegression

# Create index to only look at in the money paths at time t
in_the_money =paths[1,:] < strike

# Run Regression
X = (paths[1,in_the_money])
X2 = X*X
Xs = np.column_stack([X,X2])
Y = cash_flows[1-1,in_the_money]  * np.exp(-r)
model_sklearn = LinearRegression()
model = model_sklearn.fit(Xs, Y)
conditional_exp = model.predict(Xs)
continuations = np.zeros_like(paths[1,:])
continuations[in_the_money] = conditional_exp



# # First rule: If continuation is greater in t =0, then cash flow in t=1 is zero
cash_flows[1,:] = np.where(continuations> cash_flows[1,:], 0, cash_flows[1,:])
cash_flows


# If stopped ahead of time, subsequent cashflows = 0
exercised_early = continuations < cash_flows[1, :]
cash_flows[0:1, :][:, exercised_early] = 0
discounted_cash_flows[0,:] = cash_flows[0,:]* np.exp(-r * 3)


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def longstaff_schwartz(paths, strike, r,option_type):
    cash_flows = np.zeros_like(paths)
    if option_type == "Call":
        for i in range(0,cash_flows.shape[0]):
            cash_flows[i] = [max(round(x - strike,2),0) for x in paths[i]]
    else:
        for i in range(0,cash_flows.shape[0]):
            cash_flows[i] = [max(-round(x - strike,2),0) for x in paths[i]]
    discounted_cash_flows = np.zeros_like(cash_flows)


    T = cash_flows.shape[0]-1

    for t in range(1,T):
        
        # Look at time t+1
        # Create index to only look at in the money paths at time t
        in_the_money =paths[t,:] < strike

        # Run Regression
        X = (paths[t,in_the_money])
        X2 = X*X
        Xs = np.column_stack([X,X2])
        Y = cash_flows[t-1,in_the_money]  * np.exp(-r)
        model_sklearn = LinearRegression()
        model = model_sklearn.fit(Xs, Y)
        conditional_exp = model.predict(Xs)
        continuations = np.zeros_like(paths[t,:])
        continuations[in_the_money] = conditional_exp

        # # First rule: If continuation is greater in t =0, then cash flow in t=1 is zero
        cash_flows[t,:] = np.where(continuations> cash_flows[t,:], 0, cash_flows[t,:])

        # 2nd rule: If stopped ahead of time, subsequent cashflows = 0
        exercised_early = continuations < cash_flows[t, :]
        cash_flows[0:t, :][:, exercised_early] = 0
        discounted_cash_flows[t-1,:] = cash_flows[t-1,:]* np.exp(-r * 3)

    discounted_cash_flows[T-1,:] = cash_flows[T-1,:]* np.exp(-r * 1)


    # Return final option price
    final_cfs = np.zeros((discounted_cash_flows.shape[1], 1), dtype=float)
    for i,row in enumerate(final_cfs):
        final_cfs[i] = sum(discounted_cash_flows[:,i])
    option_price = np.mean(final_cfs)
    
    cash_flows=cash_flows.transpose()
    return option_price,cash_flows


def simulate_gbm(mu, sigma, S0, T, dt, num_paths):
    num_steps = int(T / dt) + 1
    times = np.linspace(0, T, num_steps)
    paths = np.zeros(( num_steps,num_paths))

    for i in range(num_paths):
        # Generate random normal increments
        dW = np.random.normal(0, np.sqrt(dt), num_steps - 1)
        # Calculate the cumulative sum of increments
        cumulative_dW = np.cumsum(dW)
        # Calculate the stock price path using the GBM formula
        paths[ 1:,i] = S0 * np.exp((mu - 0.5 * sigma**2) * times[1:] + sigma * cumulative_dW)
    return paths

np.random.seed(99)
# Parameters
mu = 0.00  # Drift (average return per unit time)
sigma = 0.2  # Volatility (standard deviation of the returns)
S0 = 1  # Initial stock price
T = 1  # Total time period (in years)
dt = 1/3  # Time increment (daily simulation)
num_paths = 10  # Number of simulation paths

# Simulate stock price paths
paths = simulate_gbm(mu, sigma, S0, T, dt, num_paths)
paths[0,:]=S0
paths = paths[::-1]

T,D=longstaff_schwartz(paths = paths, strike =1.1, r = 0.06, option_type="Put")




#
# Replica del ejercicio del libro
# ==============================================================================
import pandas as pd
import numpy as np

strike = 1.10
r = 0.06
option_type = "Put"
# Stock Price Paths from Longstaff-Schwartz 
# (Otherwise created w/ Monte-Carlo Simulations of Brownian Motion)
t3 = [1.34,1.54,1.03,.92,1.52,.9,1.01,1.34]
t2 = [1.08,1.26,1.07,.97,1.56,.77,.84,1.22]
t1 = [1.09,1.16,1.22,.93,1.11,.76,.92,.88]
t0 = [1,1,1,1,1,1,1,1]

paths =  pd.DataFrame((zip(t0, t1, t2,t3)))
paths=paths.to_numpy()

cash_flows = np.zeros_like(paths)

if option_type == "Call":
    for i in range(0,cash_flows.shape[0]):
        cash_flows[i] = [max(round(x - strike,2),0) for x in paths[i]]
else:
    for i in range(0,cash_flows.shape[0]):
        cash_flows[i] = [max(-round(x - strike,2),0) for x in paths[i]]
        
        

discounted_cash_flows = np.zeros_like(cash_flows)
cash_flows

from sklearn.linear_model import LinearRegression

decision = np.zeros_like(paths)
temp=cash_flows.copy()
for i in range(len(paths[1,:])-2,0,-1):
# Create index to only look at in the money paths at time t
    in_the_money =paths[:,i] < strike
    
    # Run Regression
    X = (paths[in_the_money,i])
    X2 = X*X
    Xs = np.column_stack([X,X2])
    Y = temp[in_the_money,i+1]  * np.exp(-r )
    model_sklearn = LinearRegression()
    model = model_sklearn.fit(Xs, Y)
    conditional_exp = model.predict(Xs)
    continuations = np.zeros_like(paths[:,1])
    continuations[in_the_money] = conditional_exp
    
    # # First rule: If continuation is greater in t =0, then cash flow in t=1 is zero
    decision[:,i+1] = np.where(continuations <temp[:,i], 0, temp[:,i+1])
    decision[:,i] = np.where(continuations > temp[:,i], 0, temp[:,i])
    
    exercised_early = continuations > temp[:,i]
    temp[ exercised_early,i] = 0
    

option_valuation = []

# Iterar sobre cada fila de la matriz
for row in decision:
    # Encontrar las columnas donde los valores son diferentes de cero para cada fila
    no_zero_columns = np.where(row != 0)[0]
    # Si no hay valores diferentes de cero, asignar 0 a la lista de índices
    if len(no_zero_columns) == 0:
       no_zero_columns = [-1]
    # Agregar los resultados a la lista
    option_valuation.append(no_zero_columns[0])
final_cfs = np.zeros((decision.shape[0], 1), dtype=float)
for i,row in enumerate(final_cfs):
    final_cfs[i] = sum(decision[i,:])
    option_price = np.mean(final_cfs)




in_the_money =paths[:,2] < strike

# Run Regression
X = (paths[in_the_money,2])
X2 = X*X
Xs = np.column_stack([X,X2])
Y = temp[in_the_money,3]  * np.exp(-r )
model_sklearn = LinearRegression()
model = model_sklearn.fit(Xs, Y)
conditional_exp = model.predict(Xs)
continuations = np.zeros_like(paths[:,1])
continuations[in_the_money] = conditional_exp

# # First rule: If continuation is greater in t =0, then cash flow in t=1 is zero
decision[:,3] = np.where(continuations< temp[:,2], 0, temp[:,3])



exercised_early = continuations > temp[:,2]
temp[exercised_early,2] = 0



in_the_money =paths[:,1] < strike

# Run Regression
X = (paths[in_the_money,1])
X2 = X*X
Xs = np.column_stack([X,X2])
Y = temp[in_the_money,2]  * np.exp(-r )
model_sklearn = LinearRegression()
model = model_sklearn.fit(Xs, Y)
conditional_exp = model.predict(Xs)
continuations = np.zeros_like(paths[:,1])
continuations[in_the_money] = conditional_exp

# # First rule: If continuation is greater in t =0, then cash flow in t=1 is zero
decision[:,2] = np.where(continuations< temp[:,1], 0, cash_flows[:,2])

exercised_early = continuations > temp[:,1]
temp[exercised_early,1] = 0




