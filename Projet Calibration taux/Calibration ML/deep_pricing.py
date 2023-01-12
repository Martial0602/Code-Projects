import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import estim_param_toolbox as EPT
import sys
from matplotlib.pyplot import figure
from itertools import product
from math import *
from scipy.stats import norm
from scipy import interpolate
import scipy.optimize as opt
from keras.utils.vis_utils import plot_model

def conversion_temps(temps):
    if temps=='DY':
        temps=1
    elif temps=='WK':
        temps=7
    elif temps=='MO':
        temps=30
    elif temps=='YR':
        temps=360
    else :
        print("erreur conversion_temps")
    return temps

#def prix_ZC_market(data, taux_ZC_market, T):
#    if (T>2):delta=1    
#    elif T==2 or T==3/2: delta=1/2
#    else: delta=1/12
#    if (T-delta>=1/12): return np.exp(-delta*taux_ZC_market)*prix_ZC_market(data, get_ZC_rate_market(data,T-delta),T-delta)
#    else: return 1
    
    
def prix_ZC_market(data, taux_ZC_market, T):
    i_index = (data["T"]<=T).sum()-1
    T_i = data.loc[i_index,"T"]
    delta = T- T_i
    return np.exp(-delta*taux_ZC_market)*data.loc[i_index,"prix_ZC"]


#def prix_ZC_market(data,taux_ZC_market,T):
#    return (1/(1+taux_ZC_market))**(T)
    
def get_ZC_rate_market(data,T_i):#T_i convertir en fraction d'année 
    #interpolation si T_i < max T, extrapolation flat sinon
    if T_i <= max(data["T"]):
        f1 = interpolate.interp1d(data["T"], data["Mid"],kind = 'linear')
        return f1(T_i)
    return data["Mid"].iat[-1]
    #dans la data, nous avons des taux ZC pour certaine date, 
    #pour les T_i faire une interpolation

def level_market(t,T_0,T_N, dt, data):
    #T_0 -> date de maturité 
    # T_1 = T_0 + 6M (les datas que j'ai envoyé sont des datas de 6M)
    #...
    # T_N = Maturité + Tenor 
    # level = somme ∑ P(t,T_i) * dt
    # P(t,T_i) = prix_ZC_market( get_ZC_rate_market(data,T_i) ,T_i )
    T_i=T_0
    level=0
    while T_i+dt<=(T_0+T_N) :
        level=level+(dt*prix_ZC_market( data,get_ZC_rate_market(data,T_i), T_i ))
        T_i = T_i + dt
    return  level

def Taux_swap_market(t,T_0,T_N,dt,data):
    # ( P(t,T_0 )-P(t,T_N ) )/ level_market 
    #formule 3.1.12 de la documentation
    level=level_market(t,T_0,T_N, dt, data)
    P_T0=prix_ZC_market(data,get_ZC_rate_market(data,T_0), T_0)
    P_TN=prix_ZC_market(data,get_ZC_rate_market(data,T_N+T_0), T_N+T_0)
    Taux_swap_market=(P_T0 - P_TN) / level
    return Taux_swap_market

#problème quand taux swap négatif pour d1
#problème division par t=0
def d1_black(taux_swap, K, maturite, sigma):# S -> taux swap
    #print(taux_swap,K,maturite,sigma)
    #print(taux_swap,sigma)
    return (np.log(taux_swap/K)+(maturite*sigma**2)/2 )/(sigma*np.sqrt(maturite))
                                        
def d2_black(maturite, sigma,d1):
    return d1-(sigma*np.sqrt(maturite))

def d_bachelier(taux_swap, K, maturite, sigma):
    return (taux_swap - K)/ (sigma *np.sqrt(maturite))

def PS_bachelier(taux_swap,level,d ,K,T_0,sigma, w = 1):
    return level*(taux_swap-K)*w*norm.cdf(w*d, loc=0, scale=1) + sigma*np.sqrt(T_0)*norm.pdf(d)

def recup_sigma(data_sigma, T_0, T_N):
    sigma=float(data_sigma.loc[(data_sigma["Maturite"]==T_0) & (data_sigma["Tenor"]==T_N),"Sigma"].values[0])
    #print(sigma)
    sigma=sigma/10000
    return sigma

def PS_normal(taux_swap,level,d_1,d_2, K): # prix swaption market, j'ai rajouté des paramètres
    #     (3.1.18)
    print(norm.cdf(d_1, loc=0, scale=1),norm.cdf(d_2, loc=0, scale=1))
    PS_norm=level*(taux_swap*norm.cdf(d_1, loc=0, scale=1)-K*norm.cdf(d_2, loc=0, scale=1))
    return PS_norm

def PS_market_normal(t,T_0,T_N,dt,K,data_sigma,data):
    taux_swap = Taux_swap_market(t,T_0,T_N,dt,data)
    sigma = recup_sigma(data_sigma, T_0, T_N)
    d_1 = d1_black(taux_swap, K, T_0, sigma)  ##pb????, on a pas SS
    d_2 = d2_black(T_0, sigma,d_1)
    level=level_market(t,T_0,T_N, dt, data)
    #print(taux_swap,level,sigma,d_1)
    return PS_normal(taux_swap,level,d_1,d_2,K)

def PS_market_bachelier(t,T_0,T_N,dt,K,data_sigma,data):
    taux_swap = Taux_swap_market(t,T_0,T_N,dt,data)
    sigma = recup_sigma(data_sigma, T_0, T_N)
    d = d_bachelier(taux_swap, K, T_0, sigma)  ##pb????, on a pas SS
    level=level_market(t,T_0,T_N, dt, data)
    #print(taux_swap,level,sigma,d_1)
    return PS_bachelier(taux_swap,level,d,K,T_0,sigma)



#functions for model pricing   
def B_model(a,T,t):
    return (1-np.exp(-a*(T-t)))/a
    
def A_model(T, t, a, sigma, r_bar, B):
    tmp1 = ((sigma*B)**2)/(4*a)
    tmp2 = ((a**2)*r_bar-(sigma**2)/2)/(a**2)
    return np.exp((B-T+t)*tmp2 -  tmp1)
    
def ZC_model(T, t, a, sigma, r_bar, r):# prix Zero coupon = P(t,T)
    B=B_model(a,T,t)
    A=A_model(T, t, a, sigma, r_bar, B)
    return A*np.exp(-B*r)

def volatility_total(sigma,a,Maturity, pricing_date):
    return np.sqrt(((1- np.exp(-2*a*(Maturity-pricing_date)))*sigma**2)/(2*a))

def sigma_P(sigma, a, tenor, maturity, pricing_date): # grand sigma dans le fichier
    Volatility_total= volatility_total(sigma,a,maturity, pricing_date)
    return Volatility_total/a#*(1- np.exp(-a*tenor))/a     #

def d1_model(strike, prix_T, prix_S, vol_total):
    d1=(np.log(strike*prix_T/prix_S) + (vol_total**2)/2)/vol_total
    return d1

def d2_model(d1, vol_total):
    return (d1 - vol_total)
    
def put_bond_price(K,t,T,S,a,sigma,r_bar,r):   ##même d1 et d2 qu'avant???
    prix_T=ZC_model(T,t,a,sigma,r_bar,r)
    prix_S=ZC_model(S,t,a,sigma,r_bar,r)
    vol_total=sigma_P(sigma,a,S,T,t)
    d_1=d1_model(K,prix_T,prix_S,vol_total)
    d_2=d2_model(d_1,vol_total)    
    price=K*prix_T*norm.cdf(-d_1, loc=0, scale=1) - prix_S*norm.cdf(-d_2, loc=0, scale=1)
    return price

def call_bond_price(K,t,T,S,a,sigma,r_bar,r):
    prix_T=ZC_model(T,t,a,sigma,r_bar,r)
    prix_S=ZC_model(S,t,a,sigma,r_bar,r)
    vol_total=sigma_P(sigma,a,S,T,t)
    d_1=d1_model(K,prix_T,prix_S,vol_total)
    d_2=d2_model(d_1,vol_total)
    price=prix_S*norm.cdf(d_1, loc=0, scale=1)-(K*prix_T*norm.cdf(d_2, loc=0, scale=1))
    return price
    
    
def sum_ZC_model(r, K, dt, T_0, T_N, t, a, sigma, r_bar):   #dernière somme du doc pour résoudre l'équation 
    c_i=K*dt
    T_i=T_0
    somme_equation=0
    while (T_i+dt<T_0+T_N):
        somme_equation+=c_i*ZC_model(T_i,t,a,sigma,r_bar,r)
        T_i+=dt
    somme_equation+=(1+c_i)*ZC_model(T_0+T_N,t,a,sigma,r_bar,r)
    return (somme_equation-1)
    
from sympy.solvers import solve
#ympy.solvers.solvers.solve(f, *symbols, **flags)[source]¶
def get_rate_from_equation(K, dt, T_0, T_N, t, a, sigma, r_bar):###PROBLÈME DE SOLVEUR EQUATION
    root=opt.bisect(sum_ZC_model,-1,1, args=(K, dt, T_0, T_N, t, a, sigma, r_bar), maxiter=200)
    #root=solve(sum_ZC_model, args=(K, dt, T_0, T_N, t, a, sigma, r_bar))
    return root  
    # voir fin document
    # resoudre somme c_i * ZC_model(T, t, a, sigma, r_bar, r*) = 1

def PS_model(K,dt,T_0, T_N, t, spot_rate_optimized, a, sigma, r_bar):# on entraine le réseau de neuronne dessus 
    c_i=K*dt
    #print(c_i)
    T_i=T_0
    somme_price=0
    while (T_i<=T_0+T_N) :
        if (T_i+dt>T_0+T_N): 
            c_i= 1+(K*dt)
        K_i=ZC_model(T_i, t, a, sigma, r_bar, spot_rate_optimized)
        somme_price += c_i*call_bond_price(K_i,t,T_0,T_i,a,sigma,r_bar,spot_rate_optimized)
        T_i+=dt
    return somme_price
    #    (4.2.1.10), utiliser ZC_model 

def PS_model_to_train(K,dt,T_0,T_N,t, a, sigma, r_bar):
    spot_rate=get_rate_from_equation(K, dt, T_0, T_N, t, a, sigma, r_bar)
    PS_final=PS_model(K,dt,T_0, T_N, t, spot_rate, a, sigma, r_bar)
    return PS_final
# -> avec la première 
# -> à entrainer sur la grille simulée 

def plot_history(history):
    # Plot training & validation loss values
    plt.plot(history.history['rmse'], label = "train_loss")
    plt.plot(history.history['val_rmse'], label = "test_loss")
    plt.title('Model loss')
    plt.yscale('log')
    plt.ylabel('Loss (RMSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()