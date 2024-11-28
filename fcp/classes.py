import numpy as np
import scipy.stats as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from fcp import functions
importlib.reload(functions)


class Distribution:
    
    # constructor
    def __init__(self, asset):
        self.asset = asset
        self.timeseries = None
        self.vector_returns = None
        self.mean_annual = None
        self.volatility_annual = None
        self.sharpe_ratio = None
        self.skewness = None
        self.kurtosis = None
        self.jarque_bera_stat = None
        self.jarque_bera_p_value = None
        self.is_normal = None
        self.var_95 = None
        self.median = None
        
    # cargar datos
    def load_data(self):
        self.timeseries = functions.get_assets_data([self.asset])
        self.vector_returns = self.timeseries[f'{self.asset}_Return']
    
    # calcular mÃ©tricas
    def compute_metrics(self):
        factor = 252
        self.mean_annual = np.mean(self.vector_returns) * factor
        self.volatility_annual = np.std(self.vector_returns) * np.sqrt(factor)
        self.sharpe_ratio = self.mean_annual / self.volatility_annual if self.volatility_annual > 0.0 else 0.0
        self.skewness = st.skew(self.vector_returns)
        self.kurtosis = st.kurtosis(self.vector_returns)
        n = len(self.vector_returns)
        self.jarque_bera_stat = n/6*(self.skewness**2 + 1/4*self.kurtosis**2)
        self.jarque_bera_p_value = 1 - st.chi2.cdf(self.jarque_bera_stat, df=2)
        self.is_normal = (self.jarque_bera_p_value > 0.05)
        self.var_95 = np.percentile(self.vector_returns,5)
        self.median = np.median(self.vector_returns)
        
    # plot histograma
    def plot_histogram(self):
        plt.figure()
        plt.hist(self.vector_returns, bins=100)
        plt.title(f'Histograma de {self.asset}')
        plt.show()
