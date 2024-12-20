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



        
class CapitalAssetPricingModel:
    
    # constructor
    def __init__(self, benchmark, asset):
        self.benchmark = benchmark # benchmark
        self.asset = asset # activo
        self.assets = [benchmark, asset]
        self.df = None
        self.df_close = None
        self.df_return = None
        self.alpha = None
        self.beta = None
        self.correlation = None
        self.r_squared = None
        self.p_value = None
        self.null_hypothesis = None
        self.y_predictor = None
        
    # cargar los datos de series de tiempo de los activos
    def load_data(self):
        self.df = functions.get_assets_data(self.assets)
        self.df_close = functions.get_assets_data(self.assets, 'close')
        self.df_return = functions.get_assets_data(self.assets, 'return')
        self.x_return = np.array(self.df_return[self.benchmark]) # benchmark
        self.y_return = np.array(self.df_return[self.asset]) # activo
        
    # calcular el beta y otras métricas relativas
    def compute_beta(self):
        x = np.array(self.df_return[self.benchmark]) # benchmark
        y = np.array(self.df_return[self.asset]) # activo
        slope, intercept, r_value, p_value, std_err = st.linregress(self.x_return,self.y_return)
        self.alpha = intercept
        self.beta = slope
        self.correlation = r_value
        self.r_squared = r_value**2
        self.p_value = p_value
        self.null_hypothesis = bool(p_value > 0.05)
        self.y_predictor = self.alpha + self.beta*self.x_return
        
    # graficar las series de tiempo de los activos
    def plot_timeseries(self):
        functions.plot_comparing_two_timeseries(self.assets, self.df)
    
    # graficar la regresión lineal del CAPM
    def plot_beta(self):
        title = (
                f"Linear regression \n" 
                f" asset {self.asset} "
                f" | benchmark {self.benchmark} \n"
                f" alpha (intercept) {self.alpha:.4f} "
                f" | beta (slope) {self.beta:.4f} \n"
                f" p-value {self.p_value} "
                f" | null hypothesis {self.null_hypothesis} \n"
                f" correl (r-value) {self.correlation:.4f}"
                f" | r-squared {self.r_squared:.4f}"
            )
        plt.figure(figsize=(10,6))
        plt.scatter(self.x_return, self.y_return, color='blue', label='Datos reales')
        plt.plot(self.x_return, self.y_predictor, color='red', label='Línea de regresión')
        plt.xlabel(self.benchmark)
        plt.ylabel(self.asset)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    