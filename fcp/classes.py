import numpy as np
import scipy.stats as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from scipy.optimize import minimize
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
        self.quartile_1 = None
        self.quartile_3 = None
        
    # cargar datos
    def load_data(self):
        self.timeseries = functions.get_assets_data([self.asset])
        self.vector_returns = self.timeseries[f'{self.asset}']
    
    # calcular métricas
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
        self.quartile_1 = np.percentile(self.vector_returns,25)
        self.quartile_3 = np.percentile(self.vector_returns,75)   
        
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
        

class Hedger:
    
    # constructor
    def __init__(self, position, benchmark, hedge_assets):
        self.position_assets = list(position.keys())
        self.position_betas = None
        self.position_weights = list(position.values())
        self.position_delta_usd = None
        self.position_notional_usd = None
        self.position_beta_usd = None
        self.df_position = None
        self.benchmark = benchmark
        self.hedge_assets = hedge_assets
        self.hedge_betas = None
        self.hedge_weights = None
        self.hedge_delta_usd = None
        self.hedge_notional_usd = None
        self.hedge_beta_usd = None
        self.df_hedge = None
             
    # calcular betas
    def compute_betas(self):
        #position
        self.position_betas = functions.compute_betas(self.position_assets, self.benchmark)
        self.position_delta_usd = sum(self.position_weights)
        self.position_notional_usd = sum(np.abs(list(self.position_weights)))
        b = np.array(self.position_betas)
        x = np.array(self.position_weights)
        self.position_beta_usd = np.dot(b,x)
        df_position = pd.DataFrame(
            {'asset': self.position_assets, 
             'beta': self.position_betas,
             'weight': self.position_weights
             })
        self.df_position = df_position.set_index('asset')
        
        # hedge
        self.hedge_betas = functions.compute_betas(self.hedge_assets, self.benchmark)
        
    # calcular cobertura exacta - sólo por referencia
    def compute_hedge_exact(self):
        if len(self.hedge_assets) != 2:
            raise ValueError("Error: dim(hedge_assets) debe ser 2.")
        A = np.array([[1,1],
                      self.hedge_betas
                      ])
        b = np.array([[-self.position_delta_usd],
                      [-self.position_beta_usd]
                      ])
        self.hedge_weights = np.linalg.inv(A).dot(b).flatten()
        df_hedge = pd.DataFrame(
            {'asset': self.hedge_assets, 
             'beta': self.hedge_betas,
             'weight': self.hedge_weights
             })
        self.df_hedge = df_hedge.set_index('asset')
        self.hedge_delta_usd = df_hedge['weight'].sum()
        self.hedge_notional_usd = np.abs(df_hedge['weight']).sum()
        x = df_hedge['weight'].values
        self.hedge_beta_usd = np.dot(self.hedge_betas,x)
          
    # calcular cobertura numérica
    def compute_hedge(self, regularization=0.0):
        x0 = [self.position_delta_usd / len(self.hedge_assets)] * len(self.hedge_betas)
        args = (self.position_delta_usd,
                self.position_beta_usd,
                self.hedge_betas,
                regularization)
        result = minimize(functions.cost_function_hedge, x0=x0, args=args)
        self.hedge_weights = result.x
        df_hedge = pd.DataFrame(
            {'asset': self.hedge_assets, 
             'beta': self.hedge_betas,
             'weight': self.hedge_weights
             })
        self.df_hedge = df_hedge.set_index('asset')
        self.hedge_delta_usd = df_hedge['weight'].sum()
        self.hedge_notional_usd = np.abs(df_hedge['weight']).sum()
        x = df_hedge['weight'].values
        self.hedge_beta_usd= np.dot(self.hedge_betas,x)


class PortfolioManager:
    
    # constructor
    def __init__(self, assets, notional=1, benchmark='^SPX'):
        self.assets = assets
        self.notional = notional
        self.benchmark = benchmark
        self.df_returns = None
        self.mtx_covar = None
        self.mtx_correl = None
        self.df_metrics = None
        
    # recuperar rendimientos y matriz de varianza-covarianza
    def get_data(self):
        factor = 252 # para anualizar métricas
        df_returns = functions.get_assets_data(self.assets)
        v = df_returns
        mtx_covar = df_returns.cov() * factor
        self.mtx_covar = mtx_covar
        self.mtx_correl = df_returns.corr()
        means_annual = np.array(df_returns.mean()) * factor
        volatilities_annual = np.sqrt(np.diag(mtx_covar))
        sharpe_ratio = means_annual / volatilities_annual
        betas = functions.compute_betas(self.assets, self.benchmark)
        self.df_metrics = pd.DataFrame(
            data={'asset':self.assets,
                  'mean_annual':means_annual,
                  'volatility_annual':volatilities_annual,
                  'sharpe_ratio':sharpe_ratio,
                  'beta':betas})
        self.df_metrics.set_index('asset',inplace=True)
        self.df_returns = df_returns
        
    # calcular portafolios óptimos según el tipo deseado
    def compute_portfolio(self, port_type='equi_weight', **kwargs):#   target_return=None) :
        # Recuperar el rendimiento objetivo
        target_return = kwargs.get('target_return', None)

        # constraints o restricciones
        L1_norm = [{"type": "eq", "fun": lambda x: sum(abs(x)) - 1}]
        L2_norm = [{"type": "eq", "fun": lambda x: sum(x**2) - 1}]
        markowitz = [{"type": "eq", "fun": lambda x: np.dot(x, self.df_metrics['mean_annual']) - target_return}]
        
        # bounds o condiciones de frontera
        non_negative = [(0, None) for a in self.assets]
        
        # condicion inicial: portafolios equiponderado
        x0 = np.array([1.0 / len(self.assets) for a in self.assets])
        
        #verificar tipo de portafolios y calcular pesos
        if port_type == 'long_only':
            # optimización
            result = minimize(fun=functions.compute_portfolio_variance,
                              args=(self.mtx_covar),
                              x0=x0, 
                              constraints=(L1_norm), 
                              bounds=non_negative)
            # variables de portafolios
            weights = result.x
        
        elif port_type == 'min_variance_L1':
            # optimización
            result = minimize(fun=functions.compute_portfolio_variance,
                              args=(self.mtx_covar),
                              x0=x0, 
                              constraints=(L1_norm), 
                              bounds=None)
            # variables de portafolios
            weights = result.x
            
        elif port_type == 'min_variance_L2':
            # optimización
            result = minimize(fun=functions.compute_portfolio_variance,
                              args=(self.mtx_covar),
                              x0=x0, 
                              constraints=(L2_norm), 
                              bounds=None)
            # variables de portafolios
            weights = result.x
            
        elif port_type == 'beta_weighted':
            weights = np.array(self.df_metrics['beta'])
            
        elif port_type == 'volatility_weighted':
            weights = np.array(1 / self.df_metrics['volatility_annual'])
            
        elif port_type == 'markowitz':
            if target_return == None:
                target_return = np.mean(self.df_metrics['mean_annual'])
            # optimización
            result = minimize(fun=functions.compute_portfolio_variance,
                              args=(self.mtx_covar),
                              x0=x0, 
                              constraints=(L1_norm + markowitz), 
                              bounds=non_negative)
            # variables de portafolios
            weights = result.x

        elif port_type == 'external':
            external_weights = kwargs.get('weights') or []
            if len(external_weights) == len(self.assets):
                weights = np.array(external_weights)
            else:
                print("Error: 'weights' debe tener la misma longitud que 'assets'. Se usará el portafolio equiponderado.")
                port_type = 'equi_weight'
                weights = x0
                
        else:
            # por default, equiponderado
            port_type = 'equi_weight'
            weights = x0
        
        weights = weights / sum(abs(weights)) # pesos unitarios o en porcentaje
        port = self.compute_metrics(port_type, weights)
        
        return port

                               
    def compute_metrics(self, port_type, weights):
        allocation = self.notional * weights # pesos en USD
        
        # calcular métricas de portafolios
        mean_annual = np.dot(weights, self.df_metrics['mean_annual'])
        volatility_annual = np.sqrt(functions.compute_portfolio_variance(weights, self.mtx_covar))
        sharpe_ratio = mean_annual / volatility_annual
        # calcular la serie de tiempo de los rendimientos del portafolios
        returns = []
        for i in range(len(self.assets)):
            asset = self.assets[i]
            w = weights[i]
            ts = np.array(self.df_returns[asset])
            if len(returns) == 0:
                returns = w * ts
            else:
                returns += w * ts
        df_returns = self.df_returns.copy()
        df_returns[f'port_{port_type}'] = returns
                
        # calcular métricas diarias basadas en la clase Distribution
        mean = np.mean(returns)
        median = np.median(returns)
        var_95 = np.percentile(returns, 5)
        skewness = st.skew(returns)
        kurtosis = st.kurtosis(returns)
        n = len(returns)
        jarque_bera_stat = n/6*(skewness**2 + 1/4*kurtosis**2)
        jarque_bera_p_value = 1 - st.chi2.cdf(jarque_bera_stat, df=2)
        is_normal = (jarque_bera_p_value > 0.05)
                
        # salida de la optimización de portafolios
        port = Portfolio(self.assets, 
                         self.notional, 
                         port_type, 
                         weights,
                         allocation,
                         df_returns,
                         mean,
                         median,
                         var_95,
                         skewness,
                         kurtosis,
                         jarque_bera_stat,
                         jarque_bera_p_value,
                         is_normal,
                         mean_annual,
                         volatility_annual,
                         sharpe_ratio,
                         returns)
        return port
        
        
class Portfolio:
    
    # constructor
    def __init__(self, assets, notional, port_type, weights,
                 allocation, df_returns, mean, median, var_95, 
                 skewness, kurtosis, jarque_bera_stat, 
                 jarque_bera_p_value,is_normal, mean_annual, 
                 volatility_annual, sharpe_ratio, returns):
        self.assets = assets
        self.notional = notional
        self.port_type = port_type
        self.weights = weights
        self.allocation = allocation
        self.df_returns = df_returns
        self.mean = mean
        self.median = median
        self.var_95 = var_95
        self.skewness = skewness
        self.kurtosis = kurtosis
        self.jarque_bera_stat = jarque_bera_stat
        self.jarque_bera_p_value = jarque_bera_p_value
        self.is_normal = is_normal
        self.mean_annual = mean_annual 
        self.volatility_annual = volatility_annual
        self.sharpe_ratio = sharpe_ratio
        self.returns = returns
        
    # plot del histograma del portafolios
    def plot_histogram(self):
        plt.figure()
        plt.hist(self.returns, bins=100)
        plt.title(f'Histograma del portafolios {self.port_type}')
        plt.show()
        return plt
        
    # plot de la timeseries del portafolios y de sus activos
    def plot_timeseries(self, assets_to_plot = None):
        plt.figure()
        df = pd.DataFrame()
        pname = f'port_{self.port_type}'
        df[pname] = (1 + self.df_returns[pname]).cumprod()
        df[pname] = 100 * df[pname] / df[pname][0]
        if assets_to_plot == None:
            assets_to_plot =self.assets
        else:
            assets_to_plot = list(set(assets_to_plot) & set(self.assets))
        for asset in assets_to_plot:
            df[asset] = (1 + self.df_returns[asset]).cumprod()
            df[asset] = 100 * df[asset] / df[asset][0]
        df.plot()
        plt.show()
        return plt
        