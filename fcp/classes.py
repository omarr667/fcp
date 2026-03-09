# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 10:42:24 2025

@author: meval
"""

from fcp import data, functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sci
import plotly.graph_objects as go
from scipy.optimize import minimize

class Distribution:
    
    # constructor
    def __init__(self, asset=None, portfolio=None):
        if asset:
            self.asset = asset
            self.df_prices = data.get_prices(self.asset)
            self.df_returns = data.get_returns(self.asset)
            self.vec_returns = self.df_returns[self.asset].values
        elif portfolio:
            self.asset = portfolio.port_type
            self.df_prices = portfolio.portfolio_prices
            self.df_returns = portfolio.portfolio_returns
            self.vec_returns = portfolio.portfolio_returns.values
        
            
        self.mean = None
        self.volatility = None
        self.mean_annual = None
        self.volatility_annual = None
        self.sharpe_ratio = None
        self.median = None
        self.var_95 = None
        self.skewness = None
        self.kurtosis = None
        self.p_value = None
        self.is_normal = None
    
    # método o función de plot de serie de tiempo de precios de cierre
    def plot_prices(self):
        self.df_prices.plot()
        
    # método o función de plot de serie de tiempo de rendimientos
    def plot_returns(self):
        self.df_returns.plot()
        
    # método o función de plot del histograma de rendimientos
    def plot_histogram(self):
        plt.figure()
        plt.hist(self.vec_returns, bins=100)
        plt.title(f'Histograma de {self.asset}')
        plt.show()
        
    # método o función para calcular métricas de riesgo   
    def compute(self):
        factor_annual = 252
        alpha = 0.95
        self.mean = np.mean(self.vec_returns)
        self.volatility = np.std(self.vec_returns)
        self.mean_annual = self.mean * factor_annual
        self.volatility_annual = self.volatility * np.sqrt(factor_annual)
        self.sharpe_ratio = self.mean_annual / self.volatility_annual
        self.median = np.median(self.vec_returns)
        self.var_95 = np.percentile(self.vec_returns, 5)
        n = len(self.vec_returns)
        self.skewness = sci.skew(self.vec_returns)
        self.kurtosis = sci.kurtosis(self.vec_returns)
        jb = n/6 * (self.skewness**2 + 1/4*self.kurtosis**2)
        self.p_value = 1 - sci.chi2.cdf(jb, df=2)
        self.is_normal = bool(self.p_value >= 1-alpha)
        
        
class CapitalAssetPricingModel:
     
    # constructor
    def __init__(self, benchmark, asset):
        self.benchmark = benchmark
        self.asset = asset
        self.df_prices = data.get_prices([benchmark, asset])
        self.df_returns = data.get_returns([benchmark, asset])
        self.alpha = None
        self.beta = None
        self.correlation = None
        self.p_value = None
        self.null_hypothesis = None
        self.predictor = None
         
    # plot de precios
    def plot_prices(self):
        plt.figure(figsize=(12, 6))
        plt.title("Comparación de Precios de Cierre", fontsize=14)
        ax = plt.gca()
        self.df_prices.plot(
            kind="line",
            x='Date',
            y=self.benchmark,
            ax=ax,
            grid=True,
            color="red",
            label= f"{self.benchmark} (Cierre)",
        )
        self.df_prices.plot(
            kind="line",
            secondary_y=True,
            x='Date',
            y=self.asset,
            ax=ax,
            grid=True,
            color="blue",
            label= f"{self.asset} (Cierre)",
        )
        ax.legend(loc="upper left", fontsize=10)
        ax.right_ax.legend(loc="upper right", fontsize=10)  # Leyenda del eje secundario
        plt.tight_layout()
        plt.show()
        return plt
    
    
    def plot_prices_plotly(self):
        
        assets_df = self.df_prices.copy()
        fig = go.Figure()
        # Serie 1: Activo color azul
        fig.add_trace(
            go.Scatter(
                x = assets_df["Date"], 
                y = assets_df[self.asset], 
                mode = "lines", 
                name=self.asset, 
                yaxis="y1",
                line=dict(color="cyan")
                )
        )
        # Serie 2: Benchmark color rojo
        fig.add_trace(
            go.Scatter(
                x = assets_df["Date"], 
                y = assets_df[self.benchmark], 
                mode = "lines", 
                name=self.benchmark, 
                yaxis="y2",
                line=dict(color="red")
                )
        )
        
        
        fig.update_layout(
            title="Comparación de precios de cierre",
            xaxis=dict(title="Fecha"),
            yaxis=dict(
                title=self.asset,
                showgrid=True, #
            ),
            yaxis2=dict(
                title=self.benchmark,
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=40, r=40, t=60, b=40),
        )

        
        return fig
        
        



        
    # calcular la regresión lineal del CAPM
    def compute(self):
        x = self.df_returns[self.benchmark].values
        y = self.df_returns[self.asset].values
        self.beta, self.alpha, self.correlation, self.p_value, se = sci.linregress(x, y) # esto es el CAPM
        self.r_squared = self.correlation**2
        self.null_hypothesis = bool(self.p_value >= 0.05)
        self.predictor = self.alpha + self.beta*x
        
    def plot(self):
        x = self.df_returns[self.benchmark].values
        y = self.df_returns[self.asset].values
        title = (
                f"Linear regression \n" 
                f" asset {self.asset} "
                f" | benchmark {self.benchmark} \n"
                f" alpha (intercept) {self.alpha:.4f} "
                f" | beta (slope) {self.beta:.4f} \n"
                f" p-value {self.p_value} "
                f" | correl (r-value) {self.correlation:.4f}"
                f" | r-squared {self.r_squared:.4f}"
            )
        plt.figure(figsize=(10,6))
        plt.scatter(x, y, color='blue', label='Datos reales')
        plt.plot(x, self.predictor, color='red', label='Línea de regresión')
        plt.xlabel(self.benchmark)
        plt.ylabel(self.asset)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        return plt
    
    
    def plot_regression_plotly(self, show=False):
        x = self.df_returns[self.benchmark].values
        y = self.df_returns[self.asset].values
        y_hat = self.predictor
        
        fig = go.Figure()
        # Trazar la nube de datos (x vs y)
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=y,
                mode="markers", 
                name="Datos reales", 
                marker=dict(color="cyan")
                
                )
        )
        
        # Línea de regresión
        order = np.argsort(x)
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=y_hat,
                mode="lines",
                name="Línea regresión", 
                line=dict(color="red")
                )
            )
        
        
        
        fig.update_layout(title="Regresión")
        
        return fig
    
    
class HedgeCAPM:
    """Clase que construye la cobertura.

    Para exactamente 1 activo:
        - Delta neutral
        - Beta neutral

    Para exactamente 2 activos de cobertura:
        - Neutralidad delta + beta
    """

    # Constructor
    def __init__(self, portfolio, hedge_assets):
        self.portfolio = portfolio
        self.hedge_assets = self._as_list(hedge_assets)  # Puede ingresar una lista o un texto

    # Calcular la cobertura delta-neutral (1 activo)
    def compute_delta_neutral(self):
        """Calcula la cobertura delta neutral para exactamente un activo."""
        benchmark = self.portfolio.benchmark  # Usamos el mismo benchmark del portafolio

        # En las diapositivas vimos que, para la neutralidad delta:
        #   w_h = -Delta(port)
        #
        # En nuestro caso:
        #   delta = w_h
        #   Delta(port) = self.portfolio.delta

        delta = -self.portfolio.delta
        notional = abs(delta)

        beta = functions.compute_betas(benchmark, self.hedge_assets)[0]
        betaUSD = beta * delta
        hedge = PortfolioCAPM(delta=delta,
                              betaUSD=betaUSD,
                              notional=notional,
                              asset=self.hedge_assets[0],
                              benchmark=benchmark,
                              beta=beta)
        return hedge

    # Calcular la cobertura beta-neutral (1 activo)
    def compute_beta_neutral(self):
        """Calcula la cobertura beta neutral para exactamente un activo."""
        benchmark = self.portfolio.benchmark  # Usamos el mismo benchmark del portafolio

        beta = functions.compute_betas(benchmark, self.hedge_assets)[0]
        if abs(beta) < 1e-12:
            raise ValueError("El beta es aproximadamente 0; no se puede construir una cobertura beta neutral.")

        # En las diapositivas vimos que, para la neutralidad beta:
        #   w_h = - betaUSD_portfolio / beta_h
        delta = -self.portfolio.betaUSD / beta

        notional = abs(delta)
        betaUSD = beta * delta
        hedge = PortfolioCAPM(delta=delta,
                              betaUSD=betaUSD,
                              notional=notional,
                              asset=self.hedge_assets[0],
                              benchmark=benchmark,
                              beta=beta)
        return hedge

    def compute(self):
        """Calcula la cobertura delta + beta neutral con 2 activos de cobertura."""
        assets = self.hedge_assets
        if len(assets) < 2:
            raise ValueError("La función requiere al menos 2 activos de cobertura.")
        assets = assets[0:2]  # Seleccionamos los primeros 2 activos de cobertura

        benchmark = self.portfolio.benchmark
        beta1, beta2 = functions.compute_betas(benchmark, assets)

        # Sistema de ecuaciones:
        #   w1 + w2 = -delta_portfolio
        #   beta1*w1 + beta2*w2 = -betaUSD_portfolio
        A = np.array([
            [1, 1],
            [beta1, beta2]
        ])
        b = np.array([
            -self.portfolio.delta,
            -self.portfolio.betaUSD
        ])

        # w = A^{-1} b
        #
        # Para poder invertir A, se requiere det(A) != 0.
        determinante = np.linalg.det(A)
        if abs(determinante) < 1e-12:
            raise ValueError("Sistema singular: la solución no es estable (det(A) ≈ 0).")

        w1, w2 = np.linalg.solve(A, b)

        weights = {assets[0]: float(w1), assets[1]: float(w2)}

        delta = w1 + w2
        notional = abs(w1) + abs(w2)
        betaUSD = beta1 * w1 + beta2 * w2

        assets_str = "|".join(assets)
        beta_str = "{ '%s': %s, '%s': %s}" % (assets[0], beta1, assets[1], beta2)

        hedge = PortfolioCAPM(delta=delta,
                              betaUSD=betaUSD,
                              notional=notional,
                              asset=assets_str,
                              benchmark=benchmark,
                              beta=beta_str,
                              weights=weights)
        return hedge

    # Helper
    def _as_list(self, item):
        """Transforma un elemento en una lista."""
        if isinstance(item, str):
            return [item]
        return item


class PortfolioCAPM:
    """Inicializa un portafolio con las características:
    notional, delta, betaUSD, asset=None, benchmark='^SPX'
    """

    # Constructor
    def __init__(self, delta, betaUSD, notional=None, asset=None, 
                 benchmark='^SPX', beta=None, weights=None): 
        self.delta = delta
        self.betaUSD = betaUSD
        self.notional = notional or abs(delta)
        self.asset = asset
        self.benchmark = benchmark
        self.beta = beta
        self.weights = weights

    def __repr__(self):
        """Devuelve una representación en string al imprimir la clase."""
        return (
            f"PortfolioCAPM(delta={self.delta}, betaUSD={self.betaUSD},\n"
                f"notional={self.notional}, asset={self.asset}, "
                f"benchmark={self.benchmark}, beta={self.beta},\n"
                f"weights={self.weights})"
        )


class CovarianceMatrix:
    """
    Analizando los movimientos de los activos.
    """
    
    def __init__(self, assets, **kwargs):
        if not isinstance(assets, (list, tuple)):
            raise ValueError("asset debe ser una lista")
        self.assets = list(assets)
        self.n_assets = len(self.assets)
        # Resultados
        self.df_returns = None
        self.covariance_matrix = None
        self.correlation_matrix = None
        # métricas de portafolios
        self.means = None
        self.variance_annual = None
        self.volatility_annual = None  
        self.sharpe_ratio = None
        # Eigenvalores y eigenvectores
        self.eigenvalues = None 
        self.eigenvectors = None 
        self.variance_explained = None
        self.benchmark = kwargs.get('benchmark','^SPX')
        self.betas = None
        # Portfolio
        self.variance_min = None
        self.variance_max = None
        self.volatility_min = None
        self.volatility_max = None
        
    def compute(self):
        factor = 252
        df = data.get_returns(self.assets)
        df.set_index("Date", inplace=True)
        self.df_returns = df
        # Valores medios
        X = df.values
        self.means = X.mean(axis = 0) * factor
        # Matriz de covarianzas
        self.covariance_matrix = df.cov() * factor
        self.correlation_matrix = df.corr()
        # betas de assets
        self.betas = functions.compute_betas(self.benchmark, self.assets)
        # Varianzas anuales
        self.variance_annual = np.diag(self.covariance_matrix.values)
        self.volatility_annual = np.sqrt(self.variance_annual)
        self.sharpe_ratio = self.means / self.volatility_annual
        # Eigenvalores
        evalor, evector = np.linalg.eigh(self.covariance_matrix.values)
        # Por default linalg.eigh estan ordenados de menor a mayor
        # Lo necesitamos de mayor a menor
        self.eigenvalues = evalor[::-1]
        self.eigenvectors = evector[:, ::-1]
        self.variance_explained = self.eigenvalues / np.sum(self.eigenvalues)
        # Cota de la varianza del portfolio
        self.variance_max = self.eigenvalues[0]
        self.variance_min = self.eigenvalues[-1]
        self.volatility_min = np.sqrt(self.variance_min)
        self.volatility_max = np.sqrt(self.variance_max)
        
    def compute_portfolio_variance(self, weights=None):
        self.compute()
        if weights is None:
            w = np.ones(self.n_assets) / self.n_assets
        else:
            w = np.array(weights) 
        covar = self.covariance_matrix.values 
        #portafolio_variance = np.matmul(w.T,np.matmul(covar, w))
        portafolio_variance = w.T @ covar @ w
        return portafolio_variance
    
    
class PortfolioManager(CovarianceMatrix):
    
    def __init__(self, assets,  benchmark='^SPX'):
        # Extender nuestro constructor
        super().__init__(assets, benchmark=benchmark) # Lo inicial
        self.df_metrics = None
        self.sharpe_ratio = None
    
        
    def compute(self):
        super().compute() # del padre: compute de CovarianceMatrix
        self.df_metrics = pd.DataFrame({
            "asset": self.assets, 
            "mean_annual": self.means,
            "volatility_annual": self.volatility_annual, 
            "sharpe_ratio": self.sharpe_ratio, 
            "beta": self.betas
        })
        self.df_metrics.set_index('asset', inplace=True)
    
    
    def compute_portfolio(self, port_type=None, notional=1, **kwargs):
        """
        Hace el compute del portfolio:
        Inputs:
            port_type: Tipos de portafolios: 
                'equiweight'
                'volatility_weighted
                'beta_weighted'
                'min_volatility_L1'
                'long_only'
                'markowitz'
        """
        # inputs para la optimización
        x0 = np.ones(self.n_assets) / self.n_assets
        args = (self.covariance_matrix.values)
        
        # restricciones de tipo ecuación para la optimización
        constraint_L1 = {'type':'eq','fun':lambda x: np.sum(np.abs(x)) - 1}
            
        if port_type == 'equiweight':
            weights = np.ones(self.n_assets)
            
        elif port_type == 'volatility_weighted':
            weights = 1 / self.volatility_annual
            
        elif port_type == 'beta_weighted':
            weights = self.betas
            
        elif port_type == 'min_volatility_L1':
            bounds = None
            constraints = (constraint_L1)
            result = minimize(fun=functions.portfolio_variance,
                              x0=x0,
                              args=args,
                              bounds=bounds,
                              constraints=constraints)
            weights = result.x
            
        elif port_type == 'long_only':
            bounds = [(0,None) for asset in self.assets]
            constraints = (constraint_L1)
            result = minimize(fun=functions.portfolio_variance,
                              x0=x0,
                              args=args,
                              bounds=bounds,
                              constraints=constraints)
            weights = result.x
            
        elif port_type == 'markowitz':
            target_return = kwargs.get('target_return',np.mean(self.means))
            bounds = [(0,None) for asset in self.assets]
            constraint_markowitz = {'type':'eq','fun':lambda x: (x.T @ self.means) - target_return}
            constraints = (constraint_L1, constraint_markowitz)
            result = minimize(fun=functions.portfolio_variance,
                              x0=x0,
                              args=args,
                              bounds=bounds,
                              constraints=constraints)
            weights = result.x
            
        elif port_type == 'custom':
            weights = kwargs.get('weights',np.ones(self.n_assets))
            
        else:
            port_type = 'equiweight'
            weights = np.ones(self.n_assets)
              
        # normalizar pesos en L1
        weights /= np.linalg.norm(weights,1)
        
        # cambiar pesos normalizados para satisfacer el nocional dado
        weights *= notional
        
        # cálculo de métricas de portafolios
        mean_annual = weights.T @ self.means
        variance_annual = functions.portfolio_variance(weights, self.covariance_matrix)
        volatility_annual = np.sqrt(variance_annual)
        sharpe_ratio = mean_annual / volatility_annual
        delta_usd = np.sum(weights)
        beta_usd = weights.T @ self.betas
        
        # crear la clase output de tipo Portfolio
        portfolio = Portfolio(assets = self.assets,
                              weights = weights, 
                              notional = notional,
                              port_type = port_type,
                              mean_annual = mean_annual,
                              volatility_annual = volatility_annual,
                              sharpe_ratio = sharpe_ratio,
                              delta_usd = delta_usd,
                              beta_usd = beta_usd,
                              df_returns=self.df_returns)
        
        return portfolio


class Portfolio:
    ''' Clase que representa un portafolio con sus características:
    input:
    - assets: lista de activos
    - weights: pesos de cada activo en el portafolio
    - notional: monto total del portafolio
    - port_type: tipo de portafolio (e.g., 'equiweight',
        'long_only', 'markowitz_default', 'markowitz')
    - mean_annual: rendimiento anual esperado del portafolio
    - volatility_annual: volatilidad anual del portafolio
    - sharpe_ratio: ratio de Sharpe del portafolio
    - delta_usd: exposición total en dólares del portafolio (suma de pesos
        multiplicada por el notional)
    - beta_usd: exposición total al riesgo sistemático del portafolio (suma de pesos multiplicada por los betas de los activos)
    '''
    def __init__(self, assets, weights, notional, port_type,
                 mean_annual, volatility_annual, sharpe_ratio,
                 delta_usd, beta_usd, df_returns=None):

        self.assets = assets
        self.weights = weights
        self.notional = notional
        self.port_type = port_type
        self.mean_annual = mean_annual
        self.volatility_annual = volatility_annual
        self.sharpe_ratio = sharpe_ratio
        self.n_assets = len(assets)
        self.delta_usd = delta_usd
        self.beta_usd = beta_usd
        self.df_returns = df_returns
        
        self.portfolio_returns = None
        self.portfolio_prices = None
        if self.df_returns is not None:
            returns_and_prices = self.compute_portfolio_returns()
            self.portfolio_returns = returns_and_prices[0]
            self.portfolio_prices = returns_and_prices[1]
        
        
        
        
    def __repr__(self):

        weights_fmt = [f"{w:.2%}" for w in self.weights]

        return (
            f"Portfolio(\n"
            f"  type        = {self.port_type}\n"
            f"  assets      = {self.assets}\n"
            f"  weights     = {weights_fmt}\n"
            f"  notional    = ${self.notional:,.0f}\n"
            f"  mean        = {self.mean_annual:.2%}\n"
            f"  volatility  = {self.volatility_annual:.2%}\n"
            f"  sharpe      = {self.sharpe_ratio:.3f}\n"
            f"  delta_usd   = ${self.delta_usd:,.2f}\n"
            f"  beta_usd    = ${self.beta_usd:,.2f}\n"
            f")"
        )
    
    def compute_portfolio_returns(self):
        
        df = self.df_returns.copy()
        portfolio_returns = df @ self.weights
        portfolio_returns.name = f"port_{self.port_type}"
        
        
        portfolio_prices = ( 1 +  portfolio_returns ).cumprod()
        portfolio_prices = 100 *  portfolio_prices / portfolio_prices.iloc[0]
        portfolio_prices.name = f"port_{self.port_type}"
        return portfolio_returns, portfolio_prices
    

    
    def plot_histogram(self):
        plt.Figure()
        plt.hist(self.portfolio_returns, bins=100)
        plt.title(f"Histograma del portfolio {self.port_type}")
        plt.show()
        return plt
    
    def plot_timeseries(self, assets_to_plot=None):
        df = self.df_returns.copy()
        
        # Su no se especifican activos a graficar
        # será graficar todos
        if assets_to_plot is None:
            assets_to_plot = self.assets
        
        # Dataframe vacío, donde se guardarán los valores
        df_plot = pd.DataFrame(index = df.index )
        
        # Nombre de la serie a graficar
        portfolio_name =  f"port_{self.port_type}"
        
        # Acumulación de returnos 
        df_plot[portfolio_name] = ( 1 + self.portfolio_returns ).cumprod()
        df_plot[portfolio_name] = 100 * df_plot[portfolio_name] / df_plot[portfolio_name].iloc[0]
        
        # Hacer lo mismo, pero para activo
        for asset in assets_to_plot:
            df_plot[asset] = (1 + df[asset]).cumprod()
            df_plot[asset] = 100 * df_plot[asset] /  df_plot[asset].iloc[0]
        
        # Grafica
        df_plot.plot(figsize=(12,6))
        plt.title(f"Evolución del Portfolio {self.port_type}")
        plt.grid(True)
        plt.show()
        
        return plt
            
        