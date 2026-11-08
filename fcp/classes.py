# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:11:57 2026

@author: meval
"""

import numpy as np
import scipy.stats as sci
from . import data, functions
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

class Distribution:
    """ Clase para calcular y almacenar estadísticas de distribución de un activo financiero 
    Arguments:
    ----------
    ticker : str
            nombre del activo financiero (ticker) para el cual se calcularán las estadísticas de distribución.
    
    methods:
    --------
    compute(): Calcula las estadísticas de distribución del activo financiero.
    plot_prices(show=True): Genera un gráfico de línea de los precios del activo financiero.
    histogram_plotly(show=True): Genera un histograma de los rendimientos del activo financiero
    histogram(show=True): Genera un histograma de los rendimientos del activo financiero con matplotlib.
    """
    def __init__(self, ticker):
        
        self.ticker = ticker
        self.df_prices = data.get_prices(ticker)
        self.df_returns = data.get_returns(ticker)
        self.mean = None
        self.std = None
        self.skewness = None
        self.kurtosis = None
        self.var_95 = None
        self.percentile_95 = None
        self.jb = None
        self.p_value = None
        self.is_normal = None
        self.return_year = None
        self.volatility_year = None
        self.sharpe_ratio = None
        
    def compute(self):
        x = self.df_returns.values
        factor = 252
        self.mean = np.mean(x)
        self.std = np.std(x)
        self.skewness = sci.skew(x)[0]
        self.kurtosis = sci.kurtosis(x)[0]
        self.var_95 = np.percentile(x,5)
        self.percentile_95 =np.percentile(x,95)
        jb, p_value, is_normal = functions.jarque_bera_test(x)
        self.jb = jb[0]
        self.p_value = p_value[0]
        self.is_normal = is_normal[0]
        
        self.return_year = self.mean * factor
        self.volatility_year = self.std * np.sqrt(factor)
        self.sharpe_ratio = self.return_year / self.volatility_year if self.volatility_year > 0 else None
    
    def plot_prices(self, show=True):
        """ Genera un gráfico de línea de los precios del activo financiero utilizando Plotly.
        Args:
            show (bool): Si es True, muestra el gráfico en una ventana del navegador. Por defecto es True.
        Returns:
            fig (plotly.graph_objs._figure.Figure): Objeto de figura de Plotly que contiene el gráfico de línea.
        """
        # self.df_prices.plot.line()
        fig = px.line(self.df_prices, 
                             x=self.df_prices.index,
                             y=self.df_prices[self.ticker],
                             labels={'x':'fecha',
                                     'y':'precio'},
                             title=f'Serie de precios de {self.ticker}')
        fig.update_traces(line_color='darkblue',
            hovertemplate=f"%{{x|%d/%m/%Y}}<br>USD %{{y:.2f}}<extra>{self.ticker}</extra>")
        if show:
            fig.show()
        return fig
    
        
    def histogram_plotly(self, show=True):
        """ Genera un histograma de los rendimientos del activo financiero utilizando Plotly.
        Args:
            show (bool): Si es True, muestra el histograma en una ventana del navegador. Por defecto es True.
        Returns:
            fig (plotly.graph_objs._figure.Figure): Objeto de figura de Plotly que contiene el histograma.
        """
        fig = px.histogram( self.df_returns, 
                           x = self.ticker, 
                           nbins = 50, 
                           title = f"Histogram of {self.ticker}",
                           labels= {self.ticker: "Returns"}, 
                           color_discrete_sequence=['lightblue']
            )
        fig.update_traces(marker_line_color="black", marker_line_width=1)
        
        fig.add_scatter(
            x = [self.var_95],
            y = [0], 
            mode = "markers", 
            marker = dict(color="red", size=10),
            name=f'VaR 95% = {self.var_95:.4f}'
            )
        
        fig.add_scatter(
            x = [self.percentile_95],
            y = [0], 
            mode = "markers", 
            marker = dict(color="magenta", size=10),
            name=f'percentile 95% = {self.percentile_95:.4f}'
            )
        
        if show:
            fig.show()
        
        return fig
        
    def histogram(self, show=True):
        """ Genera un histograma de los rendimientos del activo financiero utilizando Matplotlib.
        Args:
            show (bool): Si es True, muestra el histograma en una ventana del navegador. Por defecto es True.
        Returns:
            plt (matplotlib.pyplot): Objeto de pyplot que contiene el histograma.
        """
        x = self.df_returns.values
        plt.hist(x, bins=50, edgecolor='black', color='lightblue')
        plt.scatter(self.var_95,0, color='red', s=50, zorder=3, label=f'VaR 95% = {self.var_95:.4f}')
        plt.scatter(self.percentile_95,0, color='magenta', s=50, zorder=3, label=f'percentile 95% = {self.percentile_95:.4f}')
        plt.title(f"Histogram of {self.ticker}")
        plt.legend()
        if show:
            plt.show()
        
        return plt