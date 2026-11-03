# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:11:57 2026

@author: meval
"""

import numpy as np
import scipy.stats as sci
from fcp import data, functions
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

class Distribution:
    
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
        self.skewness = sci.skew(x)
        self.kurtosis = sci.kurtosis(x)
        self.var_95 = np.percentile(x,5)
        self.percentile_95 =np.percentile(x,95)
        self.jb, self.p_value, self.is_normal = functions.jarque_bera_test(x)
        self.return_year = self.mean * factor
        self.volatility_year = self.std * np.sqrt(factor)
        self.sharpe_ratio = self.return_year / self.volatility_year if self.volatility_year > 0 else None
    
    def plot_prices(self):
        # self.df_prices.plot.line()
        fig = px.line(self.df_prices, 
                             x=self.df_prices.index,
                             y=self.df_prices[self.ticker],
                             labels={'x':'fecha',
                                     'y':'precio'},
                             title=f'Serie de precios de {self.ticker}')
        fig.update_traces(line_color='darkblue',
            hovertemplate=f"%{{x|%d/%m/%Y}}<br>USD %{{y:.2f}}<extra>{self.ticker}</extra>")
        fig.show()
    
    def histogram(self):
        x = self.df_returns.values
        plt.hist(x, bins=50, edgecolor='black', color='lightblue')
        plt.scatter(self.var_95,0, color='red', s=50, zorder=3, label=f'Var 95% = {self.var_95:.4f}')
        plt.scatter(self.percentile_95,0, color='magenta', s=50, zorder=3, label=f'percentile 95% = {self.percentile_95:.4f}')
        plt.title(f"Histogram of {self.ticker}")
        plt.legend()
        plt.show()