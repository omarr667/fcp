# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 18:56:23 2025

@author: meval
"""

from fcp import data, classes
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sci

def jarque_bera_normality_test(x, alpha=0.95):
    n = len(x)
    skew = sci.skew(x)
    kurtosis = sci.kurtosis(x)
    jb = n/6 * (skew**2 + 1/4*kurtosis**2)
    p_value = 1 - sci.chi2.cdf(jb, df=2)
    is_normal = bool(p_value >= 1-alpha)
    return is_normal, p_value


def compute_display_name_asset(asset:str) -> str:
    """
    Función que genera un display name 
    Ticker + nombre
    
    Parameters
    ----------
    asset : str
        Activo

    Returns
    -------
    str
        Ticker + nombre.
    """
    universe_df = data.get_universe()
    asset_record = universe_df[universe_df["asset"] == asset]
    if asset_record.empty:
        return asset
    
    return f'{asset} - {asset_record["name"].values[0]}'


def compute_beta(benchmark, asset):
    """ Ajusta un modelo CAPM y regresa el beta correspondiente"""
    capm = classes.CapitalAssetPricingModel(benchmark, asset)
    capm.compute()
    beta = float(capm.beta)
    return beta


def compute_betas(benchmark, assets):
    """ Ajusta un modelo CAPM y regresa los betas correspondiente"""
    betas = []
    for asset in assets:
        beta = compute_beta(benchmark, asset)
        betas.append(beta)
    return betas