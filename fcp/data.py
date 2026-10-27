# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 13:04:19 2026

@author: omarr
"""
import pandas as pd
from . import database

def get_universe():
    """Devuelve el catálogo de activos guardos en base de datos"""
    with database.connect() as conn:
        df = pd.read_sql_query('SELECT * FROM universo ORDER BY ID', conn)
    
    return df


def get_asset(symbol):
    """
    Devuelve la tabla solicitada

    Parameters
    ----------
    symbol : STR
        El ticker del elemento a descargar sus precios de cierre

    Returns
    -------
    df con la columna Date, Close
    """
    # Las comillas permiten consultar tablas con caracteres especiales como
    # ^MXX, BTC-USD, EURUSD=x
    table = symbol.replace('"', '""')
    with database.connect() as conn: 
        tabla = pd.read_sql_query(
            f'SELECT * FROM "{table}" ORDER BY Date' 
            , conn)
    
    if tabla.empty:
        raise ValueError(f"No datos para {symbol}")
    
    
    return tabla
    
    
    
    
def get_prices(symbols, how='inner'):
    """
    Devuelve los precios de cierre

    Parameters
    ----------
    symbols : STR o list
        El ticker del elemento a descargar sus precios de cierre
    
    How: STR
        Tipo de merge, por default 'inner'
        --- 'inner': solo fechas comunes
        --- 'outer': todas las fechas, NaN donde no hay datos
        --- 'left': todas las fechas del primer símbolo, NaN donde no hay datos
        --- 'right': todas las fechas del último símbolo, NaN donde no hay datos
        --- 'cross': producto cartesiano, todas las combinaciones de fechas y símbolos
        --- 'left_anti': todas las fechas del primer símbolo que no están en el segundo, NaN donde no hay datos
        --- 'right_anti': todas las fechas del último símbolo que no están en el primero, NaN donde no hay datos
        --- 'asof': todas las fechas del primer símbolo que están en el segundo, NaN donde no hay datos
   
    Returns
    -------
    df con la columna Date, Close
    """
    if isinstance(symbols, str):
        symbols = [symbols]
        
    precios = None
    for symbol in symbols:    
        df = get_asset(symbol)
        df = df[["Date", "Close"]]
        df.rename(columns={"Close": symbol}, inplace=True)
        
        if precios is None:
            precios = df
        else:
            precios = pd.merge(precios, df, 
                               on="Date", 
                               how=how)
            
        
    precios.sort_values("Date", inplace=True)
    precios.set_index("Date", inplace=True)
            
    return precios
    

def get_returns(symbols, how='inner'):
    """
    Devuelve los rendimientos del activo seleccionado

    Parameters
    ----------
    symbols : STR o list
        El ticker del elemento a descargar sus precios de cierre

    How: STR
        Tipo de merge, por default 'inner'
        --- 'inner': solo fechas comunes
        --- 'outer': todas las fechas, NaN donde no hay datos
        --- 'left': todas las fechas del primer símbolo, NaN donde no hay datos
        --- 'right': todas las fechas del último símbolo, NaN donde no hay datos
        --- 'cross': producto cartesiano, todas las combinaciones de fechas y símbolos
        --- 'left_anti': todas las fechas del primer símbolo que no están en el segundo, NaN donde no hay datos
        --- 'right_anti': todas las fechas del último símbolo que no están en el primero, NaN donde no hay datos
        --- 'asof': todas las fechas del primer símbolo que están en el segundo, NaN donde no hay datos
        
    Returns
    -------
    df con la columna Date, Rendimientos
    """
    if isinstance(symbols, str):
        symbols = [symbols]
        
    returns = None
    for symbol in symbols:    
        df = get_asset(symbol)
        df = df[["Date", "Return"]]
        df.rename(columns={"Return": symbol}, inplace=True)
        
        if returns is None:
            returns = df
        else:
            returns = pd.merge(returns, df, 
                               on="Date", 
                               how=how)
            
        
    returns.sort_values("Date", inplace=True)
    returns.set_index("Date", inplace=True)
            
    return returns