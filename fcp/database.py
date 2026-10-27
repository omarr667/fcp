# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 11:40:34 2026

@author: omarr
"""
import pandas as pd
import yfinance as yf 
import warnings
import sqlite3
from contextlib import closing
from pathlib import Path

# Rutas relativas a la carpeta del paquete, funcionan sin importar desde dónde se importe fcp
DATA_DIR = Path(__file__).resolve().parent / 'data'
ASSETS_FILE = DATA_DIR / 'activos.csv'
DATABASE = DATA_DIR / 'fcp_database.db'

def connect(db_path = DATABASE):
    """
    Abre SQLite y entrega un gestor que cierra la conexión de la base al salir
    """
    return closing( sqlite3.connect(db_path))

def load_catalog(csv_path=ASSETS_FILE):
    """
    Leer el archivo CSV con el catálogo de activos a trabajar 

    Parameters
    ----------
    csv_path : str, optional
        La dirección del archivo CSV con el catálogo de activo.
            The default is ASSETS_FILE.

    Returns
    -------
    df: tabla con activos
    """
    catalog = pd.read_csv(csv_path, encoding="utf-8-sig")
    catalog["Símbolo"] = catalog["Símbolo"].str.strip() # quita espacios al incio y al final   '  AAPL  ' -> 'AAPL'
    if catalog.empty:
        raise ValueError("El catálago esta vacío, revisar el archivo")
    
    if catalog['Símbolo'].duplicated().any():
        raise ValueError("Hay símbolos (Tickers) duplicados")
        
    return catalog


        

def download_prices(asset_name, start_date, end_date):
    """
    Función que descarga datos históricos de Yahoo Finance.

    Parameters
    ----------
    asset_name : str
        Nombre del Ticker.
    start_date : str
        Fecha de inicio en formato yyyy-mm-dd
    end_date : str
        Fecha de fin en formato yyyy-mm-dd

    Returns
    -------
    df: DataFrame con los precios históricos del activo
    """    
    asset = yf.Ticker(asset_name)
    hist = asset.history(start=start_date, end=end_date)
    
    if hist.empty:
        mensaje = ('Hay error al descargar la información. '
                   'Revise que el Ticker exista y sea corrrecto')
        warnings.warn(mensaje)
        return 
    
    hist.reset_index(inplace=True)
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.date
    
    return hist



def create_database(start_date, end_date, symbols=None, 
                    csv_path = ASSETS_FILE, db_path = DATABASE):
    """
    Genera la base datos basados en el catálogo de activos. 

    Parameters
    ----------
    start_date : str
        Fecha de inicio en formato yyyy-mm-dd
    end_date : str
        Fecha de fin en formato yyyy-mm-dd
    symbols : (list o texto), optional
        Símbolos a descargar. Si None entonces descarga todos
        # Symbols = ["AAPL", "MSFT"] -> "AAPL" -> ["AAPL"]
    csv_path : str, optional
            La dirección del archivo CSV con el catálogo de activos
                The default is ASSETS_FILE.
    db_path : str, optional
        La dirección de la base de datos
        DESCRIPTION. The default is DATABASE.

    Returns
    -------
    Boolean: True: se creo correcto | False, no se creo la base de datos
    """
    # Cargar nuestro catálogo de activo
    catalogo = load_catalog(csv_path)
    print("Iniciando el proceso de crear la tabla")
    
    if symbols is not None:
        if isinstance(symbols, str):
            symbols = [symbols]
        catalogo = catalogo[ catalogo["Símbolo"].isin(symbols)   ]
        
        
    if catalogo.empty:
        raise ValueError("El catálogo es vacío. Revisar que todos los todos los activos existan en el catálogo")
        
    
    tablas = {}
    Tickers = catalogo["Símbolo"].tolist()
    n = len(Tickers)
    aux = 1
    for Ticker in Tickers:
        print(f"Descargando la información del Ticker {aux} / {n}: ", Ticker)
        df = download_prices(Ticker, start_date, end_date)
        df["Return"] = df["Close"].pct_change(fill_method=None)
        df = df.dropna()
        tablas[Ticker] = df
        aux += 1
    
    
    with connect(db_path) as conn:
        print("Guardando en base de datos")
        catalogo.to_sql("universo", conn, if_exists="replace", index=False)
        for tabla_name, precios in tablas.items():
            
            precios.to_sql(tabla_name, conn, if_exists="replace", index=False)
            
    
    # conn = sqlite3.connect(db_path)
    # catalogo.to_sql("universo", conn, if_exists="replace", index=False)
    # for tabla_name, precios in tablas.items():
    #     precios.to_sql(tabla_name, conn, if_exists="replace", index=False)
        
    # conn.close()
        
