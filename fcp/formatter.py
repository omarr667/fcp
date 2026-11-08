# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:08:25 2026

@author: omarr
"""
from . import data

def format_number(value, pct = False):
    """
    Da formato a números.

    Parameters
    ----------
    value : float | int
    pct : Boolean
        Si es True, mostramos un número decimal en porcentaje
        si es False, mostramos un número con 4 decimales

    """
    return  f"{value:.4%}" if pct else f"{value:.4f}"

def display_name(asset):
    """
    Obtiene un nombre compuesto por el Ticker - nombre largo del activo
    
    Parameters
    ----------
    asset : str
        Nombre del activo

    Returns
    -------
    str: Nombre compuesto
        {asset} - {nombre largo del activo}
    """
    universe = data.get_universe()
    asset_record = universe[ universe["asset"] == asset ]
    if asset_record.empty:
        return f'{asset} - No encontrado'
    else:
        long_name = asset_record["name"].values[0]
        return f'{asset} - {long_name}'
    
    
def get_asset_info(asset):
    """
    Obtiene la información de un activo
    
    Parameters
    ----------
    asset : str
        Nombre del activo

    Returns
    -------
    dict: Información del activo
        {
            "id": id
            "asset": asset
            "name": nombre
            .....
        }
    """
    universe = data.get_universe()
    asset_record = universe[ universe["asset"] == asset ]
    if asset_record.empty:
        return {}
    else:
        dict_info = asset_record.to_dict(orient="records")[0]
        dict_info["display_name"] = display_name(asset)
        return dict_info
    