# -*- coding: utf-8 -*-
"""
@author: Diplomado Finanzas Cuantitativas con Python
"""
import matplotlib.pyplot as plt
from fcp import data


def get_universe():
    return data.get_asset_data('universe')


def get_assets_data(asset_names, kind='both'):
    df = data.get_assets_data_general(asset_names=asset_names, kind=kind)
    return df

def get_assets_data_where(query_where, kind='both'):
    df = data.get_assets_data_general(query_where=query_where, kind=kind)
    return df


# Forma 1: Correlación y varianza
def compute_corr_cov_matrix(data_df, classification='Return'):
    """
    Calcula las matrices de correlación y covarianza para un conjunto de datos filtrado.

    Parámetros:
    - data_df (data_dfFrame de pandas): data_dfFrame que contiene los datos de los activos.
    - classification (str): Subcadena para filtrar las columnas relevantes (por defecto 'Return').

    Retorna:
    - Diccionario con las matrices de 'correlation' y 'covariance'.

    Lanza:
    - ValueError: Si no se encuentran columnas que coincidan con la clasificación.
    """
    filtered_data_df = data_df.filter(like=classification) if classification else data_df
    if filtered_data_df.empty:
        raise ValueError(f"No se encontraron columnas que contengan la subcadena '{classification}'.")

    corr_matrix = filtered_data_df.corr()
    cov_matrix = filtered_data_df.cov()

    return {'correlation': corr_matrix, 'covariance': cov_matrix}


# Forma 2: Elemento por elemento
def compute_correlation_matrix(data_df, classification='Return'):
    """
    Calcula la matriz de correlación para un conjunto de datos filtrado.

    Parámetros:
    - data_df (data_dfFrame de pandas): data_dfFrame que contiene los datos de los activos.
    - classification (str): Subcadena para filtrar las columnas relevantes (por defecto 'Return').

    Retorna:
    - data_dfFrame de pandas con la matriz de correlación.

    Lanza:
    - ValueError: Si no se encuentran columnas que coincidan con la clasificación.
    """
    filtered_data_df = data_df.filter(like=classification) if classification else data_df
    if filtered_data_df.empty:
        raise ValueError(f"No se encontraron columnas que contengan la subcadena '{classification}'.")

    corr_matrix = filtered_data_df.corr()
    return corr_matrix


def compute_covariance_matrix(data_df, classification='Return'):
    """
    Calcula la matriz de covarianza para un conjunto de datos filtrado.

    Parámetros:
    - data_df (data_dfFrame de pandas): data_dfFrame que contiene los datos de los activos.
    - classification (str): Subcadena para filtrar las columnas relevantes (por defecto 'Return').

    Retorna:
    - data_dfFrame de pandas con la matriz de covarianza.

    Lanza:
    - ValueError: Si no se encuentran columnas que coincidan con la clasificación.
    """
    filtered_data_df = data_df.filter(like=classification) if classification else data_df
    if filtered_data_df.empty:
        raise ValueError(f"No se encontraron columnas que contengan la subcadena '{classification}'.")

    cov_matrix = filtered_data_df.cov()
    return cov_matrix



# Forma 3: Conjunta con mapeo de valores, separada por comas
def compute_corr_cov_matrix_map(data_df, classification='Return'):
    """
    Calcula las matrices de correlación y covarianza para un conjunto de datos filtrado y las retorna como una tupla.

    Parámetros:
    - data_df (data_dfFrame de pandas): data_dfFrame que contiene los datos de los activos.
    - classification (str): Subcadena para filtrar las columnas relevantes (por defecto 'Return').

    Retorna:
    - Tupla con la matriz de correlación y la matriz de covarianza.

    Lanza:
    - ValueError: Si no se encuentran columnas que coincidan con la clasificación.
    """
    filtered_data_df = data_df.filter(like=classification) if classification else data_df
    if filtered_data_df.empty:
        raise ValueError(f"No se encontraron columnas que contengan la subcadena '{classification}'.")

    corr_matrix = filtered_data_df.corr()
    cov_matrix = filtered_data_df.cov()

    return corr_matrix, cov_matrix


def plot_assets(data_df, classification='Return', style='matplotlib', **kwargs):
    """
    Grafica los datos de los activos utilizando el estilo seleccionado.

    Parámetros:
    - data_df (data_dfFrame de pandas): data_dfFrame que contiene los datos de los activos.
    - classification (str): Subcadena para filtrar las columnas a graficar (por defecto 'Return').
    - style (str): Estilo de la gráfica, puede ser 'matplotlib' o 'plotly' (por defecto 'matplotlib').
    - **kwargs: Argumentos adicionales para personalizar la gráfica (títulos, etiquetas, etc.).

    Retorna:
    - Ninguno. Muestra la gráfica directamente.

    Lanza:
    - ValueError: Si el estilo proporcionado no es soportado.
    """
    filtered_data_df = data_df.filter(like=classification) if classification else data_df
    if filtered_data_df.empty:
        raise ValueError(f"No se encontraron columnas que contengan la subcadena '{classification}' para graficar.")

    if style == 'matplotlib':
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        
        for asset_name in filtered_data_df.columns:
            plt.plot(filtered_data_df.index, filtered_data_df[asset_name], label=asset_name)
        
        # Obtiene los parámetros adicionales o usa valores por defecto
        title = kwargs.get('title', f'{classification} de los Activos')
        x_label = kwargs.get('xaxis_title', 'Fecha')
        y_label = kwargs.get('yaxis_title', classification)
        
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    elif style == 'plotly':
        import plotly.graph_objects as go
        import plotly.io as pio

        pio.renderers.default = 'browser'

        fig = go.Figure()

        for column in filtered_data_df.columns:
            fig.add_trace(go.Scatter(
                x=filtered_data_df.index, 
                y=filtered_data_df[column], 
                mode='lines', 
                name=column
            ))

        # Actualiza el layout con los parámetros adicionales
        fig.update_layout(
            title=kwargs.get('title', f'{classification} de los Activos'),
            xaxis_title=kwargs.get('xaxis_title', 'Fecha'),
            yaxis_title=kwargs.get('yaxis_title', classification),
            **kwargs.get('layout', {})
        )

        fig.show()
    
    else:
        raise ValueError("El estilo proporcionado no es válido. Usa 'matplotlib' o 'plotly'.")


def plot_comparing_two_timeseries(assets, assets_df):
    """
    Grafica dos series de tiempo en una sola figura para comparar sus precios de cierre.

    Parámetros:
    - assets (list): Lista con los nombres de los dos activos a comparar.
    - assets_df (data_dfFrame): data_dfFrame que contiene los datos históricos de los activos.

    Retorno:
    - Ninguno. Muestra directamente la gráfica.
    """
    try:
        # Verifica que la lista de activos tenga al menos dos elementos
        if len(assets) < 2:
            raise ValueError("La lista de activos debe contener al menos dos elementos.")
        
        # Define los nombres de los activos
        asset_1 = assets[0]
        asset_2 = assets[1]

        # Copia y prepara el data_dfFrame para graficar
        df2 = assets_df.copy()
        df2 = df2.reset_index()  # Asegura que 'Date' esté disponible como columna
        
        # Configura la figura
        plt.figure(figsize=(12, 6))
        plt.title('Comparación de Series de Tiempo de Precios de Cierre', fontsize=14)
        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Precio', fontsize=12)
        
        # Obtiene el eje actual para asignar múltiples líneas
        ax = plt.gca()

        # Grafica la primera serie de tiempo
        df2.plot(
            kind='line', x='Date', y=f'{asset_1}_Close', ax=ax, grid=True, 
            color='blue', label=f'{asset_1} (Cierre)'
        )

        # Grafica la segunda serie de tiempo en un eje secundario
        df2.plot(
            kind='line', x='Date', y=f'{asset_2}_Close', ax=ax, grid=True, 
            color='red', secondary_y=True, label=f'{asset_2} (Cierre)'
        )

        # Configura las leyendas
        ax.legend(loc='upper left', fontsize=10)
        ax.right_ax.legend(loc='upper right', fontsize=10)  # Leyenda del eje secundario

        # Muestra la gráfica
        plt.tight_layout()
        plt.show()

    except KeyError as e:
        print(f"Error: La columna especificada no se encuentra en el data_dfFrame. Detalle: {e}")
    except ValueError as e:
        print(f"Error en los datos: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")