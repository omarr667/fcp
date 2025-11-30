# -*- coding: utf-8 -*-
"""
@author: Diplomado Finanzas Cuantitativas con Python
"""
import matplotlib.pyplot as plt
from fcp import data, classes
import numpy as np
import scipy.stats as st
from scipy import stats as sci

def jarque_bera_normality_test(x, alpha=0.95):
    n = len(x)
    skew = sci.skew(x)
    kurtosis = sci.kurtosis(x)
    jb = n/6 * (skew**2 + 1/4*kurtosis**2)
    p_value = 1 - sci.chi2.cdf(jb, df=2)
    is_normal = bool(p_value >= 1-alpha)
    return is_normal, p_value

def get_universe():
    """
    Devuelve el universo de activos disponibles.
    Returns:
        dataFrame de pandas con los activos disponibles.
    """
    return data.get_asset_data("universe")


def get_assets_data(asset_names, kind="return"):
    """
    Devuelve los datos de los activos solicitados.
    Args:
        asset_names (str o list): Nombre del activo o lista de nombres de activos.
        kind (str): Tipo de datos a obtener ('both', 'close', 'return').
    Returns:
        dataFrame de pandas con los datos de los activos solicitados.
    """

    if type(asset_names) == str:
        asset_names = [asset_names]
    df = data.get_assets_data_general(asset_names=asset_names, kind=kind)
    return df


def get_assets_data_where(query_where, kind="return"):
    """
    Devuelve los datos de los activos que cumplen con la condición especificada.
    Args:
        query_where (str): Condición para filtrar los datos.
        kind (str): Tipo de datos a obtener ('both', 'close', 'return').
    Returns:
        dataFrame de pandas con los datos de los activos que cumplen con la condición.
    """

    df = data.get_assets_data_general(query_where=query_where, kind=kind)
    return df


def compute_betas(assets, benchmark):
    """
    Calcula los betas de los activos con respecto a un benchmark.
    Args:
        assets (list): Lista de activos.
        benchmark (str): Benchmark.
    Returns:
        Lista con los betas de los activos.
    """
    betas = []
    for asset in assets:
        capm = classes.CapitalAssetPricingModel(benchmark, asset)
        capm.load_data()
        capm.compute_beta()
        betas.append(capm.beta)
    return betas


def compute_factors(asset, factors):
    """
    Calcula los betas de un activo con respecto a una lista de factores.
    Args:
        asset (str): Nombre del activo.
        factors (list): Lista de factores.
    Returns:
        Lista con los betas del activo con respecto a los factores.

    """
    betas = []
    for factor in factors:
        capm = classes.CapitalAssetPricingModel(factor, asset)
        capm.load_data()
        capm.compute_beta()
        betas.append(capm.beta)
    return betas


def compute_correlations(asset, factors):
    """
    Calcula las correlaciones de un activo con respecto a una lista de factores.
    Args:
        asset (str): Nombre del activo.
        factors (list): Lista de factores.
    Returns:
        Lista con las correlaciones del activo con respecto a los factores.
    """
    correl = []
    for factor in factors:
        capm = classes.CapitalAssetPricingModel(factor, asset)
        capm.load_data()
        capm.compute_beta()
        correl.append(capm.correlation)
    return correl


def compute_portfolio_variance(weights, mtx_cov):
    return np.matmul(np.transpose(weights), np.matmul(mtx_cov,weights))


def cost_function_hedge(
    x, position_delta_usd, position_beta_usd, hedge_betas, regularization
):
    f1 = (np.sum(x) + position_delta_usd) ** 2
    f2 = (np.dot(hedge_betas, x) + position_beta_usd) ** 2
    f3 = regularization * np.sum(np.array(x) ** 2)
    f = f1 + f2 + f3
    return f


# def compute_factors(asset, benchmarks):
#     factors = []
#     for benchmark in benchmarks:
#         capm = classes.CapitalAssetPricingModel(benchmark, asset)
#         capm.load_data()
#         capm.compute_beta()
#         factors.append(capm.beta)
#     return factors


# Forma 1: Correlación y varianza
def compute_corr_cov_matrix(data_df, classification="Return"):
    """
    No se recomienda usar esta función, ya que es más eficiente calcular las matrices por separado.

    Calcula las matrices de correlación y covarianza para un conjunto de datos filtrado.

    Parámetros:
    - data_df (data_dfFrame de pandas): data_dfFrame que contiene los datos de los activos.
    - classification (str): Subcadena para filtrar las columnas relevantes (por defecto 'Return').

    Retorna:
    - Diccionario con las matrices de 'correlation' y 'covariance'.

    Lanza:
    - ValueError: Si no se encuentran columnas que coincidan con la clasificación.
    """
    filtered_data_df = (
        data_df.filter(like=classification) if classification else data_df
    )
    if filtered_data_df.empty:
        raise ValueError(
            f"No se encontraron columnas que contengan la subcadena '{classification}'."
        )

    corr_matrix = filtered_data_df.corr()
    cov_matrix = filtered_data_df.cov()

    print(
        "Advertencia: No se recomienda usar esta función para calcular ambas matrices simultáneamente. Usa mejor df.cov()"
    )

    return {"correlation": corr_matrix, "covariance": cov_matrix}


# Forma 2: Elemento por elemento
def compute_correlation_matrix(data_df, classification="Return"):
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
    filtered_data_df = (
        data_df.filter(like=classification) if classification else data_df
    )
    if filtered_data_df.empty:
        raise ValueError(
            f"No se encontraron columnas que contengan la subcadena '{classification}'."
        )

    corr_matrix = filtered_data_df.corr()

    print(
        "Advertencia: No se recomienda usar esta función para calcular la matriz de correlación. Usa mejor df.corr()."
    )

    return corr_matrix


def compute_covariance_matrix(data_df, classification="Return"):
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
    filtered_data_df = (
        data_df.filter(like=classification) if classification else data_df
    )
    if filtered_data_df.empty:
        raise ValueError(
            f"No se encontraron columnas que contengan la subcadena '{classification}'."
        )

    cov_matrix = filtered_data_df.cov()
    return cov_matrix


# Forma 3: Conjunta con mapeo de valores, separada por comas
def compute_corr_cov_matrix_map(data_df, classification="Return"):
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
    filtered_data_df = (
        data_df.filter(like=classification) if classification else data_df
    )
    if filtered_data_df.empty:
        raise ValueError(
            f"No se encontraron columnas que contengan la subcadena '{classification}'."
        )

    corr_matrix = filtered_data_df.corr()
    cov_matrix = filtered_data_df.cov()

    return corr_matrix, cov_matrix


def plot_assets(data_df, classification="Return", style="matplotlib", **kwargs):
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
    filtered_data_df = (
        data_df.filter(like=classification) if classification else data_df
    )
    if filtered_data_df.empty:
        raise ValueError(
            f"No se encontraron columnas que contengan la subcadena '{classification}' para graficar."
        )

    if style == "matplotlib":
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))

        for asset_name in filtered_data_df.columns:
            plt.plot(
                filtered_data_df.index, filtered_data_df[asset_name], label=asset_name
            )

        # Obtiene los parámetros adicionales o usa valores por defecto
        title = kwargs.get("title", f"{classification} de los Activos")
        x_label = kwargs.get("xaxis_title", "Fecha")
        y_label = kwargs.get("yaxis_title", classification)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif style == "plotly":
        import plotly.graph_objects as go
        import plotly.io as pio

        pio.renderers.default = "browser"

        fig = go.Figure()

        for column in filtered_data_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=filtered_data_df.index,
                    y=filtered_data_df[column],
                    mode="lines",
                    name=column,
                )
            )

        # Actualiza el layout con los parámetros adicionales
        fig.update_layout(
            title=kwargs.get("title", f"{classification} de los Activos"),
            xaxis_title=kwargs.get("xaxis_title", "Fecha"),
            yaxis_title=kwargs.get("yaxis_title", classification),
            **kwargs.get("layout", {}),
        )

        fig.show()

    else:
        raise ValueError(
            "El estilo proporcionado no es válido. Usa 'matplotlib' o 'plotly'."
        )


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
            raise ValueError(
                "La lista de activos debe contener al menos dos elementos."
            )

        # Define los nombres de los activos
        asset_1 = assets[0]
        asset_2 = assets[1]

        # Copia y prepara el data_dfFrame para graficar
        df2 = assets_df.copy()
        df2 = df2.reset_index()  # Asegura que 'Date' esté disponible como columna

        # Configura la figura
        plt.figure(figsize=(12, 6))
        plt.title("Comparación de Series de Tiempo de Precios de Cierre", fontsize=14)
        plt.xlabel("Fecha", fontsize=12)
        plt.ylabel("Precio", fontsize=12)

        # Obtiene el eje actual para asignar múltiples líneas
        ax = plt.gca()

        # Grafica la primera serie de tiempo
        df2.plot(
            kind="line",
            x="Date",
            y=f"{asset_1}_Close",
            ax=ax,
            grid=True,
            color="blue",
            label=f"{asset_1} (Cierre)",
        )

        # Grafica la segunda serie de tiempo en un eje secundario
        df2.plot(
            kind="line",
            x="Date",
            y=f"{asset_2}_Close",
            ax=ax,
            grid=True,
            color="red",
            secondary_y=True,
            label=f"{asset_2} (Cierre)",
        )

        # Configura las leyendas
        ax.legend(loc="upper left", fontsize=10)
        ax.right_ax.legend(loc="upper right", fontsize=10)  # Leyenda del eje secundario

        # Muestra la gráfica
        plt.tight_layout()
        plt.show()

    except KeyError as e:
        print(
            f"Error: La columna especificada no se encuentra en el data_dfFrame. Detalle: {e}"
        )
    except ValueError as e:
        print(f"Error en los datos: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")




def call_price(inputs, t=None):
    t = t or inputs.t
    tau = inputs.T - t
    d1 = 1/(inputs.sigma*np.sqrt(tau)) * \
        (np.log(inputs.S_t / inputs.K) + \
         (inputs.r + 0.5* inputs.sigma**2) * tau)
    d2 = d1 - inputs.sigma*np.sqrt(tau)
    c = inputs.S_t * st.norm.cdf(d1) -\
        inputs.K * np.exp(-inputs.r * tau) * st.norm.cdf(d2)
    return c


def put_price(inputs, t=None):
    t = t or inputs.t
    tau = inputs.T - t
    d1 = 1/(inputs.sigma*np.sqrt(tau)) * \
        (np.log(inputs.S_t / inputs.K) + \
         (inputs.r + 0.5* inputs.sigma**2) * tau)
    d2 = d1 - inputs.sigma*np.sqrt(tau)
    p = inputs.K * np.exp(-inputs.r * tau) * st.norm.cdf(-d2) -\
        inputs.S_t * st.norm.cdf(-d1)        
    return p

def call_delta(inputs, t=None):
    t = t or inputs.t
    tau = inputs.T - t
    d1 = 1/(inputs.sigma*np.sqrt(tau)) * \
        (np.log(inputs.S_t / inputs.K) + \
         (inputs.r + 0.5* inputs.sigma**2) * tau)
    d = st.norm.cdf(d1)
    return d 

class Inputs:
    def __init__(self):
        self.r = None
        self.sigma = None
        self.t = None
        self.T = None
        self.S_t = None
        self.K = None
   