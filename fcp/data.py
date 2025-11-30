import pandas as pd
import yfinance as yf
import sqlite3 as sql
import json
import importlib.resources

def get_asset_data_yahoo(asset_name, start_date, end_date, save_csv=False, conn=False):
    """
    Descarga los datos históricos de un activo financiero desde Yahoo Finance 
    y opcionalmente los guarda como CSV o en una base de datos.

    Params:
    - asset_name (str): símbolo del activo en Yahoo Finance.
    - start_date (str): fecha de inicio en formato 'YYYY-MM-DD'.
    - end_date (str): fecha de finalización en formato 'YYYY-MM-DD'.
    - save_csv (bool): si es True, guarda los datos en un archivo CSV.
    - conn (sqlite3.Connection): conexión a la base de datos SQLite para guardar los datos.

    Return:
    - hist (DataFrame): DataFrame de los datos históricos del activo.
    """
    try:
        # Obtiene el objeto de activo de Yahoo Finance
        asset = yf.Ticker(asset_name)
        # Descarga el historial de precios para el rango de fechas especificado
        hist = asset.history(start=start_date, end=end_date)
        
        if hist.empty:
            raise ValueError(f"No se obtuvieron datos para el activo {asset_name}")

        # Restablece el índice para que 'Date' sea una columna
        hist = hist.reset_index()
        
        # Convierte la columna de fecha a formato de fecha (sin hora)
        hist['Date'] = pd.to_datetime(hist['Date']).dt.date
        
        # Guarda el DataFrame como CSV si save_csv es True
        if save_csv:
            try:
                hist.to_csv(f"fcp/fcp_data/csv/{asset_name}.csv")
            except Exception as e:
                print(f"Error al guardar CSV para {asset_name}: {e}")
        
        # Guarda el DataFrame en la base de datos si conn está disponible
        if conn:
            try:
                hist.to_sql(asset_name, conn, if_exists="replace", index=False)
            except Exception as e:
                print(f"Error al guardar en base de datos para {asset_name}: {e}")
        
        return hist
    except Exception as e:
        print(f"Error al obtener datos para {asset_name}: {e}")
        return pd.DataFrame()  # Retorna un DataFrame vacío en caso de error

def generate_local_asset_universe(start_date, end_date):
    """
    Crea el universo local de activos descargando los datos históricos de cada uno
    y guardándolos en la base de datos y como archivos CSV.

    Parámetros:
    - start_date (str): fecha de inicio en formato 'YYYY-MM-DD'.
    - end_date (str): fecha de finalización en formato 'YYYY-MM-DD'.
    """
    try:
        # Carga el archivo JSON que contiene los activos a procesar
                
        with importlib.resources.open_text("fcp.fcp_data", "universe_assets.json", encoding='utf-8') as file:
            universe_dict = json.load(file)
            
    
        # Conecta a la base de datos SQLite
    
        with importlib.resources.path("fcp.fcp_data", "fcp_database.db") as database_path:
            conn = sql.connect(database_path)
        
        # Convierte el universo de activos en un DataFrame y lo guarda en la base de datos
        universe_df = pd.DataFrame(universe_dict)
        universe_df.to_sql('universe', conn, if_exists="replace", index=False)
        
        # Obtiene los nombres de los activos para procesar
        assets_names = universe_df['asset'].to_list()
        
        # Itera sobre cada activo y obtiene sus datos
        for asset_name in assets_names:
            print(f'Generando info de {asset_name}')
            get_asset_data_yahoo(asset_name, start_date, end_date, save_csv=True, conn=conn)
        
    except FileNotFoundError:
        print(f"El archivo fcp.fcp_data no se encontró.")
    except sql.Error as e:
        print(f"Error al trabajar con la base de datos: {e}")
    except Exception as e:
        print(f"Error general en generate_local_asset_universe: {e}")
    finally:
        conn.close()

# Llama a la función para generar el universo local de activos
### Sólo ejecuta 1 vez para cargar el universo
# generate_local_asset_universe('2021-01-01', '2024-10-31')


def get_universe():
    """
    Función para ver la tabla del universo de activos

    Returns
    -------
    None.
    """
    try:
        with importlib.resources.path("fcp.fcp_data", "fcp_database.db") as database_path:
            conn = sql.connect(database_path)

        df = pd.read_sql("SELECT * FROM universe;", conn)
        conn.close()
        
        return df
    except sql.Error as e:
        print(f"Error al obtener la base del universo de activos {e}")


def get_prices(assets):
    """
    Obtiene los precios de cierre de varios activos desde SQLite y 
    los combina por la columna 'Date'.

    Parameters
    ----------
    assets : list[str]
        Lista de tickers.

    Returns
    -------
    DataFrame
        DataFrame con columna Date y los precios de cierre de cada activo.
    """
    try:
        if isinstance(assets, str): 
            assets = [assets]
        
        with importlib.resources.path("fcp.fcp_data", "fcp_database.db") as database_path:
            conn = sql.connect(database_path)
            
        # Lista donde guardaremos cada DataFrame individual
        dfs = []

        for asset in assets:
            query = f'SELECT Date, Close AS "{asset}" FROM "{asset}"'
            df_asset = (
                pd.read_sql_query(query, conn)
                  .set_index("Date")
            )
            dfs.append(df_asset)

        # Unión horizontal de todos los DataFrames
        df_final = pd.concat(dfs, axis=1).reset_index()
        df_final = df_final.dropna()
        return df_final

    except sql.Error as e:
        print(f"Error al obtener precios: {e}")
        return None

    finally:
        conn.close()

def get_returns(assets):
    """
    Calcula los rendimientos diarios de varios activos.

    Parameters
    ----------
    assets : list[str]
        Lista de tickers.

    Returns
    -------
    DataFrame
        DataFrame con columna Date y los rendimientos diarios de cada activo.
    """
    try:
        # Obtener precios
        df_prices = get_prices(assets)
        df_returns = df_prices.copy()
        df_returns.set_index("Date", inplace=True)

        # Calcular rendimientos diarios
        df_returns = df_returns.pct_change().dropna().reset_index()

        return df_returns

    except Exception as e:
        print(f"Error al calcular rendimientos: {e}")
        return None




# Obtiene datos de la base de datos
def get_asset_data(asset_name=None, where=None, query=None):
    """
    Obtiene datos de un activo específico o ejecuta una consulta SQL personalizada.

    Parámetros:
    - asset_name (str, opcional): Nombre de la tabla del activo en la base de datos.
    - where (str, opcional): Condición SQL para filtrar los datos.
    - query (str, opcional): Consulta SQL personalizada. Si se proporciona, se ignora `asset_name` y `where`.

    Retorna:
    - DataFrame de pandas con los datos obtenidos de la base de datos.

    Lanza:
    - ValueError: Si no se proporciona ni `asset_name` ni `query`.
    - sqlite3.Error: Si ocurre un error al conectar o ejecutar la consulta en la base de datos.
    """
    if not asset_name and not query:
        raise ValueError("Se debe proporcionar 'asset_name' o 'query'.")

    try:
        # Conecta a la base de datos SQLite
        with importlib.resources.path("fcp.fcp_data", "fcp_database.db") as database_path:
            conn = sql.connect(database_path)

        if asset_name:
            # Construye la consulta SQL para seleccionar todos los registros de la tabla del activo
            sql_query = f'SELECT * FROM "{asset_name}"'
            if where:
                # Añade una cláusula WHERE si se proporciona
                sql_query += f' WHERE {where}'
        else:
            # Usa la consulta personalizada proporcionada
            sql_query = query

        # Ejecuta la consulta y carga los datos en un DataFrame
        df = pd.read_sql(sql_query, conn)
    except sql.Error as e:
        print(f"Error al acceder a la base de datos: {e}")
        raise
    finally:
        # Cierra la conexión a la base de datos
        conn.close()

    return df



# Cálculo de rendimientos
def get_assets_data_general(asset_names = None, query_where = None, kind='return'):
    """
       Calcula los rendimientos de múltiples activos y los combina en un solo DataFrame.
    
       Parámetros:
       - asset_names (lista de str, opcional): Lista de nombres de los activos a procesar.
         Si no se proporciona, se obtienen de la tabla 'universe' usando `query_where`.
       - query_where (str, opcional): Condición SQL para filtrar los activos desde la tabla 'universe'.
       - kind (str, opcional): Tipo de columnas a regresar ('close', 'return', 'both')
       Retorna:
       - DataFrame de pandas combinado con las columnas de cierre y rendimiento de cada activo.
    
       Lanza:
       - ValueError: Si `asset_names` no se proporciona y `query_where` tampoco filtra ningún activo.
   """
    if query_where:
        # Obtiene la lista de activos desde la tabla 'universe' según la condición proporcionada
        universe_temp_df = get_asset_data('universe', where=query_where)
        asset_names = universe_temp_df['asset'].to_list()
        if not asset_names:
            raise ValueError("No se encontraron activos con la condición proporcionada en 'query_where'.")
        print('Generando rendimientos de los activos:')
        print(asset_names)
    
    merged = None
    for asset_name in asset_names:
        
        df = get_asset_data(asset_name)
        if df.empty:
            print(f"Advertencia: La tabla para el activo '{asset_name}' está vacía.")
            continue
        if 'Date' not in df.columns or 'Close' not in df.columns:
            print(f"Advertencia: La tabla para el activo '{asset_name}' no contiene las columnas 'Date' y 'Close'.")
            continue
        close_df = df[['Date', 'Close']]
        close_df = close_df.sort_values(by=['Date'])
        close_df['Return'] = (close_df['Close']-close_df['Close'].shift(1)) / close_df['Close'].shift(1)
        close_df = close_df.dropna()
        close_name = f'{asset_name}_Close'
        return_name = f'{asset_name}_Return'
        close_df = close_df.rename(columns={'Close': close_name, \
                                            'Return': return_name})
            
        if merged is None:
            merged = close_df.copy()
        else:
            merged = merged.merge(close_df, how='inner', on='Date')
    
    if merged is None:
        raise ValueError("No se pudieron procesar los datos de los activos proporcionados.")
    merged = merged.set_index('Date')
    
    
    if kind == 'return':
        merged = merged.filter(like='_Return') #AAPL_Return --> AAPL
        merged.columns = [col.replace('_Return', '') for col in merged.columns]
    elif kind == 'close':
        merged = merged.filter(like='_Close')
        merged.columns = [col.replace('_Close', '') for col in merged.columns]
        
    return merged


def generate_index_universe():
    ''' Regenera el universo de activos en la base de datos SQLite
    '''
    try:
        # Carga el archivo JSON que contiene los activos a procesar
                
        with importlib.resources.open_text("fcp.fcp_data", "universe_assets.json", encoding='utf-8') as file:
            universe_dict = json.load(file)
            
    
        # Conecta a la base de datos SQLite
    
        with importlib.resources.path("fcp.fcp_data", "fcp_database.db") as database_path:
            conn = sql.connect(database_path)
        
        # Convierte el universo de activos en un DataFrame y lo guarda en la base de datos
        universe_df = pd.DataFrame(universe_dict)
        universe_df.to_sql('universe', conn, if_exists="replace", index=False)
    except FileNotFoundError:
        print(f"El archivo fcp.fcp_data no se encontró.")
    except sql.Error as e:
        print(f"Error al trabajar con la base de datos: {e}")
    except Exception as e:
        print(f"Error general en generate_local_asset_universe: {e}")
    finally:
        conn.close()