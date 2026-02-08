    #     # Convierte la columna de fecha a formato de fecha (sin hora)
    #     hist['Date'] = pd.to_datetime(hist['Date']).dt.date
        
    #     # Guarda el DataFrame como CSV si save_csv es True
    #     if save_csv:
    #         try:
    #             hist.to_csv(f"fcp/fcp_data/csv/{asset_name}.csv")
    #         except Exception as e:
    #             print(f"Error al guardar CSV para {asset_name}: {e}")
        
    #     # Guarda el DataFrame en la base de datos si conn está disponible
    #     if conn:
    #         try:
    #             hist.to_sql(asset_name, conn, if_exists="replace", index=False)
    #         except Exception as e:
    #             print(f"Error al guardar en base de datos para {asset_name}: {e}")
        
    #     return hist
    # except Exception as e:
    #     print(f"Error al obtener datos para {asset_name}: {e}")
    #     return pd.DataFrame()  # Retorna un DataFrame vacío en caso de error

def generate_local_asset_universe(start_date, end_date):
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

# Obtiene datos de la base de datos
def get_asset_data(asset_name=None, where=None, query=None):
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


def get_universe():
    try:
        with importlib.resources.path("fcp.fcp_data", "fcp_database.db") as database_path:
            conn = sql.connect(database_path)
        df = pd.read_sql("SELECT * FROM universe;", conn)
        conn.close()
        
        return df
    except sql.Error as e:
        print(f"Error al obtener la base del universo de activos {e}")



def get_prices(assets):
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
    universe_df = data.get_universe()
    asset_record = universe_df[universe_df["asset"] == asset]
    if asset_record.empty:
        return asset
    
    return f'{asset} - {asset_record["name"].values[0]}'


def compute_beta(benchmark, asset):
    capm = classes.CapitalAssetPricingModel(benchmark, asset)
    capm.compute()
    beta = float(capm.beta)
    return beta


def compute_betas(benchmark, assets):
    betas = []
    for asset in assets:
        beta = compute_beta(benchmark, asset)
        betas.append(beta)
    return betas

from fcp import data, functions
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sci
import plotly.graph_objects as go

class Distribution:
    
    # constructor
    def __init__(self, asset):
        self.asset = asset
        self.df_prices = data.get_prices(self.asset)
        self.df_returns = data.get_returns(self.asset)
        self.vec_returns = self.df_returns[self.asset].values
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

    # Constructor
    def __init__(self, portfolio, hedge_assets):
        self.portfolio = portfolio
        self.hedge_assets = self._as_list(hedge_assets)
    def compute_delta_neutral(self):
        benchmark = self.portfolio.benchmark  
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
        assets = self.hedge_assets
        if len(assets) < 2:
            raise ValueError("La función requiere al menos 2 activos de cobertura.")
        assets = assets[0:2]  # Seleccionamos los primeros 2 activos de cobertura

        benchmark = self.portfolio.benchmark
        beta1, beta2 = functions.compute_betas(benchmark, assets)
        A = np.array([
            [1, 1],
            [beta1, beta2]
        ])
        b = np.array([
            -self.portfolio.delta,
            -self.portfolio.betaUSD
        ])
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
    def __init__(self, assets):
        if not isinstance(assets, (list, tuple)):
            raise ValueError("asset debe ser una lista")
            
        self.assets = list(assets)
        self.n_assets = len(self.assets)
        
        # Resultados
        self.df_returns = None
        self.covariance_matrix = None
        self.correlation_matrix = None
        
        self.variance_annual = None
        self.volatility_annual = None
        
        # Eigenvalores y eigenvectores
        self.eigenvalues = None 
        self.eigenvectors = None 
        self.variance_explained = None
        
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
        
        # Matriz de covarianzas
        self.covariance_matrix = df.cov() * factor
        self.correlation_matrix = df.corr()
        
        # Varianzas anuales
        self.variance_annual = np.diag(self.covariance_matrix.values)
        self.volatility_annual = np.sqrt(self.variance_annual)
        
        # Eigenvalores
        evalor, evector = np.linalg.eigh(self.covariance_matrix.values)
        self.eigenvalues = evalor
        self.eigenvectors = evector
        self.variance_explained = self.eigenvalues / np.sum(self.eigenvalues)
        
        # Cota de la varianza del portfolio
        self.variance_min = self.eigenvalues[0]
        self.variance_max = self.eigenvalues[-1]
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
            
            

        
        