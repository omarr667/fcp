import pandas as pd
from fcp import functions, classes

def smart_beta_check(asset_aggresivo, asset_beta_1, asset_defensivo):
    universe = functions.get_universe()
    assets = list(universe['asset'])
    assets.remove('^SPX')
    results = []
    benchmark = "^SPX"
    
    for asset in assets:
        capm = classes.CapitalAssetPricingModel(benchmark, asset)
        capm.load_data()
        capm.compute_beta()
        results.append({
            "asset": capm.asset,
            "beta": capm.beta,
            "correlation": capm.correlation,
            "null_hypothesis": capm.null_hypothesis,
        })
    
    df = pd.DataFrame(results)
    df = df.set_index('asset')
    df_betas = df[
        (df['null_hypothesis'] == False) & 
        (abs(df['correlation']) > 0.5)
    ]
    df_betas = df_betas.sort_values(by=['beta'], ascending=False)
    
    df_betas_agresivo = df_betas['beta'][df_betas['beta'] > 1.3]
    df_betas_1 = df_betas['beta'][(df_betas['beta'] > 0.95) & (df_betas['beta'] < 1.05)]
    df_betas_defensivo = df_betas['beta'][(df_betas['beta'] > 0.4) & (df_betas['beta'] < 0.7)]   
    
    is_agresivo = asset_aggresivo in df_betas_agresivo.index
    is_beta_1 = asset_beta_1 in df_betas_1.index
    is_defensivo = asset_defensivo in df_betas_defensivo.index
    
    return is_agresivo, is_beta_1, is_defensivo