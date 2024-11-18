import pandas as pd
import numpy as np

class DataFetcher():
    def __init__(self, username, password, server):
        self.username = username
        self.password = password
        self.server = server

    def yahoo_finance(self, ticker, start_date, end_date):
        pass
    
    def darwinex_mt5(self, ticker, start_date, end_date):
        pass

    def binance(self, ticker, start_date, end_date):
        pass

class DataProcessor():
    def __init__(self):
        pass

    def get_returns(self, data):
        pass

    def get_volatility(self, data):
        pass

    def get_sharpe_ratio(self, data):
        pass

    def get_drawdown(self, data):
        pass

    def get_max_drawdown(self, data):
        pass

    def get_sortino_ratio(self, data):
        pass

    def get_calmar_ratio(self, data):
        pass

    def get_var(self, data):
        pass

    def get_cvar(self, data):
        pass

    def get_beta(self, data):
        pass

    def get_alpha(self, data):
        pass

    def get_r_squared(self, data):
        pass

    def get_information_ratio(self, data):
        pass

    def get_tracking_error(self, data):
        pass

    def get_jensen_alpha(self, data):
        pass

    def get_treynor_ratio(self, data):
        pass

    def get_sterling_ratio(self, data):
        pass

    def get_information_coefficient(self, data):
        pass

    def get_modigliani_ratio(self, data):
        pass

    def get_skewness(self, data):
        pass

    def get_kurtosis(self, data):
        pass

    def get_hurst_exponent(self, data):
        pass

    def get_correlation(self, data):
        pass

    def get_cointegration(self, data):
        pass

    def get_stationarity(self, data):
        pass

    def get_autocorrelation(self, data):
        pass

    def get_heteroskedasticity(self, data):
        pass

    def get_normality(self, data):
        pass

    def get_monte_carlo_simulation(self, data):
        pass

    def get_bootstrap(self, data):
        pass

    def get_garch(self, data):
        pass

    def get_rolling_statistics(self, data):
        pass

    def get_pca(self, data):
        pass

    def get_ica(self, data):
        pass

    def get_fa(self, data):
        pass

    def get_tsne(self, data):
        pass

    def get_umap(self, data):
        pass

    def get_isomap(self, data):
        pass

    def get_lle(self, data):
        pass