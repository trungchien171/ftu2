import pandas as pd
import numpy as np
import sys
import yfinance as yf
import MetaTrader5 as mt
import warnings
from datetime import datetime, timedelta
from binance.client import Client
from typing import Tuple
from sklearn.metrics import mean_squared_error
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter, plot, grid, text
from collections import defaultdict
from IPython.display import display
from IPython.core.display import HTML
import seaborn as sns
from sklearn.covariance import ShrunkCovariance,\
	empirical_covariance, log_likelihood
from scipy import linalg
import random
from sklearn.model_selection import GridSearchCV
from scipy.stats import percentileofscore
import scipy.stats as stats

    
class DataFetcher():
    def __init__(self, username, password, server, api_key, api_secret):
        self.username = username
        self.password = password
        self.server = server
        self.api_key = api_key
        self.api_secret = api_secret

    def mt5_initialization(self):
        mt.login(username=self.username, password=self.password, server=self.server)

    def binance_client_init(self):
        client = Client(self.api_key, self.api_secret)
        return client

    def yfinance_adj_close(self, tickers, start=None, end=None, period='1mo', interval='1d'):
        if isinstance(tickers, str):
            tickers = [tickers]

        try:
            if start and end:
                print(f"Fetching data for {tickers} from {start} to {end} with interval '{interval}'")
                data = yf.download(tickers, start=start, end=end, interval=interval)
            else:
                print(f"Fetching data for {tickers} with period '{period}' and interval '{interval}'")
                data = yf.download(tickers, period=period, interval=interval)
            
            if "Adj Close" in data.columns:
                return data["Adj Close"]
            else:
                print("Warning: 'Adj Close' data not available, returning full dataset.")
            return data
        except Exception as e:
            print(f"Error while fetching data: {e}")
            return None
    
    def darwinex_mt5(self, ticker, start_date, end_date):
        self.mt5_initialization()

        df = pd.DataFrame(mt.copy_rates_range(
            ticker, 
            mt.TIMEFRAME_D1, 
            start_date, 
            end_date
            )
        )

        df['time'] = pd.to_datetime(df['time'], unit='s').dt.strftime('%Y-%m-%d')
        df.drop(['spread', 'real_volume'], axis=1, inplace=True)
        df.set_index('time', inplace=True)
        return df
        
    def binance_api(self, ticker, interval, start_date):
        client = self.binance_client_init()

        end_time = str((datetime.utcnow() + timedelta(hours=7)).timestamp())
        start_time = start_time

        df = pd.DataFrame(client.get_historical_klines(ticker, interval, start_time, end_time))
        df = df.iloc[:, 0:6]
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        df['date'] = pd.to_datetime(df['date'], unit='ms') + timedelta(hours = 7)
        return df[:-1]
    
    def get_investor_data(self):
        print("Investor Profile")
        print("----------------")
        print("Name: Emily Chen")
        print("Age: 29")
        print("Background: Tech Startup Manager, actively investing for several years")
        
        print("\nAssets:")
        print("  1. High growth stocks: $150,000")
        print("  2. Equity in tech startup stock options: $200,000")
        print("  3. Retirement account: $50,000")
        
        print("\nLiabilities:")
        print("  1. Car loan: $30,000")
        print("  2. Student loan: $15,000")


class DataProcessor():
    def __init__(self):
        pass

    def return_from_prices(self, prices, log_returns = False):
        if log_returns:
            returns = np.log(1 + prices.pct_change()).dropna(how = 'all')
        else:
            returns = prices.pct_change().dropna(how = 'all')
        return returns
    
    def get_factors(self, factors):
        factor_file = factors + ".csv"
        factor_df = pd.read_csv(factor_file)

        # Rename the first column to 'Date'
        factor_df = factor_df.rename(columns={'Unnamed: 0': 'Date'})

        # Check if the date format is monthly or daily
        if len(str(factor_df['Date'].iloc[0])) == 6:  # YYYYMM format (monthly)
            factor_df['Date'] = factor_df['Date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m'))
        else:  # Assuming YYYYMMDD format (daily)
            factor_df['Date'] = factor_df['Date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

        # Set the 'Date' column as the index
        factor_df = factor_df.set_index('Date')

        return factor_df

    
class CovarianceShrinkage():
    def __init__(self, prices, returns_data = False, frequency = 252, log_returns = False, delta = None):
        try:
            from sklearn import covariance
            self.covariance = covariance
        except (ModuleNotFoundError, ImportError):
            raise ImportError('Please install scikit-learn to use this class.')
        
        if not isinstance(prices, pd.DataFrame):
            raise ValueError('data is not in a DataFrame', RuntimeWarning)
            prices = pd.DataFrame(prices)

        self.frequency = frequency
        self.data_processor = DataProcessor()

        if returns_data:
            self.X = prices.dropna(how = "all")
        else:
            self.X = self.data_processor.returns_from_prices(prices, log_returns).dropna(how="all")

        self.S = self.X.cov().values
        self.delta = delta

    def _is_positive_semidefinite(self, matrix):
        try:
            # stackoverflow.com/questions/16266720
            np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
            return True
        except np.linalg.LinAlgError:
            return False

    def fix_nonpositive_semidefinite(self, matrix, fix_method="spectral"):
        if self._is_positive_semidefinite(matrix):
            return matrix

        warnings.warn(
            "The covariance matrix is non positive semidefinite. Amending eigenvalues."
        )

        # Eigendecomposition
        q, V = np.linalg.eigh(matrix)

        if fix_method == "spectral":
            # Remove negative eigenvalues
            q = np.where(q > 0, q, 0)
            # Reconstruct matrix
            fixed_matrix = V @ np.diag(q) @ V.T
        elif fix_method == "diag":
            min_eig = np.min(q)
            fixed_matrix = matrix - 1.1 * min_eig * np.eye(len(matrix))
        else:
            raise NotImplementedError("Method {} not implemented".format(fix_method))

        if not self._is_positive_semidefinite(fixed_matrix):
            warnings.warn(
                "Could not fix matrix. Please try a different risk model.", UserWarning
            )

        # Rebuild labels if provided
        if isinstance(matrix, pd.DataFrame):
            tickers = matrix.index
            return pd.DataFrame(fixed_matrix, index=tickers, columns=tickers)
        else:
            return fixed_matrix
        
    def _format_and_annualize(self, raw_cov_array):
        assets = self.X.columns
        cov = pd.DataFrame(raw_cov_array, index=assets, columns=assets) * self.frequency
        return self.fix_nonpositive_semidefinite(cov, fix_method="spectral")
    
    def shrunk_covariance(self, delta = None):
        if delta is not None:
            self.delta = delta
        elif self.delta is None:
            raise ValueError("Delta must be set either during initialization or when calling the method.")
        
        N = self.S.shape[1]

        # Shrinkage target
        mu = np.trace(self.S) / N
        F = np.identity(N) * mu

        # Shrinkage
        shrunk_cov = delta * F + (1 - delta) * self.S
        return self._format_and_annualize(shrunk_cov)
    
    def ledoit_wolf(self):
        X = np.nan_to_num(self.X.values)
        shrunk_cov, self.delta = self.covariance.ledoit_wolf(X)
        return self._format_and_annualize(shrunk_cov)
    
    def oracle_approximating(self):
        X = np.nan_to_num(self.X.values)
        shrunk_cov, self.delta = self.covariance.oas(X)
        return self._format_and_annualize(shrunk_cov)
    
class VcvEstimation():
    def __init__(self):
        pass

    def get_number_of_features(self, dict_data):
        return sum(len(tickers) for tickers in dict_data.values()) - 2

    def color_matrix(self, num_of_features):
        return np.random.randn(num_of_features, num_of_features)
    
    def shrinkage_factor(self):
        return np.logspace(-2, 0, 32)
    
    def negative_log_likelihood(self, shrinkage_factor, train, test):
        return [-ShrunkCovariance(shrinkage=s).fit(train).score(test) for s in shrinkage_factor]
    
    def real_covariance(self, color_matrix):
        return np.dot(color_matrix.T, color_matrix)
    
    def log_real_likelihood(self, empirical_cov, real_cov):
        return - log_likelihood(empirical_cov, linalg.inv(real_cov))
    
    def get_optimal_shrinkage_coefficient(self, tuned_parameters, train):
        cv = GridSearchCV(ShrunkCovariance(), tuned_parameters)
        cv.fit(train)
        optimal_shrinkage_coef = cv.best_estimator_.shrinkage
        return cv, optimal_shrinkage_coef
    
    def plot_shrinkage_intensity(self, shrinkage_factor, negative_log_likelihood, log_real_likelihood, best_shrinkage, best_score):
        plt.loglog(shrinkage_factor,
                negative_log_likelihood,
                "m--",
                label="Negative log-likelihood")

        plt.plot(plt.xlim(),
                2 * [log_real_likelihood],
                "b-.",
                label="Real Covariance Likelihood")

        # Adjusting View in Graph
        maxLikelihood = np.amax(negative_log_likelihood)
        minLikelihood = np.amin(negative_log_likelihood)
        min_y = minLikelihood - 7.0 * np.log((plt.ylim()[1] - plt.ylim()[0]))
        max_y = maxLikelihood + 16.0 * np.log(maxLikelihood - minLikelihood)
        min_x = shrinkage_factor[0]
        max_x = shrinkage_factor[-1]

        # Best Covariance estimator likelihood
        plt.vlines(
            best_shrinkage, min_y,
            best_score,
            color="yellow",
            linewidth=3,
            label="Cross-validation Best estimation",
        )
        #plotting in Graph
        plt.title("Regularized Covariance: Likelihood & Shrinkage Coefficient")
        plt.xlabel("Regularization parameter: Shrinkage coefficient")
        plt.ylabel("Error calculation in negative log-likelihood on test-data")
        plt.ylim(min_y, max_y)
        plt.xlim(min_x, max_x)
        plt.legend()
        plt.show()


class FamaFrench():
    def __init__(self):
        pass

    def get_factors(self, factors):
        factor_file = factors + ".csv"
        factor_df = pd.read_csv(factor_file)

        factor_df = factor_df.rename(columns={'Unnamed: 0': 'Date'})

        if len(str(factor_df['Date'].iloc[0])) == 6:  # YYYYMM format (monthly)
            factor_df['Date'] = factor_df['Date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m').strftime('%Y-%m'))
        else:  # Assuming YYYYMMDD format (daily)
            factor_df['Date'] = factor_df['Date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

        factor_df = factor_df.set_index('Date')

        return factor_df

    def plot_impact(self, model, ols_data):
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Factor", y="Coefficient", data=ols_data, palette="Blues")

        for i, row in ols_data.iterrows():
            plt.text(
                i,
                row['Coefficient'],
                f"p-value: {model.pvalues[row['Factor']]:.4f}",
                ha="center",
                va="top",
                fontsize=10,
                color="black",
            )

        plt.title("Impact of Fama-French Factors on Monthly Returns", fontsize=14)
        plt.xlabel("Factor", fontsize=12)
        plt.ylabel("Coefficient Value", fontsize=12)

        plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.tight_layout()
        plt.show()

class BlackLitterman():
    def __init__(self):
        pass

    def get_target_prices(self, price_df):
        target_prices = pd.read_excel("target_prices.xlsx", sheet_name="Tickers", index_col=0)
        target_prices = target_prices[['Target price']]
        columns_to_select = price_df.columns
        target_prices = target_prices[target_prices.index.isin(columns_to_select)]
        target_series = target_prices['Target price']
        return target_series
    
    def plot_market_prior(self, market_prior):
        plt.figure(figsize=(10, 6))

        market_prior.plot.barh(
            color='#69b3a2',
            edgecolor='black',
            linewidth=1.5 
        )

        plt.grid(axis='x', linestyle='--', alpha=0.7)

        plt.title("Market Prior Returns", fontsize=16)
        plt.xlabel("Returns", fontsize=14)
        plt.ylabel("Tickers", fontsize=14)

        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()

    def get_confidences(self, view_dict):
        def calculate_confidence(views):
            if abs(views) > 0.3:
                return random.choice([0.7, 0.8, 0.9])
            elif abs(views) > 0.1:
                return random.choice([0.4, 0.5, 0.6])
            else:
                return random.choice([0.1, 0.2, 0.3])
            
        confidences = {ticker: calculate_confidence(return_value) for ticker, return_value in view_dict.items()}
        confidences = list(confidences.values())
        return confidences
    
    def plot_portfolio(self, weights):
        weights_series = pd.Series(weights)

        weights_series = weights_series[weights_series > 0]

        fig, ax = plt.subplots(figsize=(10, 10))
        weights_series.plot.pie(
            ax=ax, 
            autopct='%1.1f%%',  # Display percentage
            colors=plt.cm.Paired.colors,  # Use a color palette
            startangle=90,  # Start angle of the first slice
            legend=True,    # Display the legend
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}  # Wedge border properties
        )

        ax.set_title("Portfolio Weights Distribution", fontsize=16, fontweight='bold')
        ax.set_ylabel("")  # Remove default ylabel for aesthetics
        plt.tight_layout()
        plt.show()

class AlgorithmicTrading():
    def __init__(self):
        pass

    def generate_signals(self, input_df, start_capital=100000, share_count=2000):
        initial_capital = float(start_capital)

        signals_df = input_df.copy()

        # Set the share size:
        share_size = share_count

        # Take a 500 share position where the Buy Signal is 1 (prior day's predictions greater than prior day's returns):
        signals_df['Position'] = share_size * signals_df['Buy Signal']

        # Make Entry / Exit Column:
        signals_df['Entry/Exit']=signals_df["Buy Signal"].diff()

        # Find the points in time where a 500 share position is bought or sold:
        signals_df['Entry/Exit Position'] = signals_df['Position'].diff()

        # Multiply share price by entry/exit positions and get the cumulative sum:
        signals_df['Portfolio Holdings'] = signals_df['Returns'] * signals_df['Entry/Exit Position'].cumsum()

        # Subtract the initial capital by the portfolio holdings to get the amount of liquid cash in the portfolio:
        signals_df['Portfolio Cash'] = initial_capital - (signals_df['Returns'] * signals_df['Entry/Exit Position']).cumsum()

        # Get the total portfolio value by adding the cash amount by the portfolio holdings (or investments):
        signals_df['Portfolio Total'] = signals_df['Portfolio Cash'] + signals_df['Portfolio Holdings']

        # Calculate the portfolio daily returns:
        signals_df['Portfolio Daily Returns'] = signals_df['Portfolio Total'].pct_change()

        # Calculate the cumulative returns:
        signals_df['Portfolio Cumulative Returns'] = (1 + signals_df['Portfolio Daily Returns']).cumprod() - 1

        signals_df = signals_df.dropna()
        
        return signals_df
    
    def algo_evaluation(self, signals_df):
        # Prepare dataframe for metrics
        metrics = [
            'Annual Return',
            'Cumulative Returns',
            'Annual Volatility',
            'Sharpe Ratio',
            'Sortino Ratio']

        columns = ['Backtest']

        # Initialize the DataFrame with index set to evaluation metrics and column as `Backtest` (just like PyFolio)
        portfolio_evaluation_df = pd.DataFrame(index=metrics, columns=columns)
        # Calculate cumulative returns:
        portfolio_evaluation_df.loc['Cumulative Returns'] = (signals_df['Portfolio Cumulative Returns'][-1]) * 100
        # Calculate annualized returns:
        portfolio_evaluation_df.loc['Annual Return'] = (signals_df['Portfolio Daily Returns'].mean() * 252) * 100
        # Calculate annual volatility:
        portfolio_evaluation_df.loc['Annual Volatility'] = (signals_df['Portfolio Daily Returns'].std() * np.sqrt(252)) * 100
        # Calculate Sharpe Ratio:
        portfolio_evaluation_df.loc['Sharpe Ratio'] = (signals_df['Portfolio Daily Returns'].mean() * 252) / (signals_df['Portfolio Daily Returns'].std() * np.sqrt(252))

        #Calculate Sortino Ratio/Downside Return:
        sortino_ratio_df = signals_df[['Portfolio Daily Returns']].copy()
        sortino_ratio_df.loc[:,'Downside Returns'] = 0

        target = 0
        mask = sortino_ratio_df['Portfolio Daily Returns'] < target
        sortino_ratio_df.loc[mask, 'Downside Returns'] = sortino_ratio_df['Portfolio Daily Returns']**2
        down_stdev = np.sqrt(sortino_ratio_df['Downside Returns'].mean()) * np.sqrt(252)
        expected_return = sortino_ratio_df['Portfolio Daily Returns'].mean() * 252
        sortino_ratio = expected_return/down_stdev

        portfolio_evaluation_df.loc['Sortino Ratio'] = sortino_ratio


        return portfolio_evaluation_df
    
    def ulying_evaluation(self, signals_df):
        # Define evaluation metrics
        metrics = [
            'Annual Return',
            'Cumulative Returns',
            'Annual Volatility',
            'Sharpe Ratio',
            'Sortino Ratio'
        ]

        # Define column for backtest
        columns = ['Backtest']

        # Initialize the DataFrame with metrics as index and one column for backtest results
        portfolio_evaluation_df = pd.DataFrame(index=metrics, columns=columns)

        # Calculate cumulative returns
        portfolio_evaluation_df.loc['Cumulative Returns'] = signals_df['Portfolio Cumulative Returns'][-1]

        # Calculate annualized returns
        daily_returns_mean = signals_df['Portfolio Daily Returns'].mean()
        portfolio_evaluation_df.loc['Annual Return'] = (daily_returns_mean * 252)

        # Calculate annual volatility
        daily_returns_std = signals_df['Portfolio Daily Returns'].std()
        portfolio_evaluation_df.loc['Annual Volatility'] = daily_returns_std * np.sqrt(252)

        # Calculate Sharpe Ratio
        portfolio_evaluation_df.loc['Sharpe Ratio'] = (daily_returns_mean * 252) / (daily_returns_std * np.sqrt(252))

        # Calculate Sortino Ratio (downside deviation)
        sortino_ratio_df = signals_df[['Portfolio Daily Returns']].copy()
        sortino_ratio_df['Downside Returns'] = 0  # Initialize downside returns column

        # Target return (e.g., risk-free rate or 0 for simplicity)
        target = 0
        mask = sortino_ratio_df['Portfolio Daily Returns'] < target
        sortino_ratio_df.loc[mask, 'Downside Returns'] = sortino_ratio_df['Portfolio Daily Returns']**2

        # Downside standard deviation
        downside_stdev = np.sqrt(sortino_ratio_df['Downside Returns'].mean()) * np.sqrt(252)
        expected_annual_return = daily_returns_mean * 252
        sortino_ratio = expected_annual_return / downside_stdev if downside_stdev > 0 else np.nan

        portfolio_evaluation_df.loc['Sortino Ratio'] = sortino_ratio

        return portfolio_evaluation_df

    def underlying_evaluation(self, signals_df):
        underlying=pd.DataFrame()
        underlying["Returns"]=signals_df["Returns"]
        underlying["Portfolio Daily Returns"]=underlying["Returns"]
        underlying["Portfolio Daily Returns"].fillna(0,inplace=True)
        underlying['Portfolio Cumulative Returns']=(1 + underlying['Portfolio Daily Returns']).cumprod() - 1

        underlying_evaluation=self.ulying_evaluation(underlying)

        return underlying_evaluation    
    
    def algo_vs_underlying(self, signals_df):
        metrics = [
            'Annual Return',
            'Cumulative Returns',
            'Annual Volatility',
            'Sharpe Ratio',
            'Sortino Ratio']

        columns = ['Algo','Underlying']
        algo=self.algo_evaluation(signals_df)
        underlying=self.underlying_evaluation(signals_df)

        comparison_df=pd.DataFrame(index=metrics,columns=columns)
        comparison_df['Algo']=algo['Backtest']
        comparison_df['Underlying']=underlying['Backtest']

        return comparison_df
    
    def trade_evaluation(self, signals_df):
        
        trade_evaluation_df = pd.DataFrame(
            columns=[
                'Entry Date', 
                'Exit Date', 
                'Shares', 
                'Entry Share Price', 
                'Exit Share Price', 
                'Entry Portfolio Holding', 
                'Exit Portfolio Holding', 
                'Profit/Loss']
        )
        
        
        entry_date = ''
        exit_date = ''
        entry_portfolio_holding = 0
        exit_portfolio_holding = 0
        share_size = 0
        entry_share_price = 0
        exit_share_price = 0

        for index, row in signals_df.iterrows():
            if row['Entry/Exit'] == 1:
                entry_date = index
                entry_portfolio_holding = row['Portfolio Total']
                share_size = row['Entry/Exit Position']
                entry_share_price = row['Returns']

            elif row['Entry/Exit'] == -1:
                exit_date = index
                exit_portfolio_holding = abs(row['Portfolio Total'])
                exit_share_price = row['Returns']
                profit_loss = exit_portfolio_holding - entry_portfolio_holding
                trade_evaluation_df = trade_evaluation_df.append(
                    {
                        'Entry Date': entry_date,
                        'Exit Date': exit_date,
                        'Shares': share_size,
                        'Entry Share Price': entry_share_price,
                        'Exit Share Price': exit_share_price,
                        'Entry Portfolio Holding': entry_portfolio_holding,
                        'Exit Portfolio Holding': exit_portfolio_holding,
                        'Profit/Loss': profit_loss
                    },
                    ignore_index=True)

        return trade_evaluation_df
    
    def underlying_returns(self, signals_df):
        underlying=pd.DataFrame()
        underlying["Returns"]=signals_df["Returns"]
        underlying["Underlying Daily Returns"]=underlying["Returns"]
        underlying["Underlying Daily Returns"].fillna(0,inplace=True)
        underlying['Underlying Cumulative Returns']=(1 + underlying['Underlying Daily Returns']).cumprod() - 1
        underlying['Algo Cumulative Returns']=signals_df["Portfolio Cumulative Returns"] * 100

        graph_df=underlying[["Underlying Cumulative Returns", "Algo Cumulative Returns"]]

        return graph_df
    
class TranscendentalKernel():
    def __init__(self):
        pass

    def plot_pdf(self, log_returns):
        plt.figure(figsize=(14, 8))
        sns.set_style('whitegrid')  # Use a clean style

        # Generate a color palette
        colors = sns.color_palette("tab10", len(log_returns.columns))

        for i, column in enumerate(log_returns.columns):
            sns.kdeplot(
                log_returns[column], 
                label=column, 
                fill=True, 
                alpha=0.6, 
                color=colors[i],
                linewidth=1.5
            )

        # Add title and labels
        plt.title('Probability Density Function (PDF) for Log Returns', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Log Return', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Add a legend
        plt.legend(title='Tickers', title_fontsize=13, fontsize=12, loc='upper left', frameon=True)

        # Enhance grid appearance
        plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Show the plot
        plt.show()

    def percentile_to_percentile_mapping(self, num_samples, log_returns, vcv_matrix, pdf_dict):
        samples = np.random.multivariate_normal(log_returns.mean(), vcv_matrix, size=num_samples)

        transformed_samples = np.zeros_like(samples)

        for i, column in enumerate(log_returns.columns):
            returns = pdf_dict[column]['returns']
            pdf = pdf_dict[column]['pdf']

            cdf = np.cumsum(pdf) * np.diff(returns)[0]

            for j in range(num_samples):
                sample_value = samples[j, i]
                percentile = np.interp(sample_value, returns, cdf)
                transformed_samples[j, i] = np.interp(percentile, cdf, returns)

        transformed_samples_df = pd.DataFrame(transformed_samples, columns=log_returns.columns)
        return transformed_samples_df
    
    def calculate_var(self, ending_value, alpha=0.05):
        ranked_values = ending_value.rank(pct=True)
        var_value = ending_value[ranked_values <= alpha].max()
        return var_value
    
    def plot_var(self, ending_value):
        sorted_values = ending_value.sort_values()

        percentiles = np.linspace(0, 1, len(sorted_values))

        # Kernel Density Estimation (KDE)
        kde = stats.gaussian_kde(sorted_values)

        # Calculate VaR at 95% and 99%
        VaR_95_threshold = sorted_values.quantile(0.05)  # 5th percentile
        VaR_99_threshold = sorted_values.quantile(0.01)  # 1st percentile

        # Generate x values for plotting the KDE
        x = np.linspace(sorted_values.min(), sorted_values.max(), 1000)
        pdf_values = kde(x)

        # Plot setup
        plt.figure(figsize=(12, 6))
        
        # Plot the estimated PDF
        plt.plot(x, pdf_values, label='Estimated PDF', color='#1f77b4', linewidth=2)
        
        # Mark the VaR thresholds
        plt.axvline(VaR_95_threshold, color='#ff7f0e', linestyle='--', linewidth=2, label=f'VaR 95%: {VaR_95_threshold:,.2f}')
        plt.axvline(VaR_99_threshold, color='#d62728', linestyle='--', linewidth=2, label=f'VaR 99%: {VaR_99_threshold:,.2f}')
        
        # Highlight areas below VaR thresholds
        plt.fill_between(
            x=x,
            y1=pdf_values,
            y2=0,
            where=x <= VaR_95_threshold,
            color='#ff7f0e',
            alpha=0.3,
            label='Below 95% Threshold'
        )
        plt.fill_between(
            x=x,
            y1=pdf_values,
            y2=0,
            where=x <= VaR_99_threshold,
            color='#d62728',
            alpha=0.3,
            label='Below 99% Threshold'
        )

        # Chart styling
        plt.title('VaR of Ending Portfolio Value', fontsize=16, fontweight='bold')
        plt.xlabel('Portfolio Value', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        
        # Customizing the grid
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Tight layout for better spacing
        plt.tight_layout()

        # Show the legend and plot
        plt.legend(loc='best', fontsize=12, title='Value at Risk', title_fontsize=14)
        plt.show()

class PortfolioPerformance():
    def __init__(self):
        pass

    def plot_mdd(self, drawdown):
        plt.figure(figsize=(14, 8))

        plt.plot(drawdown, label="Drawdown", color="red", linewidth=2, alpha=0.8)

        plt.title("Portfolio Drawdown Over Time", fontsize=16, fontweight="bold", color="darkred")
        plt.xlabel("Date", fontsize=14, color="black")
        plt.ylabel("Drawdown (%)", fontsize=14, color="black")

        max_drawdown_date = drawdown.idxmin()
        max_drawdown_value = drawdown.min()
        plt.scatter(max_drawdown_date, max_drawdown_value, color="darkblue", s=100, label="Maximum Drawdown")
        plt.text(max_drawdown_date, max_drawdown_value, f"{max_drawdown_value:.2f}%", 
                color="darkblue", fontsize=12, fontweight="bold", ha="left", va="bottom")

        plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
        plt.legend(fontsize=12, loc="best", frameon=True, shadow=True, edgecolor="black")
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_performance(self, portfolio_cum_returns, ixn_returns, msci_returns, global_clean_energy_returns, initial_money):
        plt.figure(figsize=(14, 8))

        plt.plot(portfolio_cum_returns, label="Portfolio Cumulative Return", color="blue", linewidth=2)

        # SPDR Fund
        plt.plot(
            (1 + ixn_returns).cumprod() * initial_money,
            label="iShares Global Tech ETF",
            color="red",
            linestyle='--',
            linewidth=2
        )

        # MSCI ETF
        plt.plot(
            (1 + msci_returns).cumprod() * initial_money,
            label="iShares MSCI Emerging Markets ETF",
            color="green",
            linestyle='-.',
            linewidth=2
        )

        # Global Clean Energy ETF
        plt.plot(
            (1 + global_clean_energy_returns).cumprod() * initial_money,
            label="iShares Global Clean Energy ETF",
            color="orange",
            linestyle=':',
            linewidth=2
        )

        # Add titles, labels, legend, and grid
        plt.title("Portfolio vs. Market Performance", fontsize=16, fontweight='bold')
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Cumulative Returns ($)", fontsize=14)
        plt.legend(fontsize=12, loc='upper left', frameon=True, shadow=True)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # Format x-axis for better readability
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)

        # Tight layout for better spacing
        plt.tight_layout()

        # Show plot
        plt.show()
