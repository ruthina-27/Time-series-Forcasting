"""
Time Series Forecasting Portfolio - Task 1: Preprocess and Explore the Data

This script implements comprehensive data preprocessing and exploratory data analysis
for financial time series data, including:
- Data extraction and cleaning
- Exploratory data analysis (EDA)
- Statistical analysis and stationarity testing
- Volatility analysis
- Seasonality and trend analysis
- Portfolio optimization
- LSTM forecasting
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.optimize import minimize
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# For portfolio optimization
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier

import warnings
warnings.filterwarnings('ignore')

# Set plot styles
plt.style.use('ggplot')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def main():
    """Main function to execute the time series analysis pipeline"""
    
    print("="*60)
    print("TIME SERIES FORECASTING PORTFOLIO - TASK 1")
    print("Preprocess and Explore the Data")
    print("="*60)
    
    # ========================================
    # 1. DATA EXTRACTION AND LOADING
    # ========================================
    print("\n1. DATA EXTRACTION AND LOADING")
    print("-" * 40)
    
    # Define the assets we'll analyze
    assets = ['AAPL', 'AMZN', 'AGG']
    
    # Define the time period
    end_date = "2023-12-31"
    start_date = "2015-01-01"
    
    print(f"Assets to analyze: {assets}")
    print(f"Time period: {start_date} to {end_date}")
    
    # Fetch the data
    print("\nFetching data from Yahoo Finance...")
    data = yf.download(assets, start=start_date, end=end_date)
    print(f"Data shape: {data.shape}")
    
    # Display basic information about the dataset
    print("\nData columns:")
    print(data.columns.levels[0].tolist())
    print("\nFirst few rows of Close prices:")
    print(data['Close'].head())
    
    # ========================================
    # 2. DATA CLEANING AND UNDERSTANDING
    # ========================================
    print("\n\n2. DATA CLEANING AND UNDERSTANDING")
    print("-" * 40)
    
    # Extract just the Close prices for simplicity
    prices = data['Close'].copy()
    
    # Check for missing values
    print("Missing values in each column:")
    missing_values = prices.isna().sum()
    print(missing_values)
    
    # Handle missing values by forward filling
    if prices.isna().sum().any():
        prices = prices.fillna(method='ffill')
        print("Missing values filled using forward fill method")
    else:
        print("No missing values found")
    
    # Check data types and basic info
    print(f"\nData types:\n{prices.dtypes}")
    print(f"\nDate range: {prices.index.min()} to {prices.index.max()}")
    print(f"Number of trading days: {len(prices)}")
    
    # ========================================
    # 3. EXPLORATORY DATA ANALYSIS (EDA)
    # ========================================
    print("\n\n3. EXPLORATORY DATA ANALYSIS")
    print("-" * 40)
    
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    # Plot the normalized price trends
    plt.figure(figsize=(15, 10))
    for asset in assets:
        normalized_price = prices[asset] / prices[asset].iloc[0]
        plt.plot(prices.index, normalized_price, label=asset, linewidth=2)
    plt.title('Normalized Price Trends (Base = 1.0)', fontsize=16, fontweight='bold')
    plt.ylabel('Normalized Price')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('normalized_price_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot daily percentage changes
    plt.figure(figsize=(15, 10))
    for ticker in assets:
        plt.plot(returns.index, returns[ticker], label=ticker, alpha=0.7)
    plt.title("Daily Percentage Change", fontsize=16, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('daily_returns.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================
    # 4. STATISTICAL ANALYSIS
    # ========================================
    print("\n\n4. STATISTICAL ANALYSIS")
    print("-" * 40)
    
    # Calculate and display key statistics
    print("Summary statistics of daily returns:")
    summary_stats = returns.describe().T
    summary_stats['annualized_return'] = returns.mean() * 252
    summary_stats['annualized_volatility'] = returns.std() * np.sqrt(252)
    summary_stats['sharpe_ratio'] = summary_stats['annualized_return'] / summary_stats['annualized_volatility']
    
    print("\nKey Performance Metrics:")
    print(summary_stats[['annualized_return', 'annualized_volatility', 'sharpe_ratio']].round(4))
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = returns.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f')
    plt.title('Correlation Matrix of Daily Returns', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================
    # 5. VOLATILITY ANALYSIS
    # ========================================
    print("\n\n5. VOLATILITY ANALYSIS")
    print("-" * 40)
    
    # Calculate rolling volatility (30-day window)
    rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
    
    plt.figure(figsize=(15, 10))
    for asset in assets:
        plt.plot(rolling_vol.index, rolling_vol[asset], label=f'{asset} (30-day)', linewidth=2)
    plt.title('Rolling Volatility (30-day window)', fontsize=16, fontweight='bold')
    plt.ylabel('Annualized Volatility')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rolling_volatility.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================
    # 6. TIME SERIES ANALYSIS AND STATIONARITY
    # ========================================
    print("\n\n6. TIME SERIES ANALYSIS AND STATIONARITY")
    print("-" * 40)
    
    # Focus on AAPL for detailed analysis
    forecast_asset = 'AAPL'
    asset_price = prices[forecast_asset]
    asset_returns = returns[forecast_asset]
    
    def check_stationarity(timeseries, name):
        """Perform Augmented Dickey-Fuller test for stationarity"""
        print(f"\nStationarity test for {name}:")
        result = adfuller(timeseries.dropna())
        print(f'ADF Statistic: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4f}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.4f}')
        if result[1] <= 0.05:
            print("Result: The series is stationary (reject H0)")
            return True
        else:
            print("Result: The series is non-stationary (fail to reject H0)")
            return False
    
    # Test stationarity
    price_stationary = check_stationarity(asset_price, f"{forecast_asset} prices")
    returns_stationary = check_stationarity(asset_returns, f"{forecast_asset} returns")
    
    # ========================================
    # 7. SEASONALITY AND TRENDS ANALYSIS
    # ========================================
    print("\n\n7. SEASONALITY AND TRENDS ANALYSIS")
    print("-" * 40)
    
    # Time series decomposition
    print(f"Decomposing {forecast_asset} price series...")
    decomposition = seasonal_decompose(asset_price, model='multiplicative', period=252)  # 252 trading days
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    decomposition.observed.plot(ax=axes[0], title='Original', color='blue')
    decomposition.trend.plot(ax=axes[1], title='Trend', color='green')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color='orange')
    decomposition.resid.plot(ax=axes[3], title='Residual', color='red')
    
    plt.suptitle(f'{forecast_asset} - Time Series Decomposition', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('time_series_decomposition.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================
    # 8. ARIMA MODELING AND FORECASTING
    # ========================================
    print("\n\n8. ARIMA MODELING AND FORECASTING")
    print("-" * 40)
    
    # Prepare training data (use 90% for training)
    train_size = int(len(asset_returns) * 0.9)
    train_data = asset_returns.iloc[:train_size]
    test_data = asset_returns.iloc[train_size:]
    
    print(f"Training data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    
    # Fit ARIMA model
    print("Fitting ARIMA model...")
    p, d, q = 1, 0, 1  # Example parameters
    model = ARIMA(train_data, order=(p, d, q))
    model_fit = model.fit()
    
    print("\nARIMA Model Summary:")
    print(model_fit.summary())
    
    # Forecast and evaluate
    forecast_steps = len(test_data)
    forecast = model_fit.forecast(steps=forecast_steps)
    
    # Calculate error metrics
    mae = mean_absolute_error(test_data, forecast)
    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    print(f"\nModel Evaluation on Test Data:")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    
    # Plot actual vs forecast returns
    plt.figure(figsize=(15, 8))
    plt.plot(test_data.index, test_data, label='Actual Returns', color='blue', linewidth=2)
    plt.plot(test_data.index, forecast, label='Forecasted Returns', color='red', linestyle='--', linewidth=2)
    plt.title(f'{forecast_asset} - Actual vs Forecasted Returns', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('arima_forecast_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate future forecasts
    future_steps = 30
    future_forecast = model_fit.forecast(steps=future_steps)
    future_dates = pd.date_range(start=asset_returns.index[-1] + pd.Timedelta(days=1), periods=future_steps)
    
    plt.figure(figsize=(15, 8))
    plt.plot(asset_returns.index[-90:], asset_returns.iloc[-90:], label='Historical Returns', color='blue', linewidth=2)
    plt.plot(future_dates, future_forecast, label='Future Returns Forecast', color='red', linestyle='--', linewidth=2)
    plt.title(f'{forecast_asset} - Returns Forecast for Next {future_steps} Trading Days', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.axvline(x=asset_returns.index[-1], color='green', linestyle='-', alpha=0.7, label='Forecast Start')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('future_forecast.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================
    # 9. PORTFOLIO OPTIMIZATION
    # ========================================
    print("\n\n9. PORTFOLIO OPTIMIZATION")
    print("-" * 40)
    
    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(prices)
    cov_matrix = risk_models.sample_cov(prices)
    
    # Generate random portfolios for efficient frontier
    num_portfolios = 10000
    n_assets = len(assets)
    
    np.random.seed(42)
    weights = np.random.dirichlet(np.ones(n_assets), num_portfolios)
    portfolio_returns = np.dot(weights, mu)
    portfolio_stddevs = np.sqrt(np.diag(weights @ cov_matrix @ weights.T))
    sharpe_ratios = portfolio_returns / portfolio_stddevs
    
    # Find optimal portfolios using PyPortfolioOpt
    # Minimum Volatility Portfolio
    ef_minvol = EfficientFrontier(mu, cov_matrix)
    ef_minvol.min_volatility()
    min_vol_weights = ef_minvol.clean_weights()
    min_vol_return, min_vol_stddev, _ = ef_minvol.portfolio_performance()
    
    # Maximum Sharpe Portfolio
    ef_maxsharpe = EfficientFrontier(mu, cov_matrix)
    ef_maxsharpe.max_sharpe()
    max_sharpe_weights = ef_maxsharpe.clean_weights()
    max_sharpe_return, max_sharpe_stddev, _ = ef_maxsharpe.portfolio_performance()
    
    # Plot Efficient Frontier
    plt.figure(figsize=(15, 10))
    plt.scatter(portfolio_stddevs, portfolio_returns, c=sharpe_ratios,
                marker='o', cmap='viridis', s=10, alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    
    # Plot optimal portfolios
    plt.scatter(min_vol_stddev, min_vol_return,
                marker='*', color='r', s=500, label='Minimum Volatility', edgecolors='black')
    plt.scatter(max_sharpe_stddev, max_sharpe_return,
                marker='*', color='g', s=500, label='Maximum Sharpe Ratio', edgecolors='black')
    
    # Plot individual assets
    for i, asset in enumerate(assets):
        asset_vol = np.sqrt(cov_matrix.iloc[i, i])
        asset_ret = mu[i]
        plt.scatter(asset_vol, asset_ret, marker='o', s=200, color='black')
        plt.annotate(asset, (asset_vol*1.01, asset_ret*1.01), fontsize=12, fontweight='bold')
    
    plt.title('Efficient Frontier', fontsize=16, fontweight='bold')
    plt.xlabel('Expected Volatility (Standard Deviation)')
    plt.ylabel('Expected Annual Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print Portfolio Results
    print("\nMINIMUM VOLATILITY PORTFOLIO:")
    print(f"Expected Return: {min_vol_return:.2%}")
    print(f"Expected Volatility: {min_vol_stddev:.2%}")
    print(f"Sharpe Ratio: {min_vol_return/min_vol_stddev:.2f}")
    print("Asset Allocation:")
    for asset in assets:
        print(f"  {asset}: {min_vol_weights[asset]:.2%}")
    
    print("\nMAXIMUM SHARPE RATIO PORTFOLIO:")
    print(f"Expected Return: {max_sharpe_return:.2%}")
    print(f"Expected Volatility: {max_sharpe_stddev:.2%}")
    print(f"Sharpe Ratio: {max_sharpe_return/max_sharpe_stddev:.2f}")
    print("Asset Allocation:")
    for asset in assets:
        print(f"  {asset}: {max_sharpe_weights[asset]:.2%}")
    
    # Plot Asset Allocations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    min_vol_values = [min_vol_weights[asset] for asset in assets]
    ax1.pie(min_vol_values, labels=assets, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Minimum Volatility Portfolio', fontsize=14, fontweight='bold')
    
    max_sharpe_values = [max_sharpe_weights[asset] for asset in assets]
    ax2.pie(max_sharpe_values, labels=assets, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Maximum Sharpe Ratio Portfolio', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('portfolio_allocations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================
    # 10. RISK METRICS CALCULATION
    # ========================================
    print("\n\n10. RISK METRICS")
    print("-" * 40)
    
    # Calculate Value at Risk (VaR) at 95% confidence level
    var_95 = returns.quantile(0.05)
    print("Value at Risk (VaR) at 95% confidence level:")
    for asset in assets:
        print(f"  {asset}: {var_95[asset]:.4f} ({var_95[asset]*100:.2f}%)")
    
    # Calculate Conditional Value at Risk (CVaR)
    cvar_95 = returns[returns <= var_95].mean()
    print("\nConditional Value at Risk (CVaR) at 95% confidence level:")
    for asset in assets:
        print(f"  {asset}: {cvar_95[asset]:.4f} ({cvar_95[asset]*100:.2f}%)")
    
    # Maximum Drawdown calculation
    def calculate_max_drawdown(price_series):
        """Calculate maximum drawdown"""
        cumulative = (1 + price_series.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    print("\nMaximum Drawdown:")
    for asset in assets:
        max_dd = calculate_max_drawdown(prices[asset])
        print(f"  {asset}: {max_dd:.4f} ({max_dd*100:.2f}%)")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("All visualizations have been saved as PNG files.")
    print("="*60)

if __name__ == "__main__":
    main()
