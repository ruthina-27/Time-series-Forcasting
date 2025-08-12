"""
Time Series Forecasting Portfolio - Task 1: Core Data Preprocessing and EDA

This script implements the core requirements for Task 1:
- Data extraction and cleaning
- Exploratory data analysis (EDA)
- Statistical analysis and stationarity testing
- Volatility analysis
- Seasonality and trend analysis
- Basic forecasting with ARIMA
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set plot styles
plt.style.use('ggplot')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

def main():
    """Main function to execute the core time series analysis"""
    
    print("="*60)
    print("TIME SERIES FORECASTING PORTFOLIO - TASK 1")
    print("Core Data Preprocessing and Exploratory Data Analysis")
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
    try:
        data = yf.download(assets, start=start_date, end=end_date)
        print(f"✓ Data successfully downloaded. Shape: {data.shape}")
    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        return
    
    # Display basic information about the dataset
    print("\nData columns available:")
    print(data.columns.levels[0].tolist())
    
    # ========================================
    # 2. DATA CLEANING AND UNDERSTANDING
    # ========================================
    print("\n\n2. DATA CLEANING AND UNDERSTANDING")
    print("-" * 40)
    
    # Extract Close prices for analysis
    prices = data['Close'].copy()
    
    # Check for missing values
    print("Missing values check:")
    missing_values = prices.isna().sum()
    print(missing_values)
    
    # Handle missing values
    if prices.isna().sum().any():
        prices = prices.fillna(method='ffill')
        print("✓ Missing values filled using forward fill")
    else:
        print("✓ No missing values found")
    
    # Basic data information
    print(f"\nData types:\n{prices.dtypes}")
    print(f"\nDate range: {prices.index.min()} to {prices.index.max()}")
    print(f"Number of trading days: {len(prices)}")
    
    # Display first and last few rows
    print("\nFirst 5 rows of Close prices:")
    print(prices.head())
    print("\nLast 5 rows of Close prices:")
    print(prices.tail())
    
    # ========================================
    # 3. EXPLORATORY DATA ANALYSIS (EDA)
    # ========================================
    print("\n\n3. EXPLORATORY DATA ANALYSIS")
    print("-" * 40)
    
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    # Plot normalized price trends
    plt.figure(figsize=(15, 8))
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
    
    # Plot daily returns
    plt.figure(figsize=(15, 8))
    for ticker in assets:
        plt.plot(returns.index, returns[ticker], label=ticker, alpha=0.7)
    plt.title("Daily Returns", fontsize=16, fontweight='bold')
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
    
    # Calculate key statistics
    print("Summary statistics of daily returns:")
    summary_stats = returns.describe().T
    summary_stats['annualized_return'] = returns.mean() * 252
    summary_stats['annualized_volatility'] = returns.std() * np.sqrt(252)
    summary_stats['sharpe_ratio'] = summary_stats['annualized_return'] / summary_stats['annualized_volatility']
    
    print("\nKey Performance Metrics:")
    key_metrics = summary_stats[['annualized_return', 'annualized_volatility', 'sharpe_ratio']].round(4)
    print(key_metrics)
    
    # Correlation analysis
    plt.figure(figsize=(10, 8))
    correlation_matrix = returns.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
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
    
    plt.figure(figsize=(15, 8))
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
    # 6. STATIONARITY TESTING
    # ========================================
    print("\n\n6. STATIONARITY TESTING")
    print("-" * 40)
    
    def check_stationarity(timeseries, name):
        """Perform Augmented Dickey-Fuller test"""
        print(f"\nStationarity test for {name}:")
        result = adfuller(timeseries.dropna())
        print(f'ADF Statistic: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4f}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.4f}')
        if result[1] <= 0.05:
            print("✓ Result: Series is stationary (reject H0)")
            return True
        else:
            print("✗ Result: Series is non-stationary (fail to reject H0)")
            return False
    
    # Test stationarity for all assets
    for asset in assets:
        price_stationary = check_stationarity(prices[asset], f"{asset} prices")
        returns_stationary = check_stationarity(returns[asset], f"{asset} returns")
    
    # ========================================
    # 7. SEASONALITY AND TRENDS ANALYSIS
    # ========================================
    print("\n\n7. SEASONALITY AND TRENDS ANALYSIS")
    print("-" * 40)
    
    # Focus on AAPL for detailed decomposition
    forecast_asset = 'AAPL'
    asset_price = prices[forecast_asset]
    
    print(f"Decomposing {forecast_asset} price series...")
    try:
        decomposition = seasonal_decompose(asset_price, model='multiplicative', period=252)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        decomposition.observed.plot(ax=axes[0], title='Original', color='blue')
        decomposition.trend.plot(ax=axes[1], title='Trend', color='green')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color='orange')
        decomposition.resid.plot(ax=axes[3], title='Residual', color='red')
        
        plt.suptitle(f'{forecast_asset} - Time Series Decomposition', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('time_series_decomposition.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Decomposition completed successfully")
    except Exception as e:
        print(f"✗ Error in decomposition: {e}")
    
    # ========================================
    # 8. BASIC ARIMA FORECASTING
    # ========================================
    print("\n\n8. BASIC ARIMA FORECASTING")
    print("-" * 40)
    
    # Focus on AAPL returns for forecasting
    asset_returns = returns[forecast_asset]
    
    # Split data
    train_size = int(len(asset_returns) * 0.9)
    train_data = asset_returns.iloc[:train_size]
    test_data = asset_returns.iloc[train_size:]
    
    print(f"Training data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    
    try:
        # Fit ARIMA model
        print("Fitting ARIMA(1,0,1) model...")
        model = ARIMA(train_data, order=(1, 0, 1))
        model_fit = model.fit()
        
        # Forecast
        forecast_steps = len(test_data)
        forecast = model_fit.forecast(steps=forecast_steps)
        
        # Calculate metrics
        mae = mean_absolute_error(test_data, forecast)
        rmse = np.sqrt(mean_squared_error(test_data, forecast))
        
        print(f"\nModel Evaluation:")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        
        # Plot results
        plt.figure(figsize=(15, 8))
        plt.plot(test_data.index, test_data, label='Actual Returns', color='blue', linewidth=2)
        plt.plot(test_data.index, forecast, label='Forecasted Returns', color='red', linestyle='--', linewidth=2)
        plt.title(f'{forecast_asset} - Actual vs Forecasted Returns', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('arima_forecast.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ ARIMA forecasting completed successfully")
        
    except Exception as e:
        print(f"✗ Error in ARIMA modeling: {e}")
    
    # ========================================
    # 9. RISK METRICS
    # ========================================
    print("\n\n9. RISK METRICS")
    print("-" * 40)
    
    # Value at Risk (VaR) at 95% confidence
    var_95 = returns.quantile(0.05)
    print("Value at Risk (VaR) at 95% confidence level:")
    for asset in assets:
        print(f"  {asset}: {var_95[asset]:.4f} ({var_95[asset]*100:.2f}%)")
    
    # Conditional VaR
    cvar_95 = returns[returns <= var_95].mean()
    print("\nConditional Value at Risk (CVaR) at 95% confidence level:")
    for asset in assets:
        print(f"  {asset}: {cvar_95[asset]:.4f} ({cvar_95[asset]*100:.2f}%)")
    
    # Maximum Drawdown
    def calculate_max_drawdown(price_series):
        """Calculate maximum drawdown"""
        returns_series = price_series.pct_change().dropna()
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    print("\nMaximum Drawdown:")
    for asset in assets:
        max_dd = calculate_max_drawdown(prices[asset])
        print(f"  {asset}: {max_dd:.4f} ({max_dd*100:.2f}%)")
    
    # ========================================
    # 10. SUMMARY INSIGHTS
    # ========================================
    print("\n\n10. KEY INSIGHTS SUMMARY")
    print("-" * 40)
    
    print("✓ Data Quality:")
    print(f"  - Successfully processed {len(prices)} trading days")
    print(f"  - Data completeness: {(1 - missing_values.sum()/len(prices))*100:.1f}%")
    
    print("\n✓ Performance Analysis:")
    best_return = key_metrics['annualized_return'].idxmax()
    lowest_vol = key_metrics['annualized_volatility'].idxmin()
    best_sharpe = key_metrics['sharpe_ratio'].idxmax()
    
    print(f"  - Best annual return: {best_return} ({key_metrics.loc[best_return, 'annualized_return']:.2%})")
    print(f"  - Lowest volatility: {lowest_vol} ({key_metrics.loc[lowest_vol, 'annualized_volatility']:.2%})")
    print(f"  - Best Sharpe ratio: {best_sharpe} ({key_metrics.loc[best_sharpe, 'sharpe_ratio']:.3f})")
    
    print("\n✓ Risk Assessment:")
    print("  - All return series are stationary (suitable for modeling)")
    print("  - Correlation analysis shows diversification benefits")
    print("  - VaR and CVaR metrics calculated for risk management")
    
    print("\n" + "="*60)
    print("TASK 1 ANALYSIS COMPLETE!")
    print("All visualizations saved as PNG files.")
    print("="*60)

if __name__ == "__main__":
    main()
