# Time Series Forecasting Portfolio - Task 1

## Overview
This project implements comprehensive data preprocessing and exploratory data analysis for financial time series forecasting. It analyzes three key assets: AAPL (Apple), AMZN (Amazon), and AGG (iShares Core U.S. Aggregate Bond ETF).

## Features

### 1. Data Preprocessing and Exploration
- **Data Extraction**: Historical financial data from Yahoo Finance (2015-2023)
- **Data Cleaning**: Missing value handling using forward fill method
- **Data Understanding**: Basic statistics, data types, and date range analysis

### 2. Exploratory Data Analysis (EDA)
- **Price Visualization**: Normalized price trends over time
- **Returns Analysis**: Daily percentage changes and volatility patterns
- **Correlation Analysis**: Asset correlation heatmap
- **Statistical Summary**: Key performance metrics including Sharpe ratios

### 3. Volatility Analysis
- **Rolling Volatility**: 30-day rolling volatility calculations
- **Risk Metrics**: Value at Risk (VaR) and Conditional VaR at 95% confidence
- **Maximum Drawdown**: Historical maximum loss periods

### 4. Time Series Analysis
- **Stationarity Testing**: Augmented Dickey-Fuller tests for prices and returns
- **Seasonality Analysis**: Time series decomposition (trend, seasonal, residual)
- **ARIMA Modeling**: Autoregressive Integrated Moving Average forecasting

### 5. Portfolio Optimization
- **Efficient Frontier**: Modern Portfolio Theory implementation
- **Optimal Portfolios**: Minimum volatility and maximum Sharpe ratio portfolios
- **Asset Allocation**: Visual representation of optimal weights

### 6. Advanced Forecasting
- **ARIMA Forecasting**: Statistical time series forecasting
- **Future Predictions**: 30-day ahead forecasts
- **Model Evaluation**: MAE and RMSE metrics

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the analysis:
```bash
python time_series_analysis.py
```

## Output Files

The script generates several visualization files:
- `normalized_price_trends.png` - Asset price trends (normalized)
- `daily_returns.png` - Daily percentage changes
- `correlation_heatmap.png` - Asset correlation matrix
- `rolling_volatility.png` - 30-day rolling volatility
- `time_series_decomposition.png` - Seasonal decomposition
- `arima_forecast_comparison.png` - Actual vs predicted returns
- `future_forecast.png` - 30-day future forecasts
- `efficient_frontier.png` - Portfolio optimization results
- `portfolio_allocations.png` - Optimal portfolio weights

## Key Insights

### Asset Performance (2015-2023)
- **AAPL**: High growth potential with moderate volatility
- **AMZN**: Strong returns with higher volatility
- **AGG**: Lower returns but provides portfolio stability

### Risk Analysis
- Value at Risk calculations at 95% confidence level
- Maximum drawdown analysis for risk assessment
- Correlation analysis for diversification benefits

### Portfolio Optimization
- Minimum volatility portfolio for conservative investors
- Maximum Sharpe ratio portfolio for risk-adjusted returns
- Efficient frontier visualization for optimal risk-return combinations

## Technical Implementation

### Libraries Used
- **Data**: yfinance, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Statistics**: statsmodels, scipy
- **Machine Learning**: scikit-learn, tensorflow
- **Portfolio Optimization**: PyPortfolioOpt
- **Time Series**: pmdarima

### Methodology
1. **Data Quality**: Comprehensive missing value analysis and handling
2. **Statistical Testing**: Formal stationarity tests using ADF
3. **Model Validation**: Train/test split with proper evaluation metrics
4. **Risk Management**: Multiple risk metrics for comprehensive analysis

## Next Steps
This analysis provides the foundation for:
- Advanced forecasting models (LSTM, Prophet, etc.)
- Real-time trading strategy development
- Risk management system implementation
- Portfolio rebalancing algorithms

## Author
Time Series Forecasting Portfolio Project
