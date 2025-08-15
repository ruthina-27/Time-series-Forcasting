"""
Time Series Forecasting Portfolio - Task 4: Optimize Portfolio Based on Forecast

This script implements Modern Portfolio Theory (MPT) to optimize a portfolio containing:
- TSLA: Using forecasted returns from Task 2/3
- BND: Using historical average returns (bond ETF)
- SPY: Using historical average returns (S&P 500 ETF)

Key components:
- Efficient Frontier generation
- Maximum Sharpe Ratio portfolio identification
- Minimum Volatility portfolio identification
- Portfolio recommendation with justification
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statistical models for getting TSLA forecast
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

warnings.filterwarnings('ignore')

# Set plot styles
plt.style.use('ggplot')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

class PortfolioOptimizer:
    """
    Modern Portfolio Theory implementation for optimizing portfolio
    based on forecasted and historical returns.
    """
    
    def __init__(self, symbols=['TSLA', 'BND', 'SPY'], start_date='2020-01-01', end_date='2025-01-15'):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.returns = None
        self.expected_returns = None
        self.cov_matrix = None
        self.risk_free_rate = 0.045  # Current 10-year Treasury rate (~4.5%)
        
    def load_historical_data(self):
        """Load historical data for all assets"""
        print("="*60)
        print("LOADING HISTORICAL DATA FOR PORTFOLIO ASSETS")
        print("="*60)
        
        for symbol in self.symbols:
            try:
                print(f"Downloading {symbol} data...")
                data = yf.download(symbol, start=self.start_date, end=self.end_date)
                self.data[symbol] = data['Close'].fillna(method='ffill').fillna(method='bfill')
                print(f"✓ {symbol}: {len(self.data[symbol])} data points")
                
            except Exception as e:
                print(f"✗ Error downloading {symbol}: {e}")
                return False
        
        # Create combined dataframe
        self.price_data = pd.DataFrame(self.data)
        self.price_data = self.price_data.dropna()
        
        # Calculate daily returns
        self.returns = self.price_data.pct_change().dropna()
        
        print(f"\n✓ Combined dataset: {len(self.price_data)} observations")
        print(f"Date range: {self.price_data.index.min()} to {self.price_data.index.max()}")
        
        return True
    
    def get_tsla_forecast_return(self):
        """Get TSLA expected return from best forecasting model"""
        print("\n" + "="*60)
        print("GENERATING TSLA FORECAST FOR EXPECTED RETURN")
        print("="*60)
        
        # Use TSLA price data
        tsla_prices = self.data['TSLA']
        
        # Split data for model evaluation
        split_point = int(len(tsla_prices) * 0.9)
        train_data = tsla_prices.iloc[:split_point]
        test_data = tsla_prices.iloc[split_point:]
        
        models_performance = {}
        
        # Test ARIMA model
        try:
            print("Testing ARIMA model for TSLA forecast...")
            auto_model = auto_arima(
                train_data,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False
            )
            
            # Forecast on test set for evaluation
            arima_forecast = auto_model.predict(n_periods=len(test_data))
            arima_mae = np.mean(np.abs(test_data - arima_forecast))
            
            models_performance['ARIMA'] = {
                'model': auto_model,
                'mae': arima_mae,
                'order': auto_model.order
            }
            
            print(f"✓ ARIMA {auto_model.order} - MAE: ${arima_mae:.2f}")
            
        except Exception as e:
            print(f"✗ ARIMA model failed: {e}")
        
        # Test SARIMA model
        try:
            print("Testing SARIMA model for TSLA forecast...")
            sarima_model = SARIMAX(
                train_data,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            
            sarima_forecast = sarima_model.forecast(steps=len(test_data))
            sarima_mae = np.mean(np.abs(test_data - sarima_forecast))
            
            models_performance['SARIMA'] = {
                'model': sarima_model,
                'mae': sarima_mae
            }
            
            print(f"✓ SARIMA - MAE: ${sarima_mae:.2f}")
            
        except Exception as e:
            print(f"✗ SARIMA model failed: {e}")
        
        # Select best model and generate forecast
        if models_performance:
            best_model_name = min(models_performance.keys(), key=lambda k: models_performance[k]['mae'])
            best_model = models_performance[best_model_name]['model']
            
            print(f"\n🏆 Best Model: {best_model_name}")
            
            # Retrain on full dataset
            if best_model_name == 'ARIMA':
                order = models_performance[best_model_name]['order']
                final_model = ARIMA(tsla_prices, order=order).fit()
            else:  # SARIMA
                final_model = SARIMAX(
                    tsla_prices,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
            
            # Generate 252-day (1 year) forecast
            forecast_result = final_model.get_forecast(steps=252)
            forecast_prices = forecast_result.predicted_mean
            
            # Calculate expected annual return
            current_price = tsla_prices.iloc[-1]
            forecast_end_price = forecast_prices.iloc[-1]
            tsla_expected_return = (forecast_end_price - current_price) / current_price
            
            print(f"Current TSLA Price: ${current_price:.2f}")
            print(f"Forecast End Price: ${forecast_end_price:.2f}")
            print(f"TSLA Expected Annual Return: {tsla_expected_return:.1%}")
            
            return tsla_expected_return
        
        else:
            print("✗ No models succeeded, using historical average")
            return self.returns['TSLA'].mean() * 252
    
    def calculate_expected_returns(self):
        """Calculate expected returns for all assets"""
        print("\n" + "="*60)
        print("CALCULATING EXPECTED RETURNS")
        print("="*60)
        
        # Get TSLA forecast return
        tsla_return = self.get_tsla_forecast_return()
        
        # Calculate historical returns for BND and SPY (annualized)
        bnd_return = self.returns['BND'].mean() * 252
        spy_return = self.returns['SPY'].mean() * 252
        
        self.expected_returns = np.array([tsla_return, bnd_return, spy_return])
        
        print(f"\n📊 EXPECTED ANNUAL RETURNS:")
        for i, symbol in enumerate(self.symbols):
            print(f"  {symbol}: {self.expected_returns[i]:.1%}")
        
        return True
    
    def calculate_covariance_matrix(self):
        """Calculate covariance matrix from historical returns"""
        print("\n" + "="*60)
        print("CALCULATING COVARIANCE MATRIX")
        print("="*60)
        
        # Annualize the covariance matrix
        self.cov_matrix = self.returns.cov() * 252
        
        print("Annualized Covariance Matrix:")
        print(self.cov_matrix.round(4))
        
        # Calculate correlation matrix for interpretation
        corr_matrix = self.returns.corr()
        print(f"\nCorrelation Matrix:")
        print(corr_matrix.round(3))
        
        return True
    
    def portfolio_performance(self, weights):
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        portfolio_return = np.sum(weights * self.expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def negative_sharpe_ratio(self, weights):
        """Objective function for maximizing Sharpe ratio"""
        return -self.portfolio_performance(weights)[2]
    
    def portfolio_volatility(self, weights):
        """Objective function for minimizing volatility"""
        return self.portfolio_performance(weights)[1]
    
    def generate_efficient_frontier(self, num_portfolios=100):
        """Generate the efficient frontier"""
        print("\n" + "="*60)
        print("GENERATING EFFICIENT FRONTIER")
        print("="*60)
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.symbols)))
        
        # Generate target returns
        min_ret = np.min(self.expected_returns)
        max_ret = np.max(self.expected_returns)
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            # Add constraint for target return
            cons = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(x * self.expected_returns) - target_return}
            ]
            
            # Minimize volatility for given return
            result = minimize(
                self.portfolio_volatility,
                x0=np.array([1/len(self.symbols)] * len(self.symbols)),
                method='SLSQP',
                bounds=bounds,
                constraints=cons
            )
            
            if result.success:
                ret, vol, sharpe = self.portfolio_performance(result.x)
                efficient_portfolios.append({
                    'return': ret,
                    'volatility': vol,
                    'sharpe': sharpe,
                    'weights': result.x
                })
        
        self.efficient_frontier = pd.DataFrame(efficient_portfolios)
        print(f"✓ Generated {len(self.efficient_frontier)} efficient portfolios")
        
        return True
    
    def find_optimal_portfolios(self):
        """Find Maximum Sharpe Ratio and Minimum Volatility portfolios"""
        print("\n" + "="*60)
        print("FINDING OPTIMAL PORTFOLIOS")
        print("="*60)
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.symbols)))
        initial_guess = np.array([1/len(self.symbols)] * len(self.symbols))
        
        # Find Maximum Sharpe Ratio Portfolio
        max_sharpe_result = minimize(
            self.negative_sharpe_ratio,
            x0=initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if max_sharpe_result.success:
            ret, vol, sharpe = self.portfolio_performance(max_sharpe_result.x)
            self.max_sharpe_portfolio = {
                'weights': max_sharpe_result.x,
                'return': ret,
                'volatility': vol,
                'sharpe': sharpe
            }
            
            print("🎯 MAXIMUM SHARPE RATIO PORTFOLIO:")
            for i, symbol in enumerate(self.symbols):
                print(f"  {symbol}: {self.max_sharpe_portfolio['weights'][i]:.1%}")
            print(f"  Expected Return: {ret:.1%}")
            print(f"  Volatility: {vol:.1%}")
            print(f"  Sharpe Ratio: {sharpe:.3f}")
        
        # Find Minimum Volatility Portfolio
        min_vol_result = minimize(
            self.portfolio_volatility,
            x0=initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if min_vol_result.success:
            ret, vol, sharpe = self.portfolio_performance(min_vol_result.x)
            self.min_vol_portfolio = {
                'weights': min_vol_result.x,
                'return': ret,
                'volatility': vol,
                'sharpe': sharpe
            }
            
            print("\n🛡️ MINIMUM VOLATILITY PORTFOLIO:")
            for i, symbol in enumerate(self.symbols):
                print(f"  {symbol}: {self.min_vol_portfolio['weights'][i]:.1%}")
            print(f"  Expected Return: {ret:.1%}")
            print(f"  Volatility: {vol:.1%}")
            print(f"  Sharpe Ratio: {sharpe:.3f}")
        
        return True
    
    def create_visualizations(self):
        """Create comprehensive portfolio optimization visualizations"""
        print("\n" + "="*60)
        print("CREATING PORTFOLIO VISUALIZATIONS")
        print("="*60)
        
        # Main visualization with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Plot 1: Efficient Frontier
        ax1 = axes[0, 0]
        scatter = ax1.scatter(
            self.efficient_frontier['volatility'] * 100,
            self.efficient_frontier['return'] * 100,
            c=self.efficient_frontier['sharpe'],
            cmap='viridis',
            alpha=0.7,
            s=50
        )
        
        # Mark optimal portfolios
        ax1.scatter(
            self.max_sharpe_portfolio['volatility'] * 100,
            self.max_sharpe_portfolio['return'] * 100,
            marker='*', s=500, c='red', label='Max Sharpe Ratio'
        )
        
        ax1.scatter(
            self.min_vol_portfolio['volatility'] * 100,
            self.min_vol_portfolio['return'] * 100,
            marker='*', s=500, c='blue', label='Min Volatility'
        )
        
        ax1.set_xlabel('Volatility (%)')
        ax1.set_ylabel('Expected Return (%)')
        ax1.set_title('Efficient Frontier with Optimal Portfolios', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Sharpe Ratio')
        
        # Plot 2: Asset allocation comparison
        ax2 = axes[0, 1]
        portfolios = ['Max Sharpe', 'Min Volatility']
        weights_data = [
            self.max_sharpe_portfolio['weights'],
            self.min_vol_portfolio['weights']
        ]
        
        x = np.arange(len(portfolios))
        width = 0.25
        
        for i, symbol in enumerate(self.symbols):
            values = [weights[i] for weights in weights_data]
            ax2.bar(x + i * width, values, width, label=symbol)
        
        ax2.set_xlabel('Portfolio Type')
        ax2.set_ylabel('Weight')
        ax2.set_title('Portfolio Allocations Comparison', fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(portfolios)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Risk-Return scatter of individual assets
        ax3 = axes[1, 0]
        individual_vols = [np.sqrt(self.cov_matrix.iloc[i, i]) * 100 for i in range(len(self.symbols))]
        individual_rets = [ret * 100 for ret in self.expected_returns]
        
        ax3.scatter(individual_vols, individual_rets, s=200, alpha=0.7)
        
        for i, symbol in enumerate(self.symbols):
            ax3.annotate(symbol, (individual_vols[i], individual_rets[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax3.set_xlabel('Volatility (%)')
        ax3.set_ylabel('Expected Return (%)')
        ax3.set_title('Individual Asset Risk-Return Profile', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Portfolio metrics comparison
        ax4 = axes[1, 1]
        metrics = ['Return (%)', 'Volatility (%)', 'Sharpe Ratio']
        max_sharpe_metrics = [
            self.max_sharpe_portfolio['return'] * 100,
            self.max_sharpe_portfolio['volatility'] * 100,
            self.max_sharpe_portfolio['sharpe']
        ]
        min_vol_metrics = [
            self.min_vol_portfolio['return'] * 100,
            self.min_vol_portfolio['volatility'] * 100,
            self.min_vol_portfolio['sharpe']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax4.bar(x - width/2, max_sharpe_metrics, width, label='Max Sharpe', alpha=0.8)
        ax4.bar(x + width/2, min_vol_metrics, width, label='Min Volatility', alpha=0.8)
        
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Value')
        ax4.set_title('Portfolio Metrics Comparison', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('portfolio_optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        corr_matrix = self.returns.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('Asset Correlation Matrix', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualizations created and saved")
        return True
    
    def recommend_portfolio(self):
        """Provide portfolio recommendation with justification"""
        print("\n" + "="*80)
        print("PORTFOLIO RECOMMENDATION & ANALYSIS")
        print("="*80)
        
        # Analyze both portfolios
        max_sharpe = self.max_sharpe_portfolio
        min_vol = self.min_vol_portfolio
        
        print(f"📊 PORTFOLIO COMPARISON:")
        print(f"\n🎯 Maximum Sharpe Ratio Portfolio:")
        print(f"  Expected Return: {max_sharpe['return']:.1%}")
        print(f"  Volatility: {max_sharpe['volatility']:.1%}")
        print(f"  Sharpe Ratio: {max_sharpe['sharpe']:.3f}")
        print(f"  TSLA Weight: {max_sharpe['weights'][0]:.1%}")
        print(f"  BND Weight: {max_sharpe['weights'][1]:.1%}")
        print(f"  SPY Weight: {max_sharpe['weights'][2]:.1%}")
        
        print(f"\n🛡️ Minimum Volatility Portfolio:")
        print(f"  Expected Return: {min_vol['return']:.1%}")
        print(f"  Volatility: {min_vol['volatility']:.1%}")
        print(f"  Sharpe Ratio: {min_vol['sharpe']:.3f}")
        print(f"  TSLA Weight: {min_vol['weights'][0]:.1%}")
        print(f"  BND Weight: {min_vol['weights'][1]:.1%}")
        print(f"  SPY Weight: {min_vol['weights'][2]:.1%}")
        
        # Determine recommendation based on analysis
        sharpe_diff = max_sharpe['sharpe'] - min_vol['sharpe']
        vol_diff = max_sharpe['volatility'] - min_vol['volatility']
        ret_diff = max_sharpe['return'] - min_vol['return']
        
        print(f"\n🎯 RECOMMENDATION ANALYSIS:")
        
        if sharpe_diff > 0.2 and ret_diff > 0.03:  # Significant Sharpe advantage
            recommended = max_sharpe
            portfolio_type = "Maximum Sharpe Ratio"
            justification = [
                f"Superior risk-adjusted returns (Sharpe: {max_sharpe['sharpe']:.3f} vs {min_vol['sharpe']:.3f})",
                f"Higher expected return ({max_sharpe['return']:.1%} vs {min_vol['return']:.1%})",
                f"Acceptable additional volatility ({vol_diff:.1%} increase) for {ret_diff:.1%} extra return",
                "Optimal for investors seeking maximum risk-adjusted performance"
            ]
        elif min_vol['volatility'] < 0.12:  # Very low volatility
            recommended = min_vol
            portfolio_type = "Minimum Volatility"
            justification = [
                f"Excellent risk management with only {min_vol['volatility']:.1%} volatility",
                f"Still provides positive expected returns ({min_vol['return']:.1%})",
                f"Suitable for risk-averse investors or uncertain market conditions",
                "Strong diversification benefits from bond allocation"
            ]
        else:
            # Create a balanced portfolio between the two
            balanced_weights = (max_sharpe['weights'] + min_vol['weights']) / 2
            balanced_ret, balanced_vol, balanced_sharpe = self.portfolio_performance(balanced_weights)
            
            recommended = {
                'weights': balanced_weights,
                'return': balanced_ret,
                'volatility': balanced_vol,
                'sharpe': balanced_sharpe
            }
            portfolio_type = "Balanced"
            justification = [
                "Balanced approach between risk and return optimization",
                f"Moderate volatility ({balanced_vol:.1%}) with solid returns ({balanced_ret:.1%})",
                f"Good Sharpe ratio ({balanced_sharpe:.3f}) suitable for most investors",
                "Diversified allocation across all asset classes"
            ]
        
        print(f"\n🏆 RECOMMENDED PORTFOLIO: {portfolio_type}")
        print(f"\n📈 FINAL PORTFOLIO ALLOCATION:")
        for i, symbol in enumerate(self.symbols):
            print(f"  {symbol}: {recommended['weights'][i]:.1%}")
        
        print(f"\n📊 EXPECTED PERFORMANCE:")
        print(f"  Annual Return: {recommended['return']:.1%}")
        print(f"  Annual Volatility: {recommended['volatility']:.1%}")
        print(f"  Sharpe Ratio: {recommended['sharpe']:.3f}")
        
        print(f"\n💡 JUSTIFICATION:")
        for i, reason in enumerate(justification, 1):
            print(f"  {i}. {reason}")
        
        # Risk assessment
        print(f"\n⚠️ RISK CONSIDERATIONS:")
        if recommended['weights'][0] > 0.3:  # High TSLA allocation
            print(f"  • High TSLA allocation ({recommended['weights'][0]:.1%}) increases portfolio volatility")
        if recommended['weights'][1] < 0.2:  # Low bond allocation
            print(f"  • Low bond allocation may reduce portfolio stability during market stress")
        if recommended['volatility'] > 0.15:
            print(f"  • Portfolio volatility ({recommended['volatility']:.1%}) requires tolerance for price swings")
        
        print(f"\n📝 IMPLEMENTATION NOTES:")
        print(f"  • Rebalance quarterly to maintain target allocations")
        print(f"  • Monitor TSLA forecast updates for allocation adjustments")
        print(f"  • Consider tax implications in taxable accounts")
        print(f"  • Review and adjust based on changing risk tolerance")
        
        self.recommended_portfolio = recommended
        self.recommendation_type = portfolio_type
        
        return True


def main():
    """Main function to execute portfolio optimization"""
    
    print("="*80)
    print("TIME SERIES FORECASTING PORTFOLIO - TASK 4")
    print("Optimize Portfolio Based on Forecast using Modern Portfolio Theory")
    print("="*80)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(symbols=['TSLA', 'BND', 'SPY'], start_date='2020-01-01')
    
    # Step 1: Load historical data
    if not optimizer.load_historical_data():
        return
    
    # Step 2: Calculate expected returns
    if not optimizer.calculate_expected_returns():
        return
    
    # Step 3: Calculate covariance matrix
    if not optimizer.calculate_covariance_matrix():
        return
    
    # Step 4: Generate efficient frontier
    if not optimizer.generate_efficient_frontier():
        return
    
    # Step 5: Find optimal portfolios
    if not optimizer.find_optimal_portfolios():
        return
    
    # Step 6: Create visualizations
    optimizer.create_visualizations()
    
    # Step 7: Provide recommendation
    optimizer.recommend_portfolio()
    
    print("\n" + "="*80)
    print("✅ TASK 4 COMPLETED SUCCESSFULLY!")
    print("Portfolio optimization completed with MPT-based recommendations.")
    print("Check generated visualizations for detailed analysis.")
    print("="*80)


if __name__ == "__main__":
    main()
