"""
Portfolio Strategy Backtesting

Backtests portfolio strategy performance:
- Strategy portfolio with optimized weights
- Benchmark comparison (60/40 SPY/BND)
- Performance metrics and analysis
- Strategy validation and conclusions
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
from scipy.optimize import minimize

# Statistical models for getting optimal weights
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

warnings.filterwarnings('ignore')

# Set plot styles
plt.style.use('ggplot')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

class StrategyBacktester:
    """
    Portfolio strategy backtesting and performance analysis
    """
    
    def __init__(self, symbols=['TSLA', 'BND', 'SPY'], backtest_start='2024-01-15', backtest_end='2025-01-15'):
        self.symbols = symbols
        self.backtest_start = backtest_start
        self.backtest_end = backtest_end
        self.risk_free_rate = 0.045  # 4.5% risk-free rate
        
        # Portfolio weights (will be determined from Task 4 optimization)
        self.strategy_weights = None
        self.benchmark_weights = np.array([0.0, 0.4, 0.6])  # 0% TSLA, 40% BND, 60% SPY
        
    def load_backtesting_data(self):
        """Load historical data for backtesting period"""
        print("="*60)
        print("LOADING BACKTESTING DATA")
        print("="*60)
        
        # Load full historical data for model training
        full_start = '2020-01-01'
        self.full_data = {}
        
        for symbol in self.symbols:
            try:
                print(f"Downloading {symbol} full data...")
                data = yf.download(symbol, start=full_start, end=self.backtest_end)
                self.full_data[symbol] = data['Close'].fillna(method='ffill').fillna(method='bfill')
                
            except Exception as e:
                print(f"✗ Error downloading {symbol}: {e}")
                return False
        
        # Create combined dataframe
        self.full_price_data = pd.DataFrame(self.full_data)
        self.full_price_data = self.full_price_data.dropna()
        
        # Extract backtesting period
        self.backtest_data = self.full_price_data[self.backtest_start:self.backtest_end]
        self.backtest_returns = self.backtest_data.pct_change().dropna()
        
        print(f"✓ Full dataset: {len(self.full_price_data)} observations")
        print(f"✓ Backtest period: {len(self.backtest_data)} observations")
        print(f"Backtest range: {self.backtest_data.index.min()} to {self.backtest_data.index.max()}")
        
        return True
    
    def get_optimal_strategy_weights(self):
        """Get optimal portfolio weights using Task 4 methodology"""
        print("\n" + "="*60)
        print("DETERMINING OPTIMAL STRATEGY WEIGHTS")
        print("="*60)
        
        # Use data up to backtesting start for optimization
        training_data = self.full_price_data[:self.backtest_start]
        training_returns = training_data.pct_change().dropna()
        
        # Get TSLA forecast return
        tsla_expected_return = self._get_tsla_forecast_return(training_data['TSLA'])
        
        # Calculate historical returns for BND and SPY
        bnd_return = training_returns['BND'].mean() * 252
        spy_return = training_returns['SPY'].mean() * 252
        
        expected_returns = np.array([tsla_expected_return, bnd_return, spy_return])
        cov_matrix = training_returns.cov() * 252
        
        print(f"Expected Returns:")
        for i, symbol in enumerate(self.symbols):
            print(f"  {symbol}: {expected_returns[i]:.1%}")
        
        # Find Maximum Sharpe Ratio portfolio
        def negative_sharpe_ratio(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.symbols)))
        initial_guess = np.array([1/len(self.symbols)] * len(self.symbols))
        
        result = minimize(negative_sharpe_ratio, x0=initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            self.strategy_weights = result.x
            portfolio_return = np.sum(self.strategy_weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(self.strategy_weights.T, np.dot(cov_matrix, self.strategy_weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            print(f"\n🎯 OPTIMAL STRATEGY WEIGHTS:")
            for i, symbol in enumerate(self.symbols):
                print(f"  {symbol}: {self.strategy_weights[i]:.1%}")
            print(f"Expected Return: {portfolio_return:.1%}")
            print(f"Expected Volatility: {portfolio_volatility:.1%}")
            print(f"Expected Sharpe Ratio: {sharpe_ratio:.3f}")
            
            return True
        else:
            print("✗ Optimization failed")
            return False
    
    def _get_tsla_forecast_return(self, tsla_prices):
        """Get TSLA expected return from forecasting model"""
        try:
            # Use ARIMA model for simplicity
            auto_model = auto_arima(
                tsla_prices,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False
            )
            
            # Generate 252-day forecast
            forecast_result = auto_model.get_forecast(steps=252)
            forecast_prices = forecast_result.predicted_mean
            
            current_price = tsla_prices.iloc[-1]
            forecast_end_price = forecast_prices.iloc[-1]
            return (forecast_end_price - current_price) / current_price
            
        except:
            # Fallback to historical average
            return tsla_prices.pct_change().mean() * 252
    
    def simulate_portfolios(self):
        """Simulate both strategy and benchmark portfolios"""
        print("\n" + "="*60)
        print("SIMULATING PORTFOLIO PERFORMANCE")
        print("="*60)
        
        # Calculate daily portfolio returns
        self.strategy_daily_returns = (self.backtest_returns * self.strategy_weights).sum(axis=1)
        self.benchmark_daily_returns = (self.backtest_returns * self.benchmark_weights).sum(axis=1)
        
        # Calculate cumulative returns (starting with $10,000)
        initial_value = 10000
        self.strategy_cumulative = (1 + self.strategy_daily_returns).cumprod() * initial_value
        self.benchmark_cumulative = (1 + self.benchmark_daily_returns).cumprod() * initial_value
        
        print(f"✓ Simulated {len(self.strategy_daily_returns)} trading days")
        print(f"Strategy starting value: ${initial_value:,.2f}")
        print(f"Benchmark starting value: ${initial_value:,.2f}")
        
        return True
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        print("\n" + "="*60)
        print("CALCULATING PERFORMANCE METRICS")
        print("="*60)
        
        # Total returns
        strategy_total_return = (self.strategy_cumulative.iloc[-1] / self.strategy_cumulative.iloc[0]) - 1
        benchmark_total_return = (self.benchmark_cumulative.iloc[-1] / self.benchmark_cumulative.iloc[0]) - 1
        
        # Annualized returns
        days = len(self.strategy_daily_returns)
        strategy_annual_return = (1 + strategy_total_return) ** (252/days) - 1
        benchmark_annual_return = (1 + benchmark_total_return) ** (252/days) - 1
        
        # Volatility (annualized)
        strategy_volatility = self.strategy_daily_returns.std() * np.sqrt(252)
        benchmark_volatility = self.benchmark_daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratios
        strategy_sharpe = (strategy_annual_return - self.risk_free_rate) / strategy_volatility
        benchmark_sharpe = (benchmark_annual_return - self.risk_free_rate) / benchmark_volatility
        
        # Maximum drawdown
        strategy_drawdown = (self.strategy_cumulative / self.strategy_cumulative.cummax() - 1).min()
        benchmark_drawdown = (self.benchmark_cumulative / self.benchmark_cumulative.cummax() - 1).min()
        
        # Win rate (days with positive returns)
        strategy_win_rate = (self.strategy_daily_returns > 0).mean()
        benchmark_win_rate = (self.benchmark_daily_returns > 0).mean()
        
        self.performance_metrics = {
            'strategy': {
                'total_return': strategy_total_return,
                'annual_return': strategy_annual_return,
                'volatility': strategy_volatility,
                'sharpe_ratio': strategy_sharpe,
                'max_drawdown': strategy_drawdown,
                'win_rate': strategy_win_rate,
                'final_value': self.strategy_cumulative.iloc[-1]
            },
            'benchmark': {
                'total_return': benchmark_total_return,
                'annual_return': benchmark_annual_return,
                'volatility': benchmark_volatility,
                'sharpe_ratio': benchmark_sharpe,
                'max_drawdown': benchmark_drawdown,
                'win_rate': benchmark_win_rate,
                'final_value': self.benchmark_cumulative.iloc[-1]
            }
        }
        
        print("📊 PERFORMANCE SUMMARY:")
        print(f"\n🎯 STRATEGY PORTFOLIO:")
        print(f"  Total Return: {strategy_total_return:.1%}")
        print(f"  Annualized Return: {strategy_annual_return:.1%}")
        print(f"  Volatility: {strategy_volatility:.1%}")
        print(f"  Sharpe Ratio: {strategy_sharpe:.3f}")
        print(f"  Max Drawdown: {strategy_drawdown:.1%}")
        print(f"  Win Rate: {strategy_win_rate:.1%}")
        print(f"  Final Value: ${self.strategy_cumulative.iloc[-1]:,.2f}")
        
        print(f"\n📈 BENCHMARK (60/40 SPY/BND):")
        print(f"  Total Return: {benchmark_total_return:.1%}")
        print(f"  Annualized Return: {benchmark_annual_return:.1%}")
        print(f"  Volatility: {benchmark_volatility:.1%}")
        print(f"  Sharpe Ratio: {benchmark_sharpe:.3f}")
        print(f"  Max Drawdown: {benchmark_drawdown:.1%}")
        print(f"  Win Rate: {benchmark_win_rate:.1%}")
        print(f"  Final Value: ${self.benchmark_cumulative.iloc[-1]:,.2f}")
        
        return True
    
    def create_performance_visualizations(self):
        """Create comprehensive performance visualizations"""
        print("\n" + "="*60)
        print("CREATING PERFORMANCE VISUALIZATIONS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Plot 1: Cumulative returns
        ax1 = axes[0, 0]
        ax1.plot(self.strategy_cumulative.index, self.strategy_cumulative, 
                label='Strategy Portfolio', linewidth=2, color='blue')
        ax1.plot(self.benchmark_cumulative.index, self.benchmark_cumulative, 
                label='Benchmark (60/40)', linewidth=2, color='red')
        ax1.set_title('Cumulative Returns Comparison', fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot 2: Rolling Sharpe ratio (30-day window)
        ax2 = axes[0, 1]
        rolling_window = 30
        strategy_rolling_sharpe = (self.strategy_daily_returns.rolling(rolling_window).mean() * 252 - self.risk_free_rate) / (self.strategy_daily_returns.rolling(rolling_window).std() * np.sqrt(252))
        benchmark_rolling_sharpe = (self.benchmark_daily_returns.rolling(rolling_window).mean() * 252 - self.risk_free_rate) / (self.benchmark_daily_returns.rolling(rolling_window).std() * np.sqrt(252))
        
        ax2.plot(strategy_rolling_sharpe.index, strategy_rolling_sharpe, 
                label='Strategy', linewidth=2, color='blue')
        ax2.plot(benchmark_rolling_sharpe.index, benchmark_rolling_sharpe, 
                label='Benchmark', linewidth=2, color='red')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title(f'Rolling Sharpe Ratio ({rolling_window}-day)', fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown comparison
        ax3 = axes[1, 0]
        strategy_dd = (self.strategy_cumulative / self.strategy_cumulative.cummax() - 1) * 100
        benchmark_dd = (self.benchmark_cumulative / self.benchmark_cumulative.cummax() - 1) * 100
        
        ax3.fill_between(strategy_dd.index, strategy_dd, 0, alpha=0.3, color='blue', label='Strategy')
        ax3.fill_between(benchmark_dd.index, benchmark_dd, 0, alpha=0.3, color='red', label='Benchmark')
        ax3.set_title('Drawdown Comparison', fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance metrics comparison
        ax4 = axes[1, 1]
        metrics = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
        strategy_values = [
            self.performance_metrics['strategy']['total_return'] * 100,
            self.performance_metrics['strategy']['sharpe_ratio'],
            abs(self.performance_metrics['strategy']['max_drawdown']) * 100
        ]
        benchmark_values = [
            self.performance_metrics['benchmark']['total_return'] * 100,
            self.performance_metrics['benchmark']['sharpe_ratio'],
            abs(self.performance_metrics['benchmark']['max_drawdown']) * 100
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, strategy_values, width, label='Strategy', alpha=0.8, color='blue')
        bars2 = ax4.bar(x + width/2, benchmark_values, width, label='Benchmark', alpha=0.8, color='red')
        
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Value')
        ax4.set_title('Performance Metrics Comparison', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('backtesting_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Performance visualizations created and saved")
        return True
    
    def generate_backtesting_conclusions(self):
        """Generate comprehensive backtesting analysis and conclusions"""
        print("\n" + "="*80)
        print("BACKTESTING ANALYSIS & CONCLUSIONS")
        print("="*80)
        
        strategy_metrics = self.performance_metrics['strategy']
        benchmark_metrics = self.performance_metrics['benchmark']
        
        # Performance comparison
        return_diff = strategy_metrics['total_return'] - benchmark_metrics['total_return']
        sharpe_diff = strategy_metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio']
        vol_diff = strategy_metrics['volatility'] - benchmark_metrics['volatility']
        
        print(f"📊 BACKTESTING PERIOD: {self.backtest_start} to {self.backtest_end}")
        print(f"📈 PORTFOLIO ALLOCATIONS:")
        print(f"  Strategy: TSLA {self.strategy_weights[0]:.1%}, BND {self.strategy_weights[1]:.1%}, SPY {self.strategy_weights[2]:.1%}")
        print(f"  Benchmark: TSLA 0.0%, BND 40.0%, SPY 60.0%")
        
        print(f"\n🎯 PERFORMANCE COMPARISON:")
        print(f"  Return Difference: {return_diff:+.1%} ({'Strategy Wins' if return_diff > 0 else 'Benchmark Wins'})")
        print(f"  Sharpe Difference: {sharpe_diff:+.3f} ({'Strategy Wins' if sharpe_diff > 0 else 'Benchmark Wins'})")
        print(f"  Volatility Difference: {vol_diff:+.1%} ({'Higher' if vol_diff > 0 else 'Lower'} for Strategy)")
        
        # Determine overall winner
        strategy_wins = 0
        if return_diff > 0: strategy_wins += 1
        if sharpe_diff > 0: strategy_wins += 1
        if strategy_metrics['max_drawdown'] > benchmark_metrics['max_drawdown']: strategy_wins -= 0.5  # Penalty for higher drawdown
        
        print(f"\n🏆 OVERALL ASSESSMENT:")
        if strategy_wins >= 1.5:
            conclusion = "STRATEGY OUTPERFORMED"
            performance_assessment = "The model-driven strategy successfully outperformed the benchmark"
        elif strategy_wins >= 0.5:
            conclusion = "MIXED RESULTS"
            performance_assessment = "The strategy showed competitive performance with trade-offs"
        else:
            conclusion = "BENCHMARK OUTPERFORMED"
            performance_assessment = "The simple benchmark outperformed the complex strategy"
        
        print(f"  Result: {conclusion}")
        print(f"  Assessment: {performance_assessment}")
        
        print(f"\n💡 KEY INSIGHTS:")
        
        # Risk-adjusted performance
        if sharpe_diff > 0.1:
            print(f"  ✓ Strategy provided superior risk-adjusted returns (Sharpe: {strategy_metrics['sharpe_ratio']:.3f} vs {benchmark_metrics['sharpe_ratio']:.3f})")
        elif sharpe_diff < -0.1:
            print(f"  ⚠ Strategy had inferior risk-adjusted returns despite complexity")
        else:
            print(f"  ≈ Risk-adjusted performance was similar between strategies")
        
        # Volatility analysis
        if abs(vol_diff) > 0.02:
            if vol_diff > 0:
                print(f"  ⚠ Strategy carried {vol_diff:.1%} higher volatility - risk management needed")
            else:
                print(f"  ✓ Strategy achieved lower volatility while maintaining returns")
        
        # TSLA impact analysis
        tsla_weight = self.strategy_weights[0]
        if tsla_weight > 0.3:
            print(f"  📊 High TSLA allocation ({tsla_weight:.1%}) significantly impacted performance")
        elif tsla_weight > 0.1:
            print(f"  📊 Moderate TSLA allocation ({tsla_weight:.1%}) provided diversification benefits")
        else:
            print(f"  📊 Low TSLA allocation ({tsla_weight:.1%}) suggests model favored stability")
        
        print(f"\n🔍 MODEL VALIDATION:")
        
        # Forecast accuracy assessment
        if return_diff > 0.05:  # Strategy significantly outperformed
            print(f"  ✓ Strong evidence that forecasting models added value")
            print(f"  ✓ Portfolio optimization effectively captured market opportunities")
        elif return_diff > 0:
            print(f"  ≈ Modest evidence supporting the forecasting approach")
            print(f"  ≈ Model provided some value but with limited margin")
        else:
            print(f"  ⚠ Limited evidence of forecasting model effectiveness")
            print(f"  ⚠ Simple benchmark proved difficult to beat consistently")
        
        print(f"\n📋 STRATEGY VIABILITY ASSESSMENT:")
        
        if conclusion == "STRATEGY OUTPERFORMED":
            viability = "HIGH"
            recommendation = "Continue development with refinements"
        elif conclusion == "MIXED RESULTS":
            viability = "MODERATE"
            recommendation = "Refine model parameters and risk management"
        else:
            viability = "LOW"
            recommendation = "Reconsider approach or use simpler strategies"
        
        print(f"  Viability: {viability}")
        print(f"  Recommendation: {recommendation}")
        
        print(f"\n⚠️ LIMITATIONS & CONSIDERATIONS:")
        print(f"  • Backtest period limited to {len(self.strategy_daily_returns)} trading days")
        print(f"  • No transaction costs or slippage included")
        print(f"  • Static rebalancing approach (monthly rebalancing would be more realistic)")
        print(f"  • Market regime changes not fully captured in short timeframe")
        print(f"  • Model assumptions may not hold in different market conditions")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"  1. Extend backtesting period for more robust validation")
        print(f"  2. Implement dynamic rebalancing with updated forecasts")
        print(f"  3. Add transaction cost analysis")
        print(f"  4. Test strategy across different market regimes")
        print(f"  5. Consider ensemble forecasting methods")
        
        return True


def main():
    """Main function to execute strategy backtesting"""
    
    print("="*80)
    print("PORTFOLIO STRATEGY BACKTESTING")
    print("Performance Analysis and Validation")
    print("="*80)
    
    # Initialize backtester
    backtester = StrategyBacktester(
        symbols=['TSLA', 'BND', 'SPY'], 
        backtest_start='2024-01-15', 
        backtest_end='2025-01-15'
    )
    
    # Step 1: Load backtesting data
    if not backtester.load_backtesting_data():
        return
    
    # Step 2: Get optimal strategy weights
    if not backtester.get_optimal_strategy_weights():
        return
    
    # Step 3: Simulate portfolio performance
    if not backtester.simulate_portfolios():
        return
    
    # Step 4: Calculate performance metrics
    if not backtester.calculate_performance_metrics():
        return
    
    # Step 5: Create visualizations
    backtester.create_performance_visualizations()
    
    # Step 6: Generate conclusions
    backtester.generate_backtesting_conclusions()
    
    print("\n" + "="*80)
    print("✅ TASK 5 COMPLETED SUCCESSFULLY!")
    print("Strategy backtesting completed with comprehensive performance analysis.")
    print("Check generated visualizations for detailed performance comparison.")
    print("="*80)


if __name__ == "__main__":
    main()
