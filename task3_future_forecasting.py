"""
Time Series Forecasting Portfolio - Task 3: Forecast Future Market Trends

This script implements comprehensive future forecasting for Tesla stock prices:
- Uses the best-performing model from Task 2
- Generates 6-12 month forecasts with confidence intervals
- Analyzes trends, volatility, and market opportunities/risks
- Provides detailed interpretation and insights
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
# Optional plotly import for interactive charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: Plotly not available. Interactive charts will be skipped.")

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller

# Deep learning models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Statistical analysis
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings('ignore')

# Set plot styles
plt.style.use('ggplot')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

class FutureMarketForecaster:
    """
    Advanced forecasting class for generating future market predictions
    with comprehensive trend analysis and risk assessment.
    """
    
    def __init__(self, symbol='TSLA', start_date='2015-01-01', end_date='2025-01-15'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.train_data = None
        self.test_data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.best_model = None
        self.best_model_type = None
        
    def load_and_prepare_data(self):
        """Load Tesla stock data and prepare for forecasting"""
        print(f"Loading {self.symbol} stock data for future forecasting...")
        
        try:
            # Download data
            self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
            
            # Use all available data for training the final model
            self.prices = self.data['Close'].copy()
            
            print(f"✓ Data loaded successfully. Shape: {self.data.shape}")
            print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
            print(f"Latest price: ${self.prices.iloc[-1]:.2f}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def select_best_model(self):
        """Select and train the best performing model for future forecasting"""
        print("\n" + "="*60)
        print("MODEL SELECTION AND TRAINING")
        print("="*60)
        
        models_performance = {}
        
        # Split data for model evaluation (use recent data for validation)
        split_point = int(len(self.prices) * 0.9)
        train_eval = self.prices.iloc[:split_point]
        test_eval = self.prices.iloc[split_point:]
        
        print(f"Evaluation split: {len(train_eval)} train, {len(test_eval)} test points")
        
        # Test ARIMA model
        try:
            print("\nTesting ARIMA model...")
            auto_model = auto_arima(
                train_eval,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False
            )
            
            # Forecast on test set
            arima_forecast = auto_model.predict(n_periods=len(test_eval))
            arima_mae = mean_absolute_error(test_eval, arima_forecast)
            
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
            print("Testing SARIMA model...")
            sarima_model = SARIMAX(
                train_eval,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            
            sarima_forecast = sarima_model.forecast(steps=len(test_eval))
            sarima_mae = mean_absolute_error(test_eval, sarima_forecast)
            
            models_performance['SARIMA'] = {
                'model': sarima_model,
                'mae': sarima_mae,
                'order': ((1, 1, 1), (1, 1, 1, 12))
            }
            
            print(f"✓ SARIMA - MAE: ${sarima_mae:.2f}")
            
        except Exception as e:
            print(f"✗ SARIMA model failed: {e}")
        
        # Select best model
        if models_performance:
            best_model_name = min(models_performance.keys(), key=lambda k: models_performance[k]['mae'])
            self.best_model = models_performance[best_model_name]['model']
            self.best_model_type = best_model_name
            best_mae = models_performance[best_model_name]['mae']
            
            print(f"\n🏆 Best Model: {best_model_name} (MAE: ${best_mae:.2f})")
            
            # Retrain on full dataset
            print(f"Retraining {best_model_name} on full dataset...")
            
            if best_model_name == 'ARIMA':
                order = models_performance[best_model_name]['order']
                self.best_model = ARIMA(self.prices, order=order).fit()
            elif best_model_name == 'SARIMA':
                self.best_model = SARIMAX(
                    self.prices,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
            
            print(f"✓ {best_model_name} retrained on full dataset")
            return True
        
        else:
            print("✗ No models were successfully trained")
            return False
    
    def generate_future_forecasts(self, forecast_months=9):
        """Generate future forecasts with confidence intervals"""
        print(f"\n" + "="*60)
        print(f"GENERATING {forecast_months}-MONTH FUTURE FORECASTS")
        print("="*60)
        
        # Calculate forecast steps (approximately 21 trading days per month)
        forecast_steps = forecast_months * 21
        
        print(f"Generating {forecast_steps} trading days forecast ({forecast_months} months)")
        
        try:
            if self.best_model_type == 'ARIMA':
                # ARIMA forecast with confidence intervals
                forecast_result = self.best_model.get_forecast(steps=forecast_steps)
                self.forecast_mean = forecast_result.predicted_mean
                self.forecast_ci = forecast_result.conf_int()
                
            elif self.best_model_type == 'SARIMA':
                # SARIMA forecast with confidence intervals
                forecast_result = self.best_model.get_forecast(steps=forecast_steps)
                self.forecast_mean = forecast_result.predicted_mean
                self.forecast_ci = forecast_result.conf_int()
            
            # Create future dates (business days only)
            last_date = self.prices.index[-1]
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)
            
            # Create forecast series
            self.forecast_mean.index = future_dates
            self.forecast_ci.index = future_dates
            
            print(f"✓ Forecast generated successfully")
            print(f"Forecast period: {future_dates[0]} to {future_dates[-1]}")
            print(f"Current price: ${self.prices.iloc[-1]:.2f}")
            print(f"Forecast end price: ${self.forecast_mean.iloc[-1]:.2f}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error generating forecasts: {e}")
            return False
    
    def analyze_forecast_trends(self):
        """Analyze trends and patterns in the forecast"""
        print(f"\n" + "="*60)
        print("TREND ANALYSIS")
        print("="*60)
        
        current_price = self.prices.iloc[-1]
        forecast_end_price = self.forecast_mean.iloc[-1]
        
        # Calculate overall trend
        total_return = (forecast_end_price - current_price) / current_price * 100
        
        # Calculate monthly trends
        monthly_prices = []
        for i in range(0, len(self.forecast_mean), 21):
            if i < len(self.forecast_mean):
                monthly_prices.append(self.forecast_mean.iloc[i])
        
        # Trend direction analysis
        if total_return > 5:
            trend_direction = "Strong Upward"
        elif total_return > 0:
            trend_direction = "Moderate Upward"
        elif total_return > -5:
            trend_direction = "Sideways/Stable"
        elif total_return > -15:
            trend_direction = "Moderate Downward"
        else:
            trend_direction = "Strong Downward"
        
        # Volatility analysis
        forecast_volatility = self.forecast_mean.pct_change().std() * np.sqrt(252) * 100
        historical_volatility = self.prices.pct_change().tail(252).std() * np.sqrt(252) * 100
        
        print(f"📈 TREND ANALYSIS RESULTS:")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Forecast End Price: ${forecast_end_price:.2f}")
        print(f"  Total Expected Return: {total_return:.1f}%")
        print(f"  Trend Direction: {trend_direction}")
        print(f"  Forecast Volatility: {forecast_volatility:.1f}%")
        print(f"  Historical Volatility: {historical_volatility:.1f}%")
        
        # Monthly progression analysis
        print(f"\n📊 MONTHLY PRICE PROGRESSION:")
        for i, price in enumerate(monthly_prices[:6]):  # Show first 6 months
            month_return = (price - current_price) / current_price * 100
            print(f"  Month {i+1}: ${price:.2f} ({month_return:+.1f}%)")
        
        self.trend_analysis = {
            'current_price': current_price,
            'forecast_end_price': forecast_end_price,
            'total_return': total_return,
            'trend_direction': trend_direction,
            'forecast_volatility': forecast_volatility,
            'historical_volatility': historical_volatility,
            'monthly_prices': monthly_prices
        }
        
        return True
    
    def analyze_confidence_intervals(self):
        """Analyze confidence intervals and forecast uncertainty"""
        print(f"\n" + "="*60)
        print("CONFIDENCE INTERVAL ANALYSIS")
        print("="*60)
        
        # Calculate confidence interval widths over time
        ci_lower = self.forecast_ci.iloc[:, 0]
        ci_upper = self.forecast_ci.iloc[:, 1]
        ci_width = ci_upper - ci_lower
        ci_width_pct = (ci_width / self.forecast_mean) * 100
        
        # Analyze how uncertainty grows over time
        initial_width = ci_width.iloc[0]
        final_width = ci_width.iloc[-1]
        width_growth = (final_width - initial_width) / initial_width * 100
        
        # Calculate uncertainty metrics
        avg_ci_width_pct = ci_width_pct.mean()
        max_ci_width_pct = ci_width_pct.max()
        
        print(f"🔍 CONFIDENCE INTERVAL ANALYSIS:")
        print(f"  Initial CI Width: ${initial_width:.2f} ({ci_width_pct.iloc[0]:.1f}% of forecast)")
        print(f"  Final CI Width: ${final_width:.2f} ({ci_width_pct.iloc[-1]:.1f}% of forecast)")
        print(f"  Width Growth: {width_growth:.1f}%")
        print(f"  Average CI Width: {avg_ci_width_pct:.1f}% of forecast")
        print(f"  Maximum CI Width: {max_ci_width_pct:.1f}% of forecast")
        
        # Reliability assessment
        if avg_ci_width_pct < 10:
            reliability = "High"
        elif avg_ci_width_pct < 20:
            reliability = "Moderate"
        elif avg_ci_width_pct < 30:
            reliability = "Low"
        else:
            reliability = "Very Low"
        
        print(f"  Forecast Reliability: {reliability}")
        
        # Time-based uncertainty analysis
        print(f"\n⏰ UNCERTAINTY OVER TIME:")
        quarters = [63, 126, 189]  # 3, 6, 9 months approximately
        for i, q in enumerate(quarters):
            if q < len(ci_width_pct):
                print(f"  Quarter {i+1} (Month {(i+1)*3}): ±{ci_width_pct.iloc[q]:.1f}% uncertainty")
        
        self.ci_analysis = {
            'initial_width': initial_width,
            'final_width': final_width,
            'width_growth': width_growth,
            'avg_ci_width_pct': avg_ci_width_pct,
            'max_ci_width_pct': max_ci_width_pct,
            'reliability': reliability,
            'ci_width_pct': ci_width_pct
        }
        
        return True
    
    def identify_opportunities_and_risks(self):
        """Identify market opportunities and risks based on forecast"""
        print(f"\n" + "="*60)
        print("MARKET OPPORTUNITIES & RISKS ANALYSIS")
        print("="*60)
        
        current_price = self.trend_analysis['current_price']
        forecast_mean = self.forecast_mean
        ci_lower = self.forecast_ci.iloc[:, 0]
        ci_upper = self.forecast_ci.iloc[:, 1]
        
        opportunities = []
        risks = []
        
        # Opportunity Analysis
        if self.trend_analysis['total_return'] > 10:
            opportunities.append(f"Strong upward trend with {self.trend_analysis['total_return']:.1f}% expected return")
        
        if self.trend_analysis['total_return'] > 0:
            opportunities.append("Positive expected returns over forecast period")
        
        # Find potential buying opportunities (local minima)
        forecast_changes = forecast_mean.pct_change()
        if len(forecast_changes[forecast_changes < -0.02]) > 0:  # Days with >2% drops
            opportunities.append("Potential buying opportunities during forecast dips")
        
        # Risk Analysis
        if self.trend_analysis['total_return'] < -10:
            risks.append(f"Significant downside risk with {self.trend_analysis['total_return']:.1f}% expected decline")
        
        if self.ci_analysis['avg_ci_width_pct'] > 20:
            risks.append(f"High forecast uncertainty (±{self.ci_analysis['avg_ci_width_pct']:.1f}% average)")
        
        if self.trend_analysis['forecast_volatility'] > self.trend_analysis['historical_volatility'] * 1.2:
            risks.append("Increased volatility expected compared to historical levels")
        
        # Worst-case scenario analysis
        worst_case_return = (ci_lower.iloc[-1] - current_price) / current_price * 100
        best_case_return = (ci_upper.iloc[-1] - current_price) / current_price * 100
        
        if worst_case_return < -20:
            risks.append(f"Worst-case scenario shows {worst_case_return:.1f}% potential loss")
        
        # Time-specific risks
        if self.ci_analysis['width_growth'] > 100:
            risks.append("Forecast uncertainty doubles over time horizon")
        
        print(f"🚀 MARKET OPPORTUNITIES:")
        for i, opp in enumerate(opportunities, 1):
            print(f"  {i}. {opp}")
        
        if not opportunities:
            print("  No significant opportunities identified in current forecast")
        
        print(f"\n⚠️  MARKET RISKS:")
        for i, risk in enumerate(risks, 1):
            print(f"  {i}. {risk}")
        
        if not risks:
            print("  No significant risks identified in current forecast")
        
        print(f"\n📊 SCENARIO ANALYSIS:")
        print(f"  Best Case: {best_case_return:+.1f}% return (${ci_upper.iloc[-1]:.2f})")
        print(f"  Expected: {self.trend_analysis['total_return']:+.1f}% return (${forecast_mean.iloc[-1]:.2f})")
        print(f"  Worst Case: {worst_case_return:+.1f}% return (${ci_lower.iloc[-1]:.2f})")
        
        self.opportunities_risks = {
            'opportunities': opportunities,
            'risks': risks,
            'best_case_return': best_case_return,
            'worst_case_return': worst_case_return,
            'best_case_price': ci_upper.iloc[-1],
            'worst_case_price': ci_lower.iloc[-1]
        }
        
        return True
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive forecast visualizations"""
        print(f"\nGenerating comprehensive forecast visualizations...")
        
        # Main forecast plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Plot 1: Historical data + Forecast with confidence intervals
        ax1 = axes[0, 0]
        
        # Historical data (last 2 years for context)
        historical_context = self.prices.tail(504)  # ~2 years
        ax1.plot(historical_context.index, historical_context, label='Historical Prices', 
                color='blue', linewidth=2)
        
        # Forecast
        ax1.plot(self.forecast_mean.index, self.forecast_mean, label='Forecast', 
                color='red', linewidth=2, linestyle='--')
        
        # Confidence intervals
        ax1.fill_between(self.forecast_mean.index, 
                        self.forecast_ci.iloc[:, 0], 
                        self.forecast_ci.iloc[:, 1],
                        color='red', alpha=0.2, label='95% Confidence Interval')
        
        ax1.axvline(x=self.prices.index[-1], color='green', linestyle=':', alpha=0.8, label='Forecast Start')
        ax1.set_title('Tesla Stock Price Forecast with Confidence Intervals', fontweight='bold')
        ax1.set_ylabel('Stock Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence interval width over time
        ax2 = axes[0, 1]
        ax2.plot(self.forecast_mean.index, self.ci_analysis['ci_width_pct'], 
                color='orange', linewidth=2)
        ax2.set_title('Forecast Uncertainty Over Time', fontweight='bold')
        ax2.set_ylabel('Confidence Interval Width (%)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Expected returns progression
        ax3 = axes[1, 0]
        returns_progression = ((self.forecast_mean - self.prices.iloc[-1]) / self.prices.iloc[-1] * 100)
        ax3.plot(self.forecast_mean.index, returns_progression, color='green', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_title('Expected Returns Progression', fontweight='bold')
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Risk-Return scenarios
        ax4 = axes[1, 1]
        scenarios = ['Worst Case', 'Expected', 'Best Case']
        returns = [self.opportunities_risks['worst_case_return'], 
                  self.trend_analysis['total_return'],
                  self.opportunities_risks['best_case_return']]
        colors = ['red', 'blue', 'green']
        
        bars = ax4.bar(scenarios, returns, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_title('Forecast Scenarios', fontweight='bold')
        ax4.set_ylabel('Total Return (%)')
        
        # Add value labels on bars
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{ret:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig('comprehensive_forecast_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Interactive Plotly chart (if available)
        if PLOTLY_AVAILABLE:
            fig_plotly = go.Figure()
            
            # Historical data
            historical_context = self.prices.tail(252)  # 1 year context
            fig_plotly.add_trace(go.Scatter(
                x=historical_context.index,
                y=historical_context,
                mode='lines',
                name='Historical Prices',
                line=dict(color='blue', width=2)
            ))
            
            # Forecast
            fig_plotly.add_trace(go.Scatter(
                x=self.forecast_mean.index,
                y=self.forecast_mean,
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Confidence intervals
            fig_plotly.add_trace(go.Scatter(
                x=self.forecast_mean.index,
                y=self.forecast_ci.iloc[:, 1],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig_plotly.add_trace(go.Scatter(
                x=self.forecast_mean.index,
                y=self.forecast_ci.iloc[:, 0],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 0, 0, 0.2)',
                fill='tonexty',
                name='95% Confidence Interval',
                hoverinfo='skip'
            ))
            
            fig_plotly.update_layout(
                title='Tesla Stock Price Forecast - Interactive View',
                xaxis_title='Date',
                yaxis_title='Stock Price ($)',
                hovermode='x unified',
                width=1200,
                height=600
            )
            
            fig_plotly.write_html('interactive_forecast.html')
            print("✓ Interactive forecast chart saved as 'interactive_forecast.html'")
        else:
            print("✓ Skipped interactive chart (plotly not available)")
        
        return True
    
    def generate_executive_summary(self):
        """Generate executive summary of forecast analysis"""
        print(f"\n" + "="*80)
        print("EXECUTIVE SUMMARY - TESLA FUTURE MARKET FORECAST")
        print("="*80)
        
        current_date = datetime.now().strftime("%B %d, %Y")
        forecast_end = self.forecast_mean.index[-1].strftime("%B %d, %Y")
        
        print(f"\n📅 FORECAST OVERVIEW:")
        print(f"  Analysis Date: {current_date}")
        print(f"  Forecast Period: {len(self.forecast_mean)} trading days (~9 months)")
        print(f"  Forecast End Date: {forecast_end}")
        print(f"  Model Used: {self.best_model_type}")
        print(f"  Current Tesla Price: ${self.trend_analysis['current_price']:.2f}")
        
        print(f"\n🎯 KEY FORECAST RESULTS:")
        print(f"  Expected End Price: ${self.trend_analysis['forecast_end_price']:.2f}")
        print(f"  Total Expected Return: {self.trend_analysis['total_return']:+.1f}%")
        print(f"  Trend Direction: {self.trend_analysis['trend_direction']}")
        print(f"  Forecast Reliability: {self.ci_analysis['reliability']}")
        
        print(f"\n📊 RISK ASSESSMENT:")
        print(f"  Best Case Scenario: {self.opportunities_risks['best_case_return']:+.1f}% (${self.opportunities_risks['best_case_price']:.2f})")
        print(f"  Worst Case Scenario: {self.opportunities_risks['worst_case_return']:+.1f}% (${self.opportunities_risks['worst_case_price']:.2f})")
        print(f"  Average Uncertainty: ±{self.ci_analysis['avg_ci_width_pct']:.1f}%")
        print(f"  Volatility Outlook: {self.trend_analysis['forecast_volatility']:.1f}% (vs {self.trend_analysis['historical_volatility']:.1f}% historical)")
        
        print(f"\n💡 STRATEGIC INSIGHTS:")
        
        # Investment recommendation
        if self.trend_analysis['total_return'] > 15:
            recommendation = "STRONG BUY"
        elif self.trend_analysis['total_return'] > 5:
            recommendation = "BUY"
        elif self.trend_analysis['total_return'] > -5:
            recommendation = "HOLD"
        elif self.trend_analysis['total_return'] > -15:
            recommendation = "WEAK SELL"
        else:
            recommendation = "SELL"
        
        print(f"  Investment Outlook: {recommendation}")
        
        # Key insights
        if self.trend_analysis['total_return'] > 0:
            print(f"  • Positive momentum expected over forecast horizon")
        else:
            print(f"  • Caution advised due to negative expected returns")
        
        if self.ci_analysis['reliability'] in ['High', 'Moderate']:
            print(f"  • Forecast shows {self.ci_analysis['reliability'].lower()} confidence levels")
        else:
            print(f"  • High uncertainty requires careful risk management")
        
        if len(self.opportunities_risks['opportunities']) > len(self.opportunities_risks['risks']):
            print(f"  • More opportunities than risks identified")
        else:
            print(f"  • Risk management should be prioritized")
        
        print(f"\n⚠️  IMPORTANT DISCLAIMERS:")
        print(f"  • This forecast is based on historical patterns and statistical models")
        print(f"  • Actual results may vary significantly due to market volatility")
        print(f"  • External factors (news, regulations, market sentiment) not included")
        print(f"  • Confidence intervals widen over time, reducing long-term accuracy")
        print(f"  • This analysis is for educational purposes only, not investment advice")
        
        return True


def main():
    """Main function to execute the future forecasting pipeline"""
    
    print("="*80)
    print("TIME SERIES FORECASTING PORTFOLIO - TASK 3")
    print("Forecast Future Market Trends for Tesla Stock")
    print("="*80)
    
    # Initialize forecaster
    forecaster = FutureMarketForecaster(symbol='TSLA', start_date='2015-01-01')
    
    # Step 1: Load and prepare data
    if not forecaster.load_and_prepare_data():
        return
    
    # Step 2: Select and train best model
    if not forecaster.select_best_model():
        return
    
    # Step 3: Generate future forecasts (9 months)
    if not forecaster.generate_future_forecasts(forecast_months=9):
        return
    
    # Step 4: Analyze forecast trends
    forecaster.analyze_forecast_trends()
    
    # Step 5: Analyze confidence intervals
    forecaster.analyze_confidence_intervals()
    
    # Step 6: Identify opportunities and risks
    forecaster.identify_opportunities_and_risks()
    
    # Step 7: Create comprehensive visualizations
    forecaster.create_comprehensive_visualizations()
    
    # Step 8: Generate executive summary
    forecaster.generate_executive_summary()
    
    print("\n" + "="*80)
    print("✅ TASK 3 COMPLETED SUCCESSFULLY!")
    print("Future market forecast analysis completed with comprehensive insights.")
    print("Check generated visualizations and interactive charts for detailed analysis.")
    print("="*80)


if __name__ == "__main__":
    main()
