"""
Tesla Stock Forecasting Models

Implements multiple forecasting approaches:
- ARIMA with parameter optimization
- SARIMA for seasonal patterns
- LSTM neural network model
- Model evaluation and comparison
- Performance metrics analysis
- Chronological data splitting (train: 2015-2023, test: 2024-2025)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Deep learning models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# Set plot styles
plt.style.use('ggplot')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 12

class TimeSeriesForecaster:
    """
    Tesla stock forecasting with multiple models
    including ARIMA, SARIMA, and LSTM approaches.
    """
    
    def __init__(self, symbol='TSLA', start_date='2015-07-01', end_date='2025-07-31'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.train_data = None
        self.test_data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_data(self):
        """Load and preprocess Tesla stock data"""
        print(f"Loading {self.symbol} stock data from {self.start_date} to {self.end_date}")
        
        try:
            # Download data
            self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            
            # Handle missing values
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
            
            print(f"✓ Data loaded successfully. Shape: {self.data.shape}")
            print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
            
            # Display basic info
            print("\nData columns:", self.data.columns.tolist())
            print("\nFirst 5 rows:")
            print(self.data.head())
            print("\nLast 5 rows:")
            print(self.data.tail())
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def split_data_chronologically(self, split_date='2024-01-01'):
        """Split data chronologically preserving temporal order"""
        print(f"\nSplitting data chronologically at {split_date}")
        
        # Use Close prices for forecasting
        prices = self.data['Close'].copy()
        
        # Split data
        self.train_data = prices[prices.index < split_date]
        self.test_data = prices[prices.index >= split_date]
        
        print(f"Training data: {self.train_data.index.min()} to {self.train_data.index.max()} ({len(self.train_data)} points)")
        print(f"Test data: {self.test_data.index.min()} to {self.test_data.index.max()} ({len(self.test_data)} points)")
        
        # Plot the split
        plt.figure(figsize=(15, 8))
        plt.plot(self.train_data.index, self.train_data, label='Training Data', color='blue', alpha=0.7)
        plt.plot(self.test_data.index, self.test_data, label='Test Data', color='red', alpha=0.7)
        plt.axvline(x=pd.to_datetime(split_date), color='green', linestyle='--', alpha=0.8, label='Split Point')
        plt.title(f'{self.symbol} - Chronological Data Split', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('data_split.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return True
    
    def check_stationarity(self, series, name):
        """Perform Augmented Dickey-Fuller test for stationarity"""
        print(f"\nStationarity test for {name}:")
        result = adfuller(series.dropna())
        print(f'ADF Statistic: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4f}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.4f}')
        
        if result[1] <= 0.05:
            print("✓ Result: Series is stationary")
            return True
        else:
            print("✗ Result: Series is non-stationary")
            return False
    
    def fit_arima_model(self, optimize_params=True):
        """Fit ARIMA model with optional parameter optimization"""
        print("\n" + "="*60)
        print("FITTING ARIMA MODEL")
        print("="*60)
        
        # Check stationarity of training data
        is_stationary = self.check_stationarity(self.train_data, f"{self.symbol} training prices")
        
        # If not stationary, difference the series
        if not is_stationary:
            diff_data = self.train_data.diff().dropna()
            is_diff_stationary = self.check_stationarity(diff_data, f"{self.symbol} differenced prices")
        
        try:
            if optimize_params:
                print("\nOptimizing ARIMA parameters using auto_arima...")
                # Use auto_arima for parameter optimization
                auto_model = auto_arima(
                    self.train_data,
                    start_p=0, start_q=0,
                    max_p=5, max_q=5,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    max_order=None,
                    trace=True
                )
                
                print(f"✓ Optimal ARIMA order: {auto_model.order}")
                self.arima_model = auto_model
                
            else:
                # Use default parameters
                print("Fitting ARIMA(1,1,1) model...")
                model = ARIMA(self.train_data, order=(1, 1, 1))
                self.arima_model = model.fit()
            
            print("\nARIMA Model Summary:")
            print(self.arima_model.summary())
            
            return True
            
        except Exception as e:
            print(f"✗ Error fitting ARIMA model: {e}")
            return False
    
    def fit_sarima_model(self, seasonal_order=(1, 1, 1, 12)):
        """Fit SARIMA model with seasonal components"""
        print("\n" + "="*60)
        print("FITTING SARIMA MODEL")
        print("="*60)
        
        try:
            print(f"Fitting SARIMA model with seasonal order {seasonal_order}...")
            
            # Fit SARIMA model
            sarima_model = SARIMAX(
                self.train_data,
                order=(1, 1, 1),
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.sarima_model = sarima_model.fit(disp=False)
            
            print("✓ SARIMA model fitted successfully")
            print("\nSARIMA Model Summary:")
            print(self.sarima_model.summary())
            
            return True
            
        except Exception as e:
            print(f"✗ Error fitting SARIMA model: {e}")
            return False
    
    def prepare_lstm_data(self, lookback_window=60):
        """Prepare data for LSTM model"""
        print(f"\nPreparing LSTM data with lookback window of {lookback_window} days...")
        
        # Scale the training data
        train_scaled = self.scaler.fit_transform(self.train_data.values.reshape(-1, 1))
        
        # Create sequences for LSTM
        def create_sequences(data, lookback):
            X, y = [], []
            for i in range(lookback, len(data)):
                X.append(data[i-lookback:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)
        
        self.X_train, self.y_train = create_sequences(train_scaled, lookback_window)
        
        # Reshape for LSTM input (samples, time steps, features)
        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        
        print(f"✓ LSTM training data prepared. Shape: {self.X_train.shape}")
        
        return True
    
    def fit_lstm_model(self, epochs=50, batch_size=32):
        """Fit LSTM deep learning model"""
        print("\n" + "="*60)
        print("FITTING LSTM MODEL")
        print("="*60)
        
        try:
            # Build LSTM model
            self.lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            
            # Compile model
            self.lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            print("LSTM Model Architecture:")
            self.lstm_model.summary()
            
            # Early stopping callback
            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            
            # Train model
            print(f"\nTraining LSTM model for {epochs} epochs...")
            history = self.lstm_model.fit(
                self.X_train, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Plot training history
            plt.figure(figsize=(12, 4))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.title('LSTM Model Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('lstm_training_loss.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✓ LSTM model trained successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error fitting LSTM model: {e}")
            return False
    
    def generate_forecasts(self):
        """Generate forecasts using all fitted models"""
        print("\n" + "="*60)
        print("GENERATING FORECASTS")
        print("="*60)
        
        forecast_steps = len(self.test_data)
        self.forecasts = {}
        
        # ARIMA Forecast
        if hasattr(self, 'arima_model'):
            print("Generating ARIMA forecasts...")
            try:
                arima_forecast = self.arima_model.forecast(steps=forecast_steps)
                self.forecasts['ARIMA'] = pd.Series(arima_forecast, index=self.test_data.index)
                print("✓ ARIMA forecasts generated")
            except Exception as e:
                print(f"✗ Error generating ARIMA forecasts: {e}")
        
        # SARIMA Forecast
        if hasattr(self, 'sarima_model'):
            print("Generating SARIMA forecasts...")
            try:
                sarima_forecast = self.sarima_model.forecast(steps=forecast_steps)
                self.forecasts['SARIMA'] = pd.Series(sarima_forecast, index=self.test_data.index)
                print("✓ SARIMA forecasts generated")
            except Exception as e:
                print(f"✗ Error generating SARIMA forecasts: {e}")
        
        # LSTM Forecast
        if hasattr(self, 'lstm_model'):
            print("Generating LSTM forecasts...")
            try:
                # Prepare test data for LSTM
                # Use last 60 days of training data + test data for rolling prediction
                full_data = pd.concat([self.train_data, self.test_data])
                full_scaled = self.scaler.transform(full_data.values.reshape(-1, 1))
                
                lstm_predictions = []
                lookback = 60
                
                for i in range(len(self.train_data), len(full_data)):
                    # Get the last 60 days
                    X_test = full_scaled[i-lookback:i].reshape(1, lookback, 1)
                    pred = self.lstm_model.predict(X_test, verbose=0)
                    lstm_predictions.append(pred[0, 0])
                
                # Inverse transform predictions
                lstm_predictions = np.array(lstm_predictions).reshape(-1, 1)
                lstm_predictions = self.scaler.inverse_transform(lstm_predictions).flatten()
                
                self.forecasts['LSTM'] = pd.Series(lstm_predictions, index=self.test_data.index)
                print("✓ LSTM forecasts generated")
                
            except Exception as e:
                print(f"✗ Error generating LSTM forecasts: {e}")
        
        return len(self.forecasts) > 0
    
    def calculate_metrics(self, actual, predicted):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def evaluate_models(self):
        """Evaluate all models using MAE, RMSE, and MAPE"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        self.evaluation_results = {}
        
        for model_name, forecast in self.forecasts.items():
            metrics = self.calculate_metrics(self.test_data, forecast)
            self.evaluation_results[model_name] = metrics
            
            print(f"\n{model_name} Model Performance:")
            print(f"  MAE:  ${metrics['MAE']:.2f}")
            print(f"  RMSE: ${metrics['RMSE']:.2f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
        
        # Create comparison DataFrame
        results_df = pd.DataFrame(self.evaluation_results).T
        print("\nModel Comparison Summary:")
        print(results_df.round(2))
        
        # Find best performing model for each metric
        print("\nBest Performing Models:")
        print(f"  Lowest MAE:  {results_df['MAE'].idxmin()} ({results_df['MAE'].min():.2f})")
        print(f"  Lowest RMSE: {results_df['RMSE'].idxmin()} ({results_df['RMSE'].min():.2f})")
        print(f"  Lowest MAPE: {results_df['MAPE'].idxmin()} ({results_df['MAPE'].min():.2f}%)")
        
        return results_df
    
    def plot_forecasts(self):
        """Plot actual vs forecasted values for all models"""
        print("\nGenerating forecast comparison plots...")
        
        # Create subplots
        fig, axes = plt.subplots(len(self.forecasts), 1, figsize=(15, 6*len(self.forecasts)))
        if len(self.forecasts) == 1:
            axes = [axes]
        
        colors = ['red', 'green', 'orange', 'purple']
        
        for i, (model_name, forecast) in enumerate(self.forecasts.items()):
            ax = axes[i]
            
            # Plot training data (last 100 days for context)
            train_context = self.train_data.tail(100)
            ax.plot(train_context.index, train_context, label='Training Data', color='blue', alpha=0.7)
            
            # Plot actual test data
            ax.plot(self.test_data.index, self.test_data, label='Actual', color='black', linewidth=2)
            
            # Plot forecast
            ax.plot(forecast.index, forecast, label=f'{model_name} Forecast', 
                   color=colors[i], linestyle='--', linewidth=2)
            
            # Add vertical line at split point
            ax.axvline(x=self.test_data.index[0], color='gray', linestyle=':', alpha=0.8, label='Test Start')
            
            ax.set_title(f'{self.symbol} - {model_name} Model Forecast', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Stock Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_forecasts_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Combined plot
        plt.figure(figsize=(15, 10))
        
        # Plot training data (last 100 days for context)
        train_context = self.train_data.tail(100)
        plt.plot(train_context.index, train_context, label='Training Data', color='blue', alpha=0.7)
        
        # Plot actual test data
        plt.plot(self.test_data.index, self.test_data, label='Actual', color='black', linewidth=3)
        
        # Plot all forecasts
        for i, (model_name, forecast) in enumerate(self.forecasts.items()):
            plt.plot(forecast.index, forecast, label=f'{model_name} Forecast', 
                    color=colors[i], linestyle='--', linewidth=2)
        
        # Add vertical line at split point
        plt.axvline(x=self.test_data.index[0], color='gray', linestyle=':', alpha=0.8, label='Test Start')
        
        plt.title(f'{self.symbol} - All Models Forecast Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('all_models_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        
        print(f"\n📊 DATASET OVERVIEW")
        print(f"Symbol: {self.symbol}")
        print(f"Total data points: {len(self.data)}")
        print(f"Training period: {self.train_data.index.min()} to {self.train_data.index.max()}")
        print(f"Test period: {self.test_data.index.min()} to {self.test_data.index.max()}")
        
        print(f"\n🔍 MODEL PERFORMANCE ANALYSIS")
        results_df = pd.DataFrame(self.evaluation_results).T
        
        # Rank models by each metric
        mae_ranking = results_df['MAE'].rank()
        rmse_ranking = results_df['RMSE'].rank()
        mape_ranking = results_df['MAPE'].rank()
        
        # Calculate overall ranking (lower is better)
        overall_ranking = (mae_ranking + rmse_ranking + mape_ranking) / 3
        results_df['Overall_Rank'] = overall_ranking
        results_df = results_df.sort_values('Overall_Rank')
        
        print("\nRanked Model Performance (1 = Best):")
        print(results_df.round(3))
        
        best_model = results_df.index[0]
        print(f"\n🏆 BEST OVERALL MODEL: {best_model}")
        
        print(f"\n💡 KEY INSIGHTS:")
        
        # Performance insights
        if 'LSTM' in results_df.index and 'ARIMA' in results_df.index:
            lstm_mae = results_df.loc['LSTM', 'MAE']
            arima_mae = results_df.loc['ARIMA', 'MAE']
            
            if lstm_mae < arima_mae:
                improvement = ((arima_mae - lstm_mae) / arima_mae) * 100
                print(f"  • LSTM outperforms ARIMA by {improvement:.1f}% in MAE")
            else:
                improvement = ((lstm_mae - arima_mae) / lstm_mae) * 100
                print(f"  • ARIMA outperforms LSTM by {improvement:.1f}% in MAE")
        
        # Model complexity vs performance trade-off
        print(f"  • Model Complexity: LSTM > SARIMA > ARIMA")
        print(f"  • Interpretability: ARIMA > SARIMA > LSTM")
        
        # Practical recommendations
        print(f"\n📈 PRACTICAL RECOMMENDATIONS:")
        print(f"  • For production deployment: Consider {best_model} model")
        print(f"  • For interpretability: Use ARIMA model")
        print(f"  • For complex patterns: Use LSTM model")
        print(f"  • For seasonal patterns: Use SARIMA model")
        
        return results_df


def main():
    """Main function to execute the complete forecasting pipeline"""
    
    print("="*80)
    print("TESLA STOCK FORECASTING MODELS")
    print("ARIMA, SARIMA, and LSTM Model Comparison")
    print("="*80)
    
    # Initialize forecaster
    forecaster = TimeSeriesForecaster(symbol='TSLA', start_date='2015-07-01', end_date='2025-07-31')
    
    # Step 1: Load data
    if not forecaster.load_data():
        return
    
    # Step 2: Split data chronologically
    forecaster.split_data_chronologically(split_date='2024-01-01')
    
    # Step 3: Fit ARIMA model with optimization
    forecaster.fit_arima_model(optimize_params=True)
    
    # Step 4: Fit SARIMA model
    forecaster.fit_sarima_model(seasonal_order=(1, 1, 1, 12))
    
    # Step 5: Prepare and fit LSTM model
    forecaster.prepare_lstm_data(lookback_window=60)
    forecaster.fit_lstm_model(epochs=50, batch_size=32)
    
    # Step 6: Generate forecasts
    if forecaster.generate_forecasts():
        
        # Step 7: Evaluate models
        results_df = forecaster.evaluate_models()
        
        # Step 8: Create visualizations
        forecaster.plot_forecasts()
        
        # Step 9: Generate comprehensive report
        forecaster.generate_analysis_report()
        
        print("\n" + "="*80)
        print("✅ TASK 2 COMPLETED SUCCESSFULLY!")
        print("All models have been implemented, evaluated, and compared.")
        print("Visualizations and analysis reports have been generated.")
        print("="*80)
    
    else:
        print("✗ Failed to generate forecasts. Please check the models.")


if __name__ == "__main__":
    main()
