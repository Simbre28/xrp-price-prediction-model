# -*- coding: utf-8 -*-
"""xrp-predictor.ipynb
Original file is located at
    https://colab.research.google.com/drive/1f_VJNekpd83p-mUQUtzPRkA3CkoNTrZ9
"""

# XRP Price Prediction Model with Backtesting
# Enhanced with TA-Lib and Professional Technical Indicators

# Install required libraries (run this cell first)
# !pip install yfinance pandas numpy scikit-learn ta matplotlib seaborn xgboost

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Try to import XGBoost (powerful for financial predictions)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# Import TA library for advanced technical indicators
try:
    import ta
    from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, CCIIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
    from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
    from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, VolumeWeightedAveragePrice
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("TA library not available. Install with: pip install ta")

import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('darkgrid')

class XRPPricePredictor:
    def __init__(self, lookback_period='2y'):
        """
        Initialize the XRP price predictor

        Parameters:
        lookback_period: str - How far back to fetch data (e.g., '2y', '5y', 'max')
        """
        self.lookback_period = lookback_period
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None

    def fetch_data(self):
        """Fetch XRP price data from Yahoo Finance"""
        print("Fetching XRP data from Yahoo Finance...")

        # XRP-USD ticker
        xrp = yf.Ticker("XRP-USD")
        df = xrp.history(period=self.lookback_period)

        if df.empty:
            raise ValueError("No data fetched. Check your internet connection or ticker symbol.")

        print(f"Fetched {len(df)} days of data")
        return df

    def create_features(self, df):
        """Create technical indicators and features for prediction using TA library"""
        print("Creating features...")

        data = df.copy()

        # Basic price features
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

        # === MANUAL INDICATORS (Always Available) ===

        # Moving averages
        for period in [7, 14, 21, 50, 200]:
            data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
            data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()

        # Price position relative to moving averages
        data['Price_to_SMA7'] = data['Close'] / data['SMA_7']
        data['Price_to_SMA21'] = data['Close'] / data['SMA_21']
        data['Price_to_SMA50'] = data['Close'] / data['SMA_50']

        # Volatility
        data['Volatility_7'] = data['Returns'].rolling(window=7).std()
        data['Volatility_21'] = data['Returns'].rolling(window=21).std()

        # Price momentum
        for period in [7, 14, 21]:
            data[f'Momentum_{period}'] = data['Close'] - data['Close'].shift(period)
            data[f'ROC_{period}'] = ((data['Close'] - data['Close'].shift(period)) /
                                      data['Close'].shift(period)) * 100

        # Day of week (cyclical encoding)
        data['DayOfWeek_Sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
        data['DayOfWeek_Cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)

        # === TA LIBRARY INDICATORS (If Available) ===

        if TA_AVAILABLE:
            print("Adding TA library indicators...")

            # TREND INDICATORS
            # MACD
            macd = MACD(close=data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_Signal'] = macd.macd_signal()
            data['MACD_Diff'] = macd.macd_diff()

            # ADX (Average Directional Index) - trend strength
            adx = ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'])
            data['ADX'] = adx.adx()
            data['ADX_Pos'] = adx.adx_pos()
            data['ADX_Neg'] = adx.adx_neg()

            # CCI (Commodity Channel Index)
            cci = CCIIndicator(high=data['High'], low=data['Low'], close=data['Close'])
            data['CCI'] = cci.cci()

            # MOMENTUM INDICATORS
            # RSI
            rsi = RSIIndicator(close=data['Close'], window=14)
            data['RSI'] = rsi.rsi()

            # Stochastic Oscillator
            stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
            data['Stoch_K'] = stoch.stoch()
            data['Stoch_D'] = stoch.stoch_signal()

            # Williams %R
            williams = WilliamsRIndicator(high=data['High'], low=data['Low'], close=data['Close'])
            data['Williams_R'] = williams.williams_r()

            # Rate of Change from TA
            roc_ta = ROCIndicator(close=data['Close'], window=12)
            data['ROC_TA'] = roc_ta.roc()

            # VOLATILITY INDICATORS
            # Bollinger Bands
            bb = BollingerBands(close=data['Close'])
            data['BB_High'] = bb.bollinger_hband()
            data['BB_Low'] = bb.bollinger_lband()
            data['BB_Mid'] = bb.bollinger_mavg()
            data['BB_Width'] = bb.bollinger_wband()
            data['BB_Pct'] = bb.bollinger_pband()
            data['BB_High_Indicator'] = bb.bollinger_hband_indicator()
            data['BB_Low_Indicator'] = bb.bollinger_lband_indicator()

            # Average True Range
            atr = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'])
            data['ATR'] = atr.average_true_range()

            # Keltner Channel
            keltner = KeltnerChannel(high=data['High'], low=data['Low'], close=data['Close'])
            data['Keltner_High'] = keltner.keltner_channel_hband()
            data['Keltner_Low'] = keltner.keltner_channel_lband()
            data['Keltner_Mid'] = keltner.keltner_channel_mband()

            # VOLUME INDICATORS
            # On-Balance Volume
            obv = OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume'])
            data['OBV'] = obv.on_balance_volume()

            # Chaikin Money Flow
            cmf = ChaikinMoneyFlowIndicator(high=data['High'], low=data['Low'],
                                           close=data['Close'], volume=data['Volume'])
            data['CMF'] = cmf.chaikin_money_flow()

            # Volume Weighted Average Price
            vwap = VolumeWeightedAveragePrice(high=data['High'], low=data['Low'],
                                             close=data['Close'], volume=data['Volume'])
            data['VWAP'] = vwap.volume_weighted_average_price()

            # Volume indicators
            data['Volume_SMA_7'] = data['Volume'].rolling(window=7).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_7']

            # CUSTOM COMPOSITE INDICATORS
            # Trend Strength Score
            data['Trend_Strength'] = (
                (data['Close'] > data['SMA_21']).astype(int) +
                (data['SMA_7'] > data['SMA_21']).astype(int) +
                (data['SMA_21'] > data['SMA_50']).astype(int) +
                (data['MACD_Diff'] > 0).astype(int)
            )

            # Momentum Score
            data['Momentum_Score'] = (
                (data['RSI'] > 50).astype(int) +
                (data['Stoch_K'] > 50).astype(int) +
                (data['ROC_7'] > 0).astype(int) +
                (data['Williams_R'] > -50).astype(int)
            )

            # Volatility Score
            data['Volatility_Score'] = (
                (data['BB_Width'] > data['BB_Width'].rolling(20).mean()).astype(int) +
                (data['ATR'] > data['ATR'].rolling(14).mean()).astype(int)
            )

        else:
            print("TA library not available - using manual indicators only")
            # Manual RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

            # Manual MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_Diff'] = data['MACD'] - data['MACD_Signal']

            # Manual Bollinger Bands
            data['BB_Mid'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_High'] = data['BB_Mid'] + (bb_std * 2)
            data['BB_Low'] = data['BB_Mid'] - (bb_std * 2)
            data['BB_Width'] = (data['BB_High'] - data['BB_Low']) / data['BB_Mid']

            # Manual ATR
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            data['ATR'] = true_range.rolling(14).mean()

            # Volume
            data['Volume_SMA_7'] = data['Volume'].rolling(window=7).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_7']

        # Target: Price movement over next 7 days (for classification)
        data['Future_Close'] = data['Close'].shift(-7)
        data['Target'] = (data['Future_Close'] > data['Close']).astype(int)
        data['Price_Change_Pct'] = ((data['Future_Close'] - data['Close']) / data['Close']) * 100

        # Print feature count
        feature_count = len([col for col in data.columns if col not in
                           ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends',
                            'Stock Splits', 'Target', 'Future_Close', 'Returns',
                            'Log_Returns', 'Price_Change_Pct']])
        print(f"Created {feature_count} features")

        return data

    def prepare_data(self, df):
        """Prepare data for training"""
        print("Preparing data for training...")

        # Drop rows with NaN values
        df_clean = df.dropna()

        # Select feature columns (exclude target and price columns)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends',
                       'Stock Splits', 'Target', 'Future_Close', 'Returns', 'Log_Returns',
                       'Price_Change_Pct']
        self.feature_columns = [col for col in df_clean.columns if col not in exclude_cols]

        X = df_clean[self.feature_columns]
        y = df_clean['Target']

        print(f"Total samples: {len(X)}")
        print(f"Features: {len(self.feature_columns)}")
        print(f"Positive class (price increase): {y.sum()} ({y.mean()*100:.2f}%)")

        return X, y, df_clean

    def train_model(self, X, y, test_size=0.2):
        """Train multiple models and select the best one"""
        print("\nSplitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train multiple models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }

        # Add XGBoost if available (often best for financial data)
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )

        results = {}
        print("\nTraining models...")

        for name, model in models.items():
            print(f"\n{name}:")
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

        # Select best model based on ROC-AUC
        best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
        self.model = results[best_model_name]['model']

        print(f"\n{'='*50}")
        print(f"Best Model: {best_model_name}")
        print(f"Test Set Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")
        print(f"{'='*50}")

        return results, X_train_scaled, X_test_scaled, y_train, y_test

    def backtest(self, df_full, initial_capital=10000, position_size=0.95,
                 confidence_threshold=0.55, trading_fee=0.001):
        """
        Backtest the trading strategy using walk-forward analysis

        Parameters:
        df_full: DataFrame with all data including features and targets
        initial_capital: Starting capital in USD
        position_size: Fraction of capital to use per trade (0-1)
        confidence_threshold: Minimum prediction probability to enter trade
        trading_fee: Transaction fee as a decimal (0.001 = 0.1%)
        """
        print("\n" + "="*70)
        print("BACKTESTING WITH WALK-FORWARD ANALYSIS")
        print("="*70)

        df_clean = df_full.dropna()

        # Use 70% for initial training, 30% for walk-forward testing
        train_size = int(len(df_clean) * 0.7)

        # Initialize tracking variables
        capital = initial_capital
        positions = []  # Track all trades
        portfolio_values = []
        dates = []

        # Initial model training
        X_initial = df_clean[self.feature_columns].iloc[:train_size]
        y_initial = df_clean['Target'].iloc[:train_size]

        X_initial_scaled = self.scaler.fit_transform(X_initial)
        self.model.fit(X_initial_scaled, y_initial)

        print(f"Initial training period: {df_clean.index[0].date()} to {df_clean.index[train_size-1].date()}")
        print(f"Backtest period: {df_clean.index[train_size].date()} to {df_clean.index[-1].date()}")
        print(f"Initial capital: ${initial_capital:,.2f}")
        print(f"Position size: {position_size*100:.1f}% of capital")
        print(f"Confidence threshold: {confidence_threshold*100:.1f}%")
        print(f"Trading fee: {trading_fee*100:.2f}%")

        # Walk-forward backtesting
        in_position = False
        entry_price = 0
        xrp_amount = 0
        retrain_interval = 30  # Retrain model every 30 days

        for i in range(train_size, len(df_clean) - 7):  # -7 to ensure we have future data
            current_date = df_clean.index[i]
            current_price = df_clean['Close'].iloc[i]
            actual_future_price = df_clean['Future_Close'].iloc[i]
            actual_change_pct = df_clean['Price_Change_Pct'].iloc[i]

            # Retrain model periodically
            if (i - train_size) % retrain_interval == 0 and i > train_size:
                X_retrain = df_clean[self.feature_columns].iloc[:i]
                y_retrain = df_clean['Target'].iloc[:i]
                X_retrain_scaled = self.scaler.fit_transform(X_retrain)
                self.model.fit(X_retrain_scaled, y_retrain)

            # Get current features and make prediction
            X_current = df_clean[self.feature_columns].iloc[i:i+1]
            X_current_scaled = self.scaler.transform(X_current)

            prediction = self.model.predict(X_current_scaled)[0]
            probability = self.model.predict_proba(X_current_scaled)[0]
            confidence = max(probability)

            # Trading logic
            if not in_position and prediction == 1 and probability[1] >= confidence_threshold:
                # Enter long position
                trade_amount = capital * position_size
                fee = trade_amount * trading_fee
                xrp_amount = (trade_amount - fee) / current_price
                entry_price = current_price
                capital -= trade_amount
                in_position = True

                positions.append({
                    'entry_date': current_date,
                    'entry_price': entry_price,
                    'exit_date': df_clean.index[i + 7] if i + 7 < len(df_clean) else df_clean.index[-1],
                    'exit_price': actual_future_price,
                    'confidence': probability[1],
                    'actual_change_pct': actual_change_pct,
                    'type': 'LONG'
                })

            elif in_position:
                # Check if 7 days have passed or we're at the end
                days_in_position = (current_date - positions[-1]['entry_date']).days

                if days_in_position >= 7 or i >= len(df_clean) - 8:
                    # Exit position
                    exit_value = xrp_amount * current_price
                    fee = exit_value * trading_fee
                    capital += (exit_value - fee)

                    # Update position info
                    positions[-1]['actual_exit_price'] = current_price
                    positions[-1]['profit_loss'] = exit_value - (entry_price * xrp_amount)
                    positions[-1]['profit_loss_pct'] = ((current_price - entry_price) / entry_price) * 100

                    in_position = False
                    xrp_amount = 0

            # Track portfolio value
            if in_position:
                portfolio_value = capital + (xrp_amount * current_price)
            else:
                portfolio_value = capital

            portfolio_values.append(portfolio_value)
            dates.append(current_date)

        # Close any remaining position
        if in_position:
            final_price = df_clean['Close'].iloc[-1]
            exit_value = xrp_amount * final_price
            fee = exit_value * trading_fee
            capital += (exit_value - fee)
            positions[-1]['actual_exit_price'] = final_price
            positions[-1]['profit_loss'] = exit_value - (entry_price * xrp_amount)
            positions[-1]['profit_loss_pct'] = ((final_price - entry_price) / entry_price) * 100

        # Calculate metrics
        final_capital = capital
        total_return = ((final_capital - initial_capital) / initial_capital) * 100

        # Convert positions to DataFrame
        positions_df = pd.DataFrame(positions)

        if len(positions_df) > 0:
            winning_trades = len(positions_df[positions_df['profit_loss'] > 0])
            losing_trades = len(positions_df[positions_df['profit_loss'] <= 0])
            win_rate = (winning_trades / len(positions_df)) * 100 if len(positions_df) > 0 else 0

            avg_win = positions_df[positions_df['profit_loss'] > 0]['profit_loss'].mean() if winning_trades > 0 else 0
            avg_loss = positions_df[positions_df['profit_loss'] <= 0]['profit_loss'].mean() if losing_trades > 0 else 0

            # Calculate maximum drawdown
            portfolio_series = pd.Series(portfolio_values, index=dates)
            cumulative_max = portfolio_series.cummax()
            drawdown = (portfolio_series - cumulative_max) / cumulative_max
            max_drawdown = drawdown.min() * 100

            # Buy and hold comparison
            buy_hold_return = ((df_clean['Close'].iloc[-1] - df_clean['Close'].iloc[train_size]) /
                              df_clean['Close'].iloc[train_size]) * 100

        else:
            winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = max_drawdown = 0
            buy_hold_return = 0

        # Print results
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        print(f"\nInitial Capital: ${initial_capital:,.2f}")
        print(f"Final Capital: ${final_capital:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        print(f"Alpha (vs Buy & Hold): {total_return - buy_hold_return:.2f}%")
        print(f"\nTotal Trades: {len(positions_df)}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")

        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss)
            print(f"Profit Factor: {profit_factor:.2f}")

        # Visualizations
        self.plot_backtest_results(portfolio_values, dates, positions_df,
                                   df_clean.iloc[train_size:], initial_capital)

        return {
            'final_capital': final_capital,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'total_trades': len(positions_df),
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'positions': positions_df,
            'portfolio_values': portfolio_values,
            'dates': dates
        }

    def plot_backtest_results(self, portfolio_values, dates, positions_df, price_data, initial_capital):
        """Plot backtest results"""
        fig = plt.figure(figsize=(18, 12))

        # Plot 1: Portfolio Value Over Time
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(dates, portfolio_values, label='Portfolio Value', linewidth=2)
        ax1.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital', alpha=0.5)
        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cumulative Returns
        ax2 = plt.subplot(3, 2, 2)
        returns = [(pv / initial_capital - 1) * 100 for pv in portfolio_values]
        ax2.plot(dates, returns, label='Strategy Returns', linewidth=2, color='green')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Cumulative Returns (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Returns (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: XRP Price with Trade Markers
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(price_data.index, price_data['Close'], label='XRP Price', linewidth=2, alpha=0.7)

        if len(positions_df) > 0:
            # Mark entry and exit points
            winning_trades = positions_df[positions_df['profit_loss'] > 0]
            losing_trades = positions_df[positions_df['profit_loss'] <= 0]

            for _, trade in winning_trades.iterrows():
                ax3.scatter(trade['entry_date'], trade['entry_price'],
                           color='green', marker='^', s=100, zorder=5)
                ax3.scatter(trade['exit_date'], trade['exit_price'],
                           color='green', marker='v', s=100, zorder=5)

            for _, trade in losing_trades.iterrows():
                ax3.scatter(trade['entry_date'], trade['entry_price'],
                           color='red', marker='^', s=100, zorder=5)
                ax3.scatter(trade['exit_date'], trade['exit_price'],
                           color='red', marker='v', s=100, zorder=5)

        ax3.set_title('XRP Price with Trade Entries/Exits', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Price ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Trade Distribution
        ax4 = plt.subplot(3, 2, 4)
        if len(positions_df) > 0:
            profits = positions_df['profit_loss'].values
            colors = ['green' if p > 0 else 'red' for p in profits]
            ax4.bar(range(len(profits)), profits, color=colors, alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.set_title('Profit/Loss per Trade', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Trade Number')
            ax4.set_ylabel('Profit/Loss ($)')
            ax4.grid(True, alpha=0.3)

        # Plot 5: Returns Distribution
        ax5 = plt.subplot(3, 2, 5)
        if len(positions_df) > 0:
            ax5.hist(positions_df['profit_loss_pct'], bins=20, edgecolor='black', alpha=0.7)
            ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax5.set_title('Distribution of Returns (%)', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Return (%)')
            ax5.set_ylabel('Frequency')
            ax5.grid(True, alpha=0.3)

        # Plot 6: Drawdown
        ax6 = plt.subplot(3, 2, 6)
        portfolio_series = pd.Series(portfolio_values, index=dates)
        cumulative_max = portfolio_series.cummax()
        drawdown = (portfolio_series - cumulative_max) / cumulative_max * 100
        ax6.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
        ax6.plot(dates, drawdown, color='darkred', linewidth=2)
        ax6.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Drawdown (%)')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Display trade details table
        if len(positions_df) > 0:
            print("\n" + "="*70)
            print("TRADE DETAILS (First 10 trades)")
            print("="*70)
            display_cols = ['entry_date', 'entry_price', 'exit_price',
                           'profit_loss', 'profit_loss_pct', 'confidence']
            print(positions_df[display_cols].head(10).to_string(index=False))

    def evaluate_model(self, results, y_test, best_model_name, df_clean, X_test):
        """Evaluate and visualize model performance including R² for magnitude predictions"""
        best_result = results[best_model_name]

        print("\n" + "="*70)
        print("MODEL PERFORMANCE METRICS")
        print("="*70)

        # Classification metrics
        print("\n📊 DIRECTION PREDICTION (Up/Down):")
        print(f"Accuracy: {best_result['accuracy']*100:.2f}%")
        print(f"Precision: {best_result['precision']*100:.2f}%")
        print(f"Recall: {best_result['recall']*100:.2f}%")
        print(f"F1-Score: {best_result['f1_score']*100:.2f}%")
        print(f"ROC-AUC Score: {best_result['roc_auc']:.4f}")

        # Calculate R² for magnitude predictions
        # Get actual percentage changes for test set
        test_indices = X_test.index if hasattr(X_test, 'index') else range(len(X_test))
        actual_changes = df_clean.loc[test_indices, 'Price_Change_Pct'].values

        # Get predicted percentage changes based on model confidence
        predicted_changes = []
        X_test_scaled = self.scaler.transform(X_test)

        for i in range(len(X_test)):
            pred = self.model.predict(X_test_scaled[i:i+1])[0]
            prob = self.model.predict_proba(X_test_scaled[i:i+1])[0]
            confidence = max(prob)

            # Get similar historical predictions
            X_all = df_clean[self.feature_columns]
            X_all_scaled = self.scaler.transform(X_all)
            all_preds = self.model.predict(X_all_scaled)
            all_probs = self.model.predict_proba(X_all_scaled)

            similar_mask = (
                (np.max(all_probs, axis=1) >= confidence - 0.10) &
                (np.max(all_probs, axis=1) <= confidence + 0.10) &
                (all_preds == pred)
            )

            similar_changes = df_clean['Price_Change_Pct'][similar_mask]
            pred_change = similar_changes.mean() if len(similar_changes) > 0 else 0
            predicted_changes.append(pred_change)

        predicted_changes = np.array(predicted_changes)

        # Remove NaN values
        valid_mask = ~(np.isnan(actual_changes) | np.isnan(predicted_changes))
        actual_changes_clean = actual_changes[valid_mask]
        predicted_changes_clean = predicted_changes[valid_mask]

        if len(actual_changes_clean) > 0:
            r2 = r2_score(actual_changes_clean, predicted_changes_clean)
            mae = mean_absolute_error(actual_changes_clean, predicted_changes_clean)
            rmse = np.sqrt(mean_squared_error(actual_changes_clean, predicted_changes_clean))

            print(f"\n📈 MAGNITUDE PREDICTION (Percentage Change):")
            print(f"R² Score: {r2:.4f}")
            print(f"Mean Absolute Error: {mae:.2f}%")
            print(f"Root Mean Squared Error: {rmse:.2f}%")

            # Interpretation
            print(f"\n💡 INTERPRETATION:")
            print(f"The model correctly predicts direction {best_result['accuracy']*100:.1f}% of the time.")
            if r2 > 0:
                print(f"The model explains {r2*100:.1f}% of the variance in price changes.")
            else:
                print(f"The R² is {r2:.4f}, indicating the magnitude predictions need improvement.")
            print(f"On average, magnitude predictions are off by {mae:.2f}%.")

        print("\nClassification Report:")
        print(classification_report(y_test, best_result['predictions'],
                                   target_names=['Decrease', 'Increase']))

        # Confusion Matrix
        cm = confusion_matrix(y_test, best_result['predictions'])

        plt.figure(figsize=(18, 10))

        # Plot 1: Confusion Matrix
        plt.subplot(2, 3, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        # Plot 2: Model Comparison
        plt.subplot(2, 3, 2)
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        roc_aucs = [results[name]['roc_auc'] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        plt.bar(x + width/2, roc_aucs, width, label='ROC-AUC', alpha=0.8)
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.ylim([0, 1])

        # Plot 3: Feature Importance
        plt.subplot(2, 3, 3)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-15:]

            plt.barh(range(len(indices)), importances[indices], alpha=0.8)
            plt.yticks(range(len(indices)), [self.feature_columns[i] for i in indices], fontsize=8)
            plt.xlabel('Importance')
            plt.title('Top 15 Feature Importances')

        # Plot 4: Predicted vs Actual Percentage Changes
        plt.subplot(2, 3, 4)
        if len(actual_changes_clean) > 0:
            plt.scatter(actual_changes_clean, predicted_changes_clean, alpha=0.5)
            plt.plot([actual_changes_clean.min(), actual_changes_clean.max()],
                    [actual_changes_clean.min(), actual_changes_clean.max()],
                    'r--', lw=2, label='Perfect Prediction')
            plt.xlabel('Actual Change (%)')
            plt.ylabel('Predicted Change (%)')
            plt.title(f'Predicted vs Actual (R²={r2:.3f})')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Plot 5: Residuals Distribution
        plt.subplot(2, 3, 5)
        if len(actual_changes_clean) > 0:
            residuals = actual_changes_clean - predicted_changes_clean
            plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
            plt.xlabel('Prediction Error (%)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Prediction Errors')
            plt.grid(True, alpha=0.3)

        # Plot 6: Accuracy by Confidence Level
        plt.subplot(2, 3, 6)
        confidence_levels = np.max(best_result['probabilities'].reshape(-1, 1) if best_result['probabilities'].ndim == 1
                                   else best_result['probabilities'], axis=1)

        # Bin by confidence
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
        accuracies_by_conf = []
        counts = []

        for i in range(len(bins)-1):
            mask = (confidence_levels >= bins[i]) & (confidence_levels < bins[i+1])
            if mask.sum() > 0:
                acc = accuracy_score(y_test[mask], best_result['predictions'][mask])
                accuracies_by_conf.append(acc * 100)
                counts.append(mask.sum())
            else:
                accuracies_by_conf.append(0)
                counts.append(0)

        bars = plt.bar(bin_labels, accuracies_by_conf, alpha=0.7)
        plt.axhline(y=best_result['accuracy']*100, color='r', linestyle='--',
                   label=f'Overall: {best_result["accuracy"]*100:.1f}%')
        plt.xlabel('Confidence Level')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy by Prediction Confidence')
        plt.legend()
        plt.ylim([0, 100])

        # Add counts on top of bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            if count > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'n={count}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()

        return r2 if len(actual_changes_clean) > 0 else None

    def predict_next_week(self, recent_data=None):
        """Predict price movement for the next week with expected percentage change"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        if recent_data is None:
            recent_data = self.fetch_data()
            recent_data = self.create_features(recent_data)

        latest = recent_data[self.feature_columns].iloc[-1:].copy()
        latest_scaled = self.scaler.transform(latest)

        prediction = self.model.predict(latest_scaled)[0]
        probability = self.model.predict_proba(latest_scaled)[0]

        # Calculate expected percentage change based on historical data
        # Look at past predictions with similar confidence levels
        df_with_preds = recent_data.dropna()

        # Get predictions for all historical data
        X_all = df_with_preds[self.feature_columns]
        X_all_scaled = self.scaler.transform(X_all)
        all_predictions = self.model.predict(X_all_scaled)
        all_probabilities = self.model.predict_proba(X_all_scaled)

        # Filter for predictions with similar confidence (within 10%)
        current_confidence = max(probability)
        confidence_range = 0.10

        similar_confidence_mask = (
            (np.max(all_probabilities, axis=1) >= current_confidence - confidence_range) &
            (np.max(all_probabilities, axis=1) <= current_confidence + confidence_range) &
            (all_predictions == prediction)
        )

        # Get actual percentage changes for similar predictions
        similar_changes = df_with_preds['Price_Change_Pct'][similar_confidence_mask]

        if len(similar_changes) > 0:
            expected_change = similar_changes.mean()
            change_std = similar_changes.std()
            change_median = similar_changes.median()

            # Calculate range (25th to 75th percentile)
            change_25th = similar_changes.quantile(0.25)
            change_75th = similar_changes.quantile(0.75)
        else:
            # Fallback to overall statistics if no similar predictions found
            if prediction == 1:
                similar_changes = df_with_preds[df_with_preds['Target'] == 1]['Price_Change_Pct']
            else:
                similar_changes = df_with_preds[df_with_preds['Target'] == 0]['Price_Change_Pct']

            expected_change = similar_changes.mean() if len(similar_changes) > 0 else 0
            change_std = similar_changes.std() if len(similar_changes) > 0 else 0
            change_median = similar_changes.median() if len(similar_changes) > 0 else 0
            change_25th = similar_changes.quantile(0.25) if len(similar_changes) > 0 else 0
            change_75th = similar_changes.quantile(0.75) if len(similar_changes) > 0 else 0

        # Get current price
        current_price = recent_data['Close'].iloc[-1]

        return {
            'prediction': 'INCREASE' if prediction == 1 else 'DECREASE',
            'confidence': max(probability) * 100,
            'probability_increase': probability[1] * 100,
            'probability_decrease': probability[0] * 100,
            'expected_change_pct': expected_change,
            'expected_change_median_pct': change_median,
            'expected_change_std': change_std,
            'expected_range_low': change_25th,
            'expected_range_high': change_75th,
            'current_price': current_price,
            'expected_price': current_price * (1 + expected_change / 100),
            'expected_price_range_low': current_price * (1 + change_25th / 100),
            'expected_price_range_high': current_price * (1 + change_75th / 100)
        }

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("XRP Price Prediction Model with Advanced Technical Indicators")
    print("="*70)

    if TA_AVAILABLE:
        print("✓ TA library loaded - Using professional technical indicators")
    else:
        print("⚠ TA library not found - Using manual indicators only")
        print("  Install with: pip install ta")

    if XGBOOST_AVAILABLE:
        print("✓ XGBoost loaded - Enhanced model performance available")
    else:
        print("⚠ XGBoost not found - Using standard models only")
        print("  Install with: pip install xgboost")

    print("="*70)

    # Initialize predictor
    predictor = XRPPricePredictor(lookback_period='3y')

    # Fetch and prepare data
    df = predictor.fetch_data()
    df = predictor.create_features(df)
    X, y, df_clean = predictor.prepare_data(df)

    # Train models
    results, X_train, X_test, y_train, y_test = predictor.train_model(X, y)

    # Store X_test for evaluation
    X_test_df = X.iloc[-len(X_test):]

    # Get best model name
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])

    # Evaluate (now includes R² calculation)
    r2_score_value = predictor.evaluate_model(results, y_test, best_model_name, df_clean, X_test_df)

    # Run backtest
    backtest_results = predictor.backtest(
        df_clean,
        initial_capital=10000,
        position_size=0.95,
        confidence_threshold=0.55,
        trading_fee=0.001
    )

    # Make prediction for next week
    print("\n" + "="*70)
    print("PREDICTION FOR NEXT WEEK")
    print("="*70)

    prediction_result = predictor.predict_next_week(df)

    print(f"\nDirection: {prediction_result['prediction']}")
    print(f"Confidence: {prediction_result['confidence']:.2f}%")
    print(f"Probability of Increase: {prediction_result['probability_increase']:.2f}%")
    print(f"Probability of Decrease: {prediction_result['probability_decrease']:.2f}%")

    print(f"\n{'='*70}")
    print("EXPECTED PRICE CHANGE")
    print(f"{'='*70}")
    print(f"Expected Change: {prediction_result['expected_change_pct']:+.2f}%")
    print(f"Expected Change (Median): {prediction_result['expected_change_median_pct']:+.2f}%")
    print(f"Typical Range: {prediction_result['expected_range_low']:+.2f}% to {prediction_result['expected_range_high']:+.2f}%")

    print(f"\n{'='*70}")
    print("PRICE PROJECTIONS")
    print(f"{'='*70}")
    print(f"Current Price: ${prediction_result['current_price']:.4f}")
    print(f"Expected Price (7 days): ${prediction_result['expected_price']:.4f}")
    print(f"Price Range: ${prediction_result['expected_price_range_low']:.4f} - ${prediction_result['expected_price_range_high']:.4f}")

    # Calculate potential profit/loss on $1000 investment
    investment = 1000
    expected_profit = investment * (prediction_result['expected_change_pct'] / 100)
    range_low_profit = investment * (prediction_result['expected_range_low'] / 100)
    range_high_profit = investment * (prediction_result['expected_range_high'] / 100)

    print(f"\n{'='*70}")
    print(f"POTENTIAL RETURN ON ${investment:,.0f} INVESTMENT")
    print(f"{'='*70}")
    print(f"Expected Return: ${expected_profit:+.2f} ({prediction_result['expected_change_pct']:+.2f}%)")
    print(f"Typical Range: ${range_low_profit:+.2f} to ${range_high_profit:+.2f}")

    print("\n" + "="*70)
    print("DISCLAIMER")
    print("="*70)
    print("This is a prediction model for educational purposes only.")
    print("Past performance does not guarantee future results.")
    print("Cryptocurrency trading carries significant risk.")
    print("Always do thorough research and consult financial advisors.")
    print("="*70)

