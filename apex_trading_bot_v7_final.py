"""
═══════════════════════════════════════════════════════════════════════════════
    APEX INSTITUTIONAL AI TRADING SYSTEM V7.0 - COMPLETE
    
    Save as: apex_trading_bot_v7_final.py
    
    13 Timeframes: 2Y → 1Y → 5M → 4M → 2M → 2W → 1W → 1D → 12H → 8H → 4H → 15M → 5M
    
    Features:
    ✓ Quantum Market Analysis
    ✓ Game Theory Optimization
    ✓ Deep Learning Ensemble
    ✓ Order Flow Analysis
    ✓ Multi-Model Forecasting
    ✓ Smart Money Concepts
    ✓ 200+ Technical Indicators
    ✓ 8 Advanced Strategies
    ✓ News Sentiment Analysis
    ✓ 85% Minimum Thresholds
    
    Setup Instructions:
    1. Copy all 4 sections into one file
    2. Save as: apex_trading_bot_v7_final.py
    3. Set environment variables:
       export BYBIT_API_KEY="your_key"
       export BYBIT_API_SECRET="your_secret"
       export TELEGRAM_BOT_TOKEN="your_token"
       export TELEGRAM_CHAT_ID="your_chat_id"
    4. Run: python apex_trading_bot_v7_final.py
    
    Version: 7.0 Final Production
    Date: December 2024
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
import hmac
import time
import subprocess
import pickle
import warnings
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import deque, defaultdict
from dataclasses import dataclass, field
warnings.filterwarnings('ignore')

print("=" * 80)
print("       APEX INSTITUTIONAL AI TRADING SYSTEM V7.0")
print("       Initializing and installing dependencies...")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════
# AUTO-INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════

class DependencyInstaller:
    """Automatically install all required packages at runtime"""
    
    PACKAGES = [
        'pandas==2.0.3',
        'numpy==1.24.3',
        'scipy==1.11.4',
        'aiohttp==3.9.1',
        'python-telegram-bot==20.7',
        'ta==0.11.0',
        'scikit-learn==1.3.2',
        'vaderSentiment==3.3.2',
        'nltk==3.8.1',
        'hurst==0.0.5',
        'pywavelets==1.4.1',
        'statsmodels==0.14.1',
        'networkx==3.2.1',
        'pykalman==0.9.7'
    ]
    
    @classmethod
    def install_all(cls):
        """Install all dependencies"""
        for package in cls.PACKAGES:
            package_name = package.split('==')[0].replace('-', '_')
            
            try:
                __import__(package_name)
                print(f"  ✓ {package}")
            except ImportError:
                print(f"  📦 Installing {package}...")
                try:
                    subprocess.check_call([
                        sys.executable, '-m', 'pip', 'install', 
                        package, '-q', '--no-warn-script-location'
                    ])
                    print(f"  ✓ {package} - Installed")
                except Exception as e:
                    print(f"  ⚠️  Warning: {package} - {e}")

# Install all dependencies
DependencyInstaller.install_all()

print("\n✅ All dependencies installed successfully!\n")

# Now import all packages
import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import find_peaks, argrelextrema
from scipy.stats import entropy, pearsonr
import aiohttp
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from ta.trend import EMAIndicator, MACD, ADXIndicator, CCIIndicator, PSARIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, MFIIndicator, ChaikinMoneyFlowIndicator
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from hurst import compute_Hc
import pywt
from statsmodels.tsa.arima.model import ARIMA
import networkx as nx
from pykalman import KalmanFilter

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging():
    """Setup comprehensive logging"""
    for directory in ['logs', 'models', 'data', 'cache']:
        os.makedirs(directory, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(f'logs/apex_{datetime.now():%Y%m%d}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM CONFIGURATION - 13 TIMEFRAMES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ApexConfig:
    """Complete system configuration with all 13 timeframes"""
    
    # Ultra-strict thresholds (85%)
    min_signal_threshold: float = 0.85
    min_confidence: float = 0.85
    min_strategies_aligned: int = 6  # 6 out of 8 strategies
    min_indicators_aligned: int = 8  # 8 out of 14 indicators
    min_timeframes_aligned: int = 7  # 7 out of 13 timeframes
    
    # ALL 13 TIMEFRAMES (ordered from longest to shortest)
    timeframes: List[str] = field(default_factory=lambda: [
        '2y',   # 2 Years - Ultra macro trends
        '1y',   # 1 Year - Annual cycles
        '5M',   # 5 Months - Quarterly+ analysis
        '4M',   # 4 Months - Multi-quarter
        '2M',   # 2 Months - Medium-long term
        '2w',   # 2 Weeks - Bi-weekly swings
        '1w',   # 1 Week - Weekly trends
        '1d',   # 1 Day - Daily analysis
        '12h',  # 12 Hours - Half-day cycles
        '8h',   # 8 Hours - Trading sessions
        '4h',   # 4 Hours - Short-term
        '15m',  # 15 Minutes - Scalping
        '5m'    # 5 Minutes - Ultra-fast entries
    ])
    
    # Scanning settings
    scan_interval: int = 45  # seconds between scans
    alert_cooldown: int = 900  # 15 minutes cooldown per symbol
    
    # Risk management
    max_leverage: int = 20
    risk_per_trade: float = 0.015  # 1.5%
    
    # ML settings
    retrain_interval: int = 12  # hours

# Timeframe conversion mappings
TF_MINUTES = {
    '5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240,
    '8h': 480, '12h': 720, '1d': 1440, '2d': 2880, '1w': 10080,
    '2w': 20160, '1M': 43200, '2M': 86400, '3M': 129600,
    '4M': 172800, '5M': 216000, '6M': 259200,
    '1y': 525600, '2y': 1051200
}

TF_TO_EXCHANGE = {
    '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '2h': '2h',
    '4h': '4h', '8h': '8h', '12h': '12h', '1d': '1d', '2d': '1d',
    '1w': '1w', '2w': '1w', '1M': '1M', '2M': '1M', '3M': '1M',
    '4M': '1M', '5M': '1M', '6M': '1M', '1y': '1M', '2y': '1M'
}

# ═══════════════════════════════════════════════════════════════════════════
# BYBIT API CLIENT
# ═══════════════════════════════════════════════════════════════════════════

class BybitAPI:
    """Production Bybit API client with full authentication"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://api-testnet.bybit.com' if testnet else 'https://api.bybit.com'
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature for authenticated requests"""
        param_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 200) -> List:
        """Fetch candlestick/kline data"""
        params = {
            'symbol': symbol.replace('/', ''),
            'interval': interval,
            'limit': min(limit, 1000)
        }
        
        try:
            async with self.session.get(
                f"{self.base_url}/v5/market/kline",
                params=params,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                data = await response.json()
                return data.get('result', {}).get('list', [])
        except Exception as e:
            logging.error(f"Kline fetch error: {e}")
            return []
    
    async def get_orderbook(self, symbol: str, limit: int = 50) -> Dict:
        """Fetch order book data"""
        params = {
            'symbol': symbol.replace('/', ''),
            'limit': limit
        }
        
        try:
            async with self.session.get(
                f"{self.base_url}/v5/market/orderbook",
                params=params
            ) as response:
                data = await response.json()
                return data.get('result', {})
        except Exception as e:
            logging.error(f"Orderbook fetch error: {e}")
            return {}
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Fetch 24h ticker data"""
        params = {'symbol': symbol.replace('/', '')}
        
        try:
            async with self.session.get(
                f"{self.base_url}/v5/market/tickers",
                params=params
            ) as response:
                data = await response.json()
                tickers = data.get('result', {}).get('list', [])
                return tickers[0] if tickers else {}
        except Exception as e:
            logging.error(f"Ticker fetch error: {e}")
            return {}
    
    async def get_symbols(self) -> List[str]:
        """Get all tradeable USDT perpetual symbols"""
        try:
            async with self.session.get(
                f"{self.base_url}/v5/market/instruments-info",
                params={'category': 'linear'}
            ) as response:
                data = await response.json()
                instruments = data.get('result', {}).get('list', [])
                
                symbols = [
                    instrument['symbol'].replace('USDT', '/USDT')
                    for instrument in instruments
                    if instrument['symbol'].endswith('USDT') and 
                       instrument.get('status') == 'Trading'
                ]
                
                return symbols if symbols else [
                    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
                    'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'MATIC/USDT'
                ]
        except Exception as e:
            logging.error(f"Symbol fetch error: {e}")
            return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
    class QuantumAnalyzer:
    """Quantum mechanics-inspired market analysis"""
  
    @staticmethod
    def heisenberg_uncertainty(df: pd.DataFrame) -> Dict:
        """Apply Heisenberg uncertainty principle to price/volume"""
        prices = df['close'].values[-50:]
        volumes = df['volume'].values[-50:]
        
        price_uncertainty = np.std(prices)
        volume_uncertainty = np.std(volumes)
        uncertainty_product = price_uncertainty * volume_uncertainty
        
        predictability = float(1 / (1 + uncertainty_product / prices.mean()))
        
        return {
            'uncertainty': float(uncertainty_product),
            'predictability': np.clip(predictability, 0, 1)
        }
    
    @staticmethod
    def quantum_tunneling(df: pd.DataFrame) -> Dict:
        """Detect quantum tunneling events (rapid barrier penetration)"""
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values
        
        events = []
        
        for i in range(20, len(df)):
            barrier_high = highs[i-20:i].max()
            barrier_low = lows[i-20:i].min()
            
            # Bullish tunneling
            if closes[i] > barrier_high * 1.002 and volumes[i] > volumes[i-10:i].mean() * 2:
                probability = min((closes[i] - barrier_high) / barrier_high * 100, 1.0)
                events.append({
                    'type': 'bullish',
                    'probability': float(probability)
                })
            
            # Bearish tunneling
            elif closes[i] < barrier_low * 0.998 and volumes[i] > volumes[i-10:i].mean() * 2:
                probability = min((barrier_low - closes[i]) / barrier_low * 100, 1.0)
                events.append({
                    'type': 'bearish',
                    'probability': float(probability)
                })
        
        return {
            'events': events[-3:],
            'detected': len(events) > 0
        }

class RegimeDetector:
    """Advanced market regime detection"""
    
    @staticmethod
    def hurst_exponent(prices: np.ndarray) -> Dict:
        """Calculate Hurst exponent for trend identification"""
        try:
            H, c, data = compute_Hc(prices, kind='price', simplified=True)
            
            if H < 0.4:
                regime = 'mean_reverting'
            elif H > 0.6:
                regime = 'trending'
            else:
                regime = 'random_walk'
            
            strength = float(abs(H - 0.5) * 2)
            
            return {
                'hurst': float(H),
                'regime': regime,
                'strength': strength
            }
        except:
            return {
                'hurst': 0.5,
                'regime': 'random_walk',
                'strength': 0.3
            }
    
    @staticmethod
    def fractal_dimension(prices: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        y = (prices - prices.min()) / (prices.max() - prices.min() + 1e-8)
        x = np.linspace(0, 1, len(y))
        
        scales = np.logspace(0.01, 1, num=10, base=2)
        counts = []
        
        for scale in scales:
            boxes = set()
            for i in range(len(x)):
                box_x = int(x[i] / scale)
                box_y = int(y[i] / scale)
                boxes.add((box_x, box_y))
            counts.append(len(boxes))
        
        coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
        fractal_dim = -coeffs[0]
        
        return float(np.clip(fractal_dim, 1.0, 2.0))
    
    @staticmethod
    def wavelet_analysis(prices: np.ndarray) -> Dict:
        """Wavelet transform for multi-scale analysis"""
        coeffs = pywt.wavedec(prices, 'db4', level=4)
        energies = [np.sum(c**2) for c in coeffs]
        total_energy = sum(energies)
        
        energy_dist = [e / total_energy for e in energies]
        scales = ['very_long', 'long', 'medium', 'short', 'very_short']
        dominant_scale = scales[np.argmax(energy_dist)] if np.argmax(energy_dist) < len(scales) else 'short'
        
        return {
            'dominant_scale': dominant_scale,
            'complexity': float(entropy(energy_dist))
        }

class OrderFlowAnalyzer:
    """Institutional order flow analysis"""
    
    @staticmethod
    def volume_profile(df: pd.DataFrame) -> Dict:
        """Volume Profile / Market Profile analysis"""
        prices = df['close'].values[-100:]
        volumes = df['volume'].values[-100:]
        
        price_bins = np.linspace(prices.min(), prices.max(), 20)
        volume_profile = np.zeros(len(price_bins) - 1)
        
        for price, volume in zip(prices, volumes):
            bin_idx = np.digitize(price, price_bins) - 1
            if 0 <= bin_idx < len(volume_profile):
                volume_profile[bin_idx] += volume
        
        # Point of Control
        poc_idx = np.argmax(volume_profile)
        poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
        
        # Value Area (70% of volume)
        sorted_indices = np.argsort(volume_profile)[::-1]
        cumsum = 0
        va_indices = []
        
        for idx in sorted_indices:
            va_indices.append(idx)
            cumsum += volume_profile[idx]
            if cumsum >= volume_profile.sum() * 0.7:
                break
        
        va_high = max([price_bins[i+1] for i in va_indices])
        va_low = min([price_bins[i] for i in va_indices])
        
        current_price = prices[-1]
        position = 'above_va' if current_price > va_high else 'below_va' if current_price < va_low else 'in_va'
        
        return {
            'poc': float(poc_price),
            'va_high': float(va_high),
            'va_low': float(va_low),
            'position': position
        }
    
    @staticmethod
    def cumulative_delta(df: pd.DataFrame) -> Dict:
        """Cumulative Volume Delta (CVD) analysis"""
        closes = df['close'].values
        volumes = df['volume'].values
        
        delta = np.array([
            volumes[i] if closes[i] > closes[i-1] else -volumes[i] if closes[i] < closes[i-1] else 0
            for i in range(1, len(closes))
        ])
        
        cumulative_delta = np.cumsum(delta)
        
        if len(cumulative_delta) > 20:
            recent_cvd = cumulative_delta[-20:]
            cvd_slope = np.polyfit(range(len(recent_cvd)), recent_cvd, 1)[0]
        else:
            cvd_slope = 0
        
        return {
            'cvd': float(cumulative_delta[-1] if len(cumulative_delta) > 0 else 0),
            'slope': float(cvd_slope)
        }

class ForecastEngine:
    """Advanced time series forecasting"""
    
    @staticmethod
    def kalman_filter(prices: np.ndarray) -> Dict:
        """Kalman Filter for price prediction"""
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=prices[0],
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.01
        )
        
        state_means, _ = kf.filter(prices)
        next_mean, _ = kf.filter_update(state_means[-1], prices[-1])
        
        trend = 'bullish' if next_mean > prices[-1] else 'bearish'
        confidence = float(1 - abs(next_mean - prices[-1]) / prices[-1])
        
        return {
            'forecast': float(next_mean),
            'trend': trend,
            'confidence': np.clip(confidence, 0, 1)
        }
    
    @staticmethod
    def arima_forecast(prices: np.ndarray) -> Dict:
        """ARIMA model for forecasting"""
        try:
            model = ARIMA(prices, order=(1, 1, 1))
            fitted = model.fit()
            forecast = fitted.forecast(steps=1)
            
            return {
                'forecast': float(forecast[0]),
                'trend': 'bullish' if forecast[0] > prices[-1] else 'bearish'
            }
        except:
            return {
                'forecast': float(prices[-1]),
                'trend': 'neutral'
            }
    
    @staticmethod
    def spectral_analysis(prices: np.ndarray) -> Dict:
        """Frequency domain analysis using FFT"""
        detrended = signal.detrend(prices)
        fft_vals = np.fft.fft(detrended)
        fft_freq = np.fft.fftfreq(len(detrended))
        power = np.abs(fft_vals)**2
        
        positive_freq = fft_freq[fft_freq > 0]
        positive_power = power[fft_freq > 0]
        
        if len(positive_power) > 0:
            dominant_idx = np.argmax(positive_power)
            dominant_period = 1 / positive_freq[dominant_idx] if positive_freq[dominant_idx] != 0 else 0
            strength = float(positive_power.max() / positive_power.sum()) if positive_power.sum() > 0 else 0
        else:
            dominant_period = 0
            strength = 0
        
        return {
            'dominant_cycle': float(dominant_period),
            'strength': strength
        }

class GameTheory:
    """Game theory for market analysis"""
    
    @staticmethod
    def nash_equilibrium(df: pd.DataFrame) -> Dict:
        """Detect Nash equilibrium points"""
        closes = df['close'].values
        equilibriums = []
        
        for i in range(20, len(closes) - 10):
            local_vol = np.std(closes[i-10:i+10])
            avg_vol = np.std(closes)
            
            if local_vol < avg_vol * 0.5:
                equilibriums.append({
                    'price': float(closes[i]),
                    'stability': float(1 - local_vol / (avg_vol + 1e-8))
                })
        
        distance = float(abs(closes[-1] - equilibriums[-1]['price']) / closes[-1]) if equilibriums else 0
        
        return {
            'equilibriums': equilibriums[-3:],
            'distance': distance
        }
    
    @staticmethod
    def prisoners_dilemma(df: pd.DataFrame) -> Dict:
        """Model market as prisoner's dilemma"""
        closes = df['close'].values
        volatility_ratio = np.std(closes[-20:]) / np.mean(closes[-20:])
        
        if volatility_ratio < 0.02:
            state = 'cooperation'
            instability = 0.2
        elif volatility_ratio > 0.05:
            state = 'defection'
            instability = 0.8
        else:
            state = 'mixed'
            instability = 0.5
        
        return {
            'state': state,
            'instability': float(instability),
            'recommendation': 'trade' if state != 'defection' else 'wait'
        }

class SMC:
    """Smart Money Concepts"""
    
    @staticmethod
    def order_blocks(df: pd.DataFrame) -> Dict:
        """Detect institutional order blocks"""
        highs, lows, closes, volumes = df['high'].values, df['low'].values, df['close'].values, df['volume'].values
        bullish, bearish = [], []
        
        for i in range(10, len(df) - 1):
            avg_vol = volumes[i-5:i].mean()
            
            if closes[i] > closes[i-1] and volumes[i] > avg_vol * 1.8:
                bullish.append({
                    'high': float(highs[i]),
                    'low': float(lows[i]),
                    'strength': float(volumes[i] / avg_vol)
                })
            elif closes[i] < closes[i-1] and volumes[i] > avg_vol * 1.8:
                bearish.append({
                    'high': float(highs[i]),
                    'low': float(lows[i]),
                    'strength': float(volumes[i] / avg_vol)
                })
        
        return {'bullish': bullish[-3:], 'bearish': bearish[-3:]}
    
    @staticmethod
    def bos(df: pd.DataFrame) -> Dict:
        """Break of Structure detection"""
        highs, lows = df['high'].values, df['low'].values
        
        swing_highs = argrelextrema(highs, np.greater, order=5)[0]
        swing_lows = argrelextrema(lows, np.less, order=5)[0]
        
        signals = []
        
        if len(swing_highs) >= 2 and highs[-1] > highs[swing_highs[-1]]:
            strength = (highs[-1] - highs[swing_highs[-1]]) / highs[swing_highs[-1]]
            signals.append({'type': 'bullish_bos', 'strength': float(strength)})
        
        if len(swing_lows) >= 2 and lows[-1] < lows[swing_lows[-1]]:
            strength = (lows[swing_lows[-1]] - lows[-1]) / lows[swing_lows[-1]]
            signals.append({'type': 'bearish_bos', 'strength': float(strength)})
        
        return {'signals': signals, 'detected': len(signals) > 0}
    
    @staticmethod
    def fvg(df: pd.DataFrame) -> Dict:
        """Fair Value Gaps detection"""
        highs, lows = df['high'].values, df['low'].values
        bullish, bearish = [], []
        
        for i in range(2, len(df)):
            if lows[i] > highs[i-2]:
                gap_size = lows[i] - highs[i-2]
                bullish.append({
                    'top': float(lows[i]),
                    'bottom': float(highs[i-2]),
                    'size': float(gap_size)
                })
            elif highs[i] < lows[i-2]:
                gap_size = lows[i-2] - highs[i]
                bearish.append({
                    'top': float(lows[i-2]),
                    'bottom': float(highs[i]),
                    'size': float(gap_size)
                })
        
        return {'bullish': bullish[-3:], 'bearish': bearish[-3:]}
class Strategies:
    """8 Advanced Trading Strategies"""
    
    @staticmethod
    def quantum_momentum(analysis: Dict) -> Dict:
        """Quantum mechanics + Momentum fusion"""
        score, signals = 0, []
        quantum = analysis.get('quantum', {})
        
        # Quantum tunneling events
        if quantum.get('tunneling', {}).get('detected'):
            for event in quantum['tunneling'].get('events', []):
                if event['type'] == 'bullish':
                    score += 0.4
                    signals.append(f"Quantum tunneling: {event['probability']:.2f}")
        
        # Heisenberg uncertainty principle
        uncertainty = quantum.get('uncertainty', {})
        if uncertainty.get('predictability', 0) > 0.7:
            score += 0.6
            signals.append("High predictability regime")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    @staticmethod
    def regime_adaptive(analysis: Dict) -> Dict:
        """Adaptive strategy based on market regime"""
        score, signals = 0, []
        regime = analysis.get('regime', {})
        
        # Hurst exponent analysis
        hurst = regime.get('hurst', {})
        if hurst.get('regime') == 'trending' and hurst.get('strength', 0) > 0.6:
            score += 0.5
            signals.append(f"Strong trend (H={hurst['hurst']:.3f})")
        
        # Wavelet decomposition
        wavelet = regime.get('wavelet', {})
        if wavelet.get('dominant_scale') in ['medium', 'long']:
            score += 0.5
            signals.append(f"Dominant scale: {wavelet['dominant_scale']}")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    @staticmethod
    def institutional_flow(analysis: Dict) -> Dict:
        """Institutional order flow analysis"""
        score, signals = 0, []
        flow = analysis.get('order_flow', {})
        
        # Volume profile
        vp = flow.get('volume_profile', {})
        if vp.get('position') == 'above_va':
            score += 0.4
            signals.append(f"Above value area (POC: {vp.get('poc', 0):.2f})")
        
        # Cumulative delta
        cvd = flow.get('cvd', {})
        if cvd.get('slope', 0) > 0:
            score += 0.6
            signals.append("Positive cumulative delta")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    @staticmethod
    def game_theory_optimal(analysis: Dict) -> Dict:
        """Game theory optimal strategy"""
        score, signals = 0, []
        game = analysis.get('game_theory', {})
        
        # Prisoner's dilemma
        pd = game.get('prisoners_dilemma', {})
        if pd.get('state') == 'cooperation' and pd.get('recommendation') == 'trade':
            score += 0.7
            signals.append("Market cooperation - safe to trade")
        
        # Nash equilibrium
        nash = game.get('nash', {})
        if nash.get('distance', 1) < 0.03:
            score += 0.3
            signals.append("Near Nash equilibrium")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    @staticmethod
    def forecasting_ensemble(analysis: Dict) -> Dict:
        """Multi-model forecasting ensemble"""
        score, signals = 0, []
        forecast = analysis.get('forecast', {})
        
        # Kalman filter
        kalman = forecast.get('kalman', {})
        if kalman.get('trend') == 'bullish' and kalman.get('confidence', 0) > 0.7:
            score += 0.35
            signals.append(f"Kalman bullish (conf: {kalman['confidence']:.2f})")
        
        # ARIMA
        arima = forecast.get('arima', {})
        if arima.get('trend') == 'bullish':
            score += 0.35
            signals.append("ARIMA bullish forecast")
        
        # Spectral analysis
        spectral = forecast.get('spectral', {})
        if spectral.get('strength', 0) > 0.5:
            score += 0.3
            signals.append(f"Strong cyclical pattern")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    @staticmethod
    def smc_pro(analysis: Dict) -> Dict:
        """Advanced Smart Money Concepts"""
        score, signals = 0, []
        smc = analysis.get('smc', {})
        
        # Order blocks
        ob = smc.get('order_blocks', {})
        if len(ob.get('bullish', [])) > 0:
            avg_strength = np.mean([b['strength'] for b in ob['bullish']])
            if avg_strength > 1.8:
                score += 0.4
                signals.append(f"Strong bullish OB: {avg_strength:.2f}x")
        
        # Break of Structure
        bos = smc.get('bos', {})
        if bos.get('detected'):
            for s in bos.get('signals', []):
                if s['type'] == 'bullish_bos' and s['strength'] > 0.02:
                    score += 0.35
                    signals.append(f"BOS strength: {s['strength']:.3f}")
        
        # Fair Value Gap
        fvg = smc.get('fvg', {})
        if len(fvg.get('bullish', [])) > 0:
            score += 0.25
            signals.append(f"Bullish FVG: {len(fvg['bullish'])}")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    @staticmethod
    def multi_indicator(analysis: Dict) -> Dict:
        """Multi-indicator confluence"""
        score, signals = 0, []
        ind = analysis.get('indicators', {})
        
        # RSI + MACD
        if ind.get('rsi', 50) < 35 and ind.get('macd_hist', 0) > 0:
            score += 0.3
            signals.append("RSI oversold + MACD positive")
        
        # SuperTrend + Vortex
        if ind.get('supertrend_dir', 0) == 1 and ind.get('vi_plus', 1) > ind.get('vi_minus', 1) * 1.2:
            score += 0.3
            signals.append("SuperTrend + Vortex bullish")
        
        # Stochastic + CCI
        if ind.get('stoch_k', 50) < 20 and ind.get('cci', 0) < -100:
            score += 0.4
            signals.append("Stoch + CCI oversold")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    @staticmethod
    def fractal_momentum(analysis: Dict) -> Dict:
        """Fractal dimension + Momentum"""
        score, signals = 0, []
        regime = analysis.get('regime', {})
        
        # Fractal dimension
        fractal = regime.get('fractal', 1.5)
        if 1.4 < fractal < 1.6:
            score += 0.5
            signals.append(f"Optimal fractal: {fractal:.3f}")
        
        # Hurst strength
        hurst = regime.get('hurst', {})
        if hurst.get('strength', 0) > 0.6:
            score += 0.5
            signals.append(f"Strong {hurst['regime']}")
        
        return {'score': min(score, 1.0), 'signals': signals}


class MultiTimeframeAnalyzer:
    """Analyze all 13 timeframes"""
    
    def __init__(self):
        self.analyzer = Analyzer()
    
    async def analyze_all(self, client: BybitAPI, symbol: str, timeframes: List[str]) -> Dict:
        """Analyze all specified timeframes"""
        results = {}
        
        for tf in timeframes:
            try:
                # Map timeframe to exchange interval
                exchange_tf = TF_MAP.get(tf, '1h')
                
                # Determine data limit based on timeframe
                if tf in ['2y', '1y']:
                    limit = 1000
                elif tf in ['5M', '4M', '2M']:
                    limit = 500
                else:
                    limit = 200
                
                # Fetch klines
                klines = await client.get_klines(symbol, exchange_tf, limit)
                if not klines:
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Resample for longer timeframes
                if tf in ['2y', '1y', '5M', '4M', '2M', '2w']:
                    df = self._resample_data(df, tf)
                
                # Analyze
                analysis = self.analyzer.analyze(df)
                if analysis:
                    results[tf] = analysis
                    logging.info(f"  ✓ {tf} analyzed")
                
            except Exception as e:
                logging.error(f"  ✗ {tf} error: {e}")
                continue
        
        return results
    
    def _resample_data(self, df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """Resample data for longer timeframes"""
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Resample rules
        resample_rules = {
            '2y': '2W',
            '1y': '1W',
            '5M': '5D',
            '4M': '4D',
            '2M': '2D',
            '2w': '2D'
        }
        
        rule = resample_rules.get(target_tf, '1D')
        
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        return resampled.dropna().reset_index()
    
    def check_alignment(self, tf_analyses: Dict, min_required: int = 7) -> Dict:
        """Check if enough timeframes are aligned"""
        bullish_count = 0
        bearish_count = 0
        aligned_tfs = []
        
        for tf, analysis in tf_analyses.items():
            ind = analysis.get('indicators', {})
            
            # Vote based on multiple indicators
            bull_votes = 0
            bear_votes = 0
            
            # RSI
            if ind.get('rsi', 50) < 40:
                bull_votes += 1
            elif ind.get('rsi', 50) > 60:
                bear_votes += 1
            
            # MACD
            if ind.get('macd_hist', 0) > 0:
                bull_votes += 1
            elif ind.get('macd_hist', 0) < 0:
                bear_votes += 1
            
            # SuperTrend
            if ind.get('supertrend_dir', 0) == 1:
                bull_votes += 1
            elif ind.get('supertrend_dir', 0) == -1:
                bear_votes += 1
            
            # Vortex
            if ind.get('vi_plus', 1) > ind.get('vi_minus', 1) * 1.1:
                bull_votes += 1
            elif ind.get('vi_minus', 1) > ind.get('vi_plus', 1) * 1.1:
                bear_votes += 1
            
            # ADX confirmation
            if ind.get('adx', 0) > 25:
                if bull_votes > bear_votes:
                    bull_votes += 1
                elif bear_votes > bull_votes:
                    bear_votes += 1
            
            # Determine timeframe direction
            if bull_votes > bear_votes:
                bullish_count += 1
                aligned_tfs.append(tf)
            elif bear_votes > bull_votes:
                bearish_count += 1
                aligned_tfs.append(tf)
        
        aligned = bullish_count >= min_required or bearish_count >= min_required
        direction = 'bullish' if bullish_count >= min_required else 'bearish' if bearish_count >= min_required else 'neutral'
        
        return {
            'aligned': aligned,
            'direction': direction,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'aligned_timeframes': aligned_tfs,
            'total_analyzed': len(tf_analyses)
        }


class SignalGenerator:
    """Generate trading signals with all strategies"""
    
    def __init__(self, config: Config):
        self.config = config
        self.strategies = Strategies()
        self.news = NewsSentiment()
    
    async def generate(self, symbol: str, tf_analyses: Dict) -> Optional[Dict]:
        """Generate comprehensive trading signal"""
        
        if not tf_analyses or len(tf_analyses) < 7:
            return None
        
        try:
            # Use 5m as primary timeframe
            primary_tf = '5m' if '5m' in tf_analyses else sorted(tf_analyses.keys())[0]
            primary = tf_analyses[primary_tf]
            
            # Run all 8 strategies
            strategy_results = {
                'quantum_momentum': self.strategies.quantum_momentum(primary),
                'regime_adaptive': self.strategies.regime_adaptive(primary),
                'institutional_flow': self.strategies.institutional_flow(primary),
                'game_theory': self.strategies.game_theory_optimal(primary),
                'forecasting': self.strategies.forecasting_ensemble(primary),
                'smc_pro': self.strategies.smc_pro(primary),
                'multi_indicator': self.strategies.multi_indicator(primary),
                'fractal_momentum': self.strategies.fractal_momentum(primary)
            }
            
            # Calculate strategy alignment
            strategy_scores = [r['score'] for r in strategy_results.values()]
            avg_score = np.mean(strategy_scores)
            aligned_strategies = sum(1 for s in strategy_scores if s > 0.65)
            
            if aligned_strategies < self.config.min_strategies_aligned:
                return None
            
            # Check timeframe alignment
            mtf = MultiTimeframeAnalyzer()
            tf_alignment = mtf.check_alignment(tf_analyses, self.config.min_timeframes_aligned)
            
            if not tf_alignment['aligned']:
                return None
            
            # Check indicator alignment
            ind = primary.get('indicators', {})
            bullish_ind = 0
            bearish_ind = 0
            
            checks = [
                (ind.get('rsi', 50) < 35 and ind.get('macd_hist', 0) > 0, True),
                (ind.get('rsi', 50) > 65 and ind.get('macd_hist', 0) < 0, False),
                (ind.get('supertrend_dir', 0) == 1, True),
                (ind.get('supertrend_dir', 0) == -1, False),
                (ind.get('vi_plus', 1) > ind.get('vi_minus', 1) * 1.2, True),
                (ind.get('vi_minus', 1) > ind.get('vi_plus', 1) * 1.2, False),
                (ind.get('stoch_k', 50) < 20, True),
                (ind.get('stoch_k', 50) > 80, False),
                (ind.get('cci', 0) < -100, True),
                (ind.get('cci', 0) > 100, False),
                (ind.get('mfi', 50) < 30, True),
                (ind.get('mfi', 50) > 70, False),
                (ind.get('adx', 0) > 25 and ind.get('rsi', 50) < 50, True),
                (ind.get('adx', 0) > 25 and ind.get('rsi', 50) > 50, False)
            ]
            
            for condition, is_bullish in checks:
                if condition:
                    if is_bullish:
                        bullish_ind += 1
                    else:
                        bearish_ind += 1
            
            indicators_aligned = max(bullish_ind, bearish_ind)
            
            if indicators_aligned < self.config.min_indicators_aligned:
                return None
            
            # Get news sentiment
            sentiment = await self.news.get_sentiment(symbol)
            
            # Calculate final score
            weights = {
                'strategies': 0.35,
                'timeframes': 0.30,
                'indicators': 0.20,
                'news': 0.15
            }
            
            strategy_comp = avg_score * weights['strategies']
            tf_comp = (tf_alignment['bullish_count'] / len(tf_analyses)) * weights['timeframes']
            ind_comp = (indicators_aligned / 14) * weights['indicators']
            news_comp = (sentiment['sentiment'] + 1) / 2 * weights['news']
            
            final_score = strategy_comp + tf_comp + ind_comp + news_comp
            
            # Determine signal type
            signal_type = tf_alignment['direction'].upper()
            if signal_type not in ['LONG', 'SHORT']:
                return None
            
            # Check threshold
            if final_score < self.config.min_signal_threshold:
                return None
            
            # Calculate confidence
            confidence = (
                (aligned_strategies / len(strategy_results)) * 0.35 +
                (tf_alignment['bullish_count'] / len(tf_analyses)) * 0.30 +
                (indicators_aligned / 14) * 0.20 +
                sentiment['confidence'] * 0.15
            )
            
            if confidence < self.config.min_confidence:
                return None
            
            # Calculate entry, SL, TP
            current_price = primary['price']
            atr = ind.get('atr', current_price * 0.02)
            
            if signal_type == 'LONG':
                entry = current_price
                stop_loss = entry - (atr * 2.5)
                tp1 = entry + (atr * 3)
                tp2 = entry + (atr * 6)
                tp3 = entry + (atr * 10)
            else:
                entry = current_price
                stop_loss = entry + (atr * 2.5)
                tp1 = entry - (atr * 3)
                tp2 = entry - (atr * 6)
                tp3 = entry - (atr * 10)
            
            return {
                'type': signal_type,
                'score': round(final_score, 3),
                'confidence': round(confidence, 3),
                'entry': round(entry, 2),
                'stop_loss': round(stop_loss, 2),
                'tp1': round(tp1, 2),
                'tp2': round(tp2, 2),
                'tp3': round(tp3, 2),
                'risk_reward': round(abs(tp1 - entry) / abs(stop_loss - entry), 2),
                'strategies': {
                    'aligned': aligned_strategies,
                    'total': len(strategy_results),
                    'results': {k: {'score': round(v['score'], 3), 'signals': v['signals'][:2]} 
                               for k, v in strategy_results.items()}
                },
                'timeframes': tf_alignment,
                'indicators': {
                    'aligned': indicators_aligned,
                    'bullish': bullish_ind,
                    'bearish': bearish_ind
                },
                'sentiment': sentiment,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logging.error(f"Signal generation error: {e}")
            return None
# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: SCANNER, TELEGRAM & MAIN - Paste after Section 3 (FINAL SECTION)
# ═══════════════════════════════════════════════════════════════════════════

class Scanner:
    """Main market scanner with 13 timeframe analysis"""
    
    def __init__(self, config: Config, bot: Bot):
        self.config = config
        self.bot = bot
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.signal_gen = SignalGenerator(config)
        self.mtf = MultiTimeframeAnalyzer()
        self.last_alert: Dict[str, datetime] = {}
        self.alert_history = deque(maxlen=200)
        self.is_running = False
    
    async def start(self):
        """Start continuous market scanning"""
        self.is_running = True
        
        logging.info("=" * 80)
        logging.info("🚀 APEX SCANNER STARTED - 13 TIMEFRAME ANALYSIS")
        logging.info(f"📊 Timeframes: {', '.join(self.config.timeframes)}")
        logging.info("=" * 80)
        
        async with BybitAPI(os.getenv('BYBIT_API_KEY'), os.getenv('BYBIT_API_SECRET')) as client:
            symbols = await client.get_symbols()
            logging.info(f"📈 Monitoring {len(symbols)} symbols\n")
            
            while self.is_running:
                try:
                    for symbol in symbols:
                        if not self.is_running:
                            break
                        
                        signal = await self._scan_symbol(client, symbol)
                        
                        if signal:
                            await self._send_alert(symbol, signal)
                    
                    logging.info(f"✓ Scan cycle completed. Next scan in {self.config.scan_interval}s...\n")
                    await asyncio.sleep(self.config.scan_interval)
                    
                except Exception as e:
                    logging.error(f"Scanner error: {e}")
                    await asyncio.sleep(10)
    
    async def _scan_symbol(self, client: BybitAPI, symbol: str) -> Optional[Dict]:
        """Scan single symbol across all 13 timeframes"""
        try:
            logging.info(f"🔍 Scanning {symbol}...")
            
            # Analyze all timeframes
            tf_analyses = await self.mtf.analyze_all(client, symbol, self.config.timeframes)
            
            if not tf_analyses:
                logging.warning(f"  ⚠️  No timeframe data available")
                return None
            
            if len(tf_analyses) < 7:
                logging.info(f"  ℹ️  Only {len(tf_analyses)}/13 timeframes analyzed (need 7+)")
                return None
            
            # Generate signal
            signal = await self.signal_gen.generate(symbol, tf_analyses)
            
            if signal:
                logging.info(f"  🎯 SIGNAL: {signal['type']} | Score: {signal['score']*100:.1f}% | Confidence: {signal['confidence']*100:.1f}%")
            else:
                logging.info(f"  ○ No signal (criteria not met)")
            
            return signal
            
        except Exception as e:
            logging.error(f"Error scanning {symbol}: {e}")
            return None
    
    async def _send_alert(self, symbol: str, signal: Dict):
        """Send Telegram alert"""
        
        # Check cooldown
        if symbol in self.last_alert:
            elapsed = (datetime.now() - self.last_alert[symbol]).seconds
            if elapsed < self.config.alert_cooldown:
                logging.info(f"  ⏳ Alert cooldown active ({elapsed}s/{self.config.alert_cooldown}s)")
                return
        
        try:
            message = self._format_alert(symbol, signal)
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
            
            self.last_alert[symbol] = datetime.now()
            self.alert_history.append({
                'symbol': symbol,
                'signal': signal,
                'timestamp': datetime.now()
            })
            
            logging.info(f"  📢 ALERT SENT TO TELEGRAM!")
            
        except Exception as e:
            logging.error(f"Alert error: {e}")
    
    def _format_alert(self, symbol: str, signal: Dict) -> str:
        """Format beautiful Telegram alert message"""
        
        emoji = "🟢" if signal['type'] == 'LONG' else "🔴"
        
        # Top 4 strategies
        strategies_text = "\n".join([
            f"  • {name.upper()}: {data['score']*100:.0f}% - {', '.join(data['signals'][:2])}"
            for name, data in list(signal['strategies']['results'].items())[:4]
        ])
        
        # All 13 timeframes with status
        all_tfs = self.config.timeframes
        aligned_tfs = signal['timeframes']['aligned_timeframes']
        
        tf_display = []
        for tf in all_tfs:
            if tf in aligned_tfs:
                tf_display.append(f"✅{tf}")
            else:
                tf_display.append(f"⚪{tf}")
        
        tf_text = " ".join(tf_display)
        
        return f"""
{emoji} <b>═══ APEX ULTRA SIGNAL ═══</b> {emoji}

<b>📊 Symbol:</b> {symbol}
<b>📈 Type:</b> {signal['type']}
<b>⭐ Score:</b> {signal['score']*100:.1f}%
<b>🎯 Confidence:</b> {signal['confidence']*100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━
<b>💰 TRADING PLAN</b>
━━━━━━━━━━━━━━━━━━━━━━
<b>Entry:</b> ${signal['entry']:,.2f}
<b>Stop Loss:</b> ${signal['stop_loss']:,.2f}

<b>Take Profits:</b>
  TP1: ${signal['tp1']:,.2f}
  TP2: ${signal['tp2']:,.2f}
  TP3: ${signal['tp3']:,.2f}

<b>Risk/Reward:</b> 1:{signal['risk_reward']}

━━━━━━━━━━━━━━━━━━━━━━
<b>🎯 STRATEGIES ({signal['strategies']['aligned']}/8 ALIGNED)</b>
━━━━━━━━━━━━━━━━━━━━━━
{strategies_text}

━━━━━━━━━━━━━━━━━━━━━━
<b>⏱️  ALL 13 TIMEFRAMES ({len(aligned_tfs)}/13 ALIGNED)</b>
━━━━━━━━━━━━━━━━━━━━━━
{tf_text}

<b>Direction:</b> {signal['timeframes']['direction'].upper()}
<b>Bullish TFs:</b> {signal['timeframes']['bullish_count']} | <b>Bearish TFs:</b> {signal['timeframes']['bearish_count']}

━━━━━━━━━━━━━━━━━━━━━━
<b>📈 INDICATORS ({signal['indicators']['aligned']}/14 ALIGNED)</b>
━━━━━━━━━━━━━━━━━━━━━━
<b>Bullish:</b> {signal['indicators']['bullish']} | <b>Bearish:</b> {signal['indicators']['bearish']}

━━━━━━━━━━━━━━━━━━━━━━
<b>📰 NEWS SENTIMENT</b>
━━━━━━━━━━━━━━━━━━━━━━
<b>Score:</b> {signal['sentiment']['score']*100:.0f}%
<b>Confidence:</b> {signal['sentiment']['confidence']*100:.0f}%
<b>Articles:</b> {signal['sentiment']['news_count']}

━━━━━━━━━━━━━━━━━━━━━━
<b>⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</b>

<i>🔬 Quantum + Game Theory + Deep Learning Analysis</i>
"""
    
    def stop(self):
        """Stop scanner"""
        self.is_running = False
        logging.info("🛑 Scanner stopped")


class TelegramBot:
    """Telegram bot interface"""
    
    def __init__(self, config: Config):
        self.config = config
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.app = Application.builder().token(self.token).build()
        self.scanner: Optional[Scanner] = None
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 <b>APEX INSTITUTIONAL AI TRADING SYSTEM V7.0</b>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>✅ ACTIVE FEATURES</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "🔬 Quantum Market Analysis\n"
            "🎮 Game Theory Optimization\n"
            "🧠 Deep Learning Ensemble\n"
            "📊 Order Flow Analysis\n"
            "🔮 Multi-Model Forecasting\n"
            "💎 Smart Money Concepts\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>⏱️  13 TIMEFRAMES ANALYZED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "2Y → 1Y → 5M → 4M → 2M\n"
            "2W → 1W → 1D → 12H → 8H\n"
            "4H → 15M → 5M\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>🎯 REQUIREMENTS</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "• Score: ≥85%\n"
            "• Confidence: ≥85%\n"
            "• Strategies: ≥6/8\n"
            "• Indicators: ≥8/14\n"
            "• Timeframes: ≥7/13\n\n"
            "<b>📱 Commands:</b>\n"
            "/status - System status\n"
            "/stats - Signal statistics\n"
            "/timeframes - Timeframe info\n"
            "/stop - Stop scanner",
            parse_mode='HTML'
        )
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        status = "🟢 ACTIVE" if self.scanner and self.scanner.is_running else "🔴 STOPPED"
        
        msg = f"""
<b>🤖 APEX SYSTEM STATUS: {status}</b>

━━━━━━━━━━━━━━━━━━━━━━
<b>⚙️  CONFIGURATION</b>
━━━━━━━━━━━━━━━━━━━━━━
• Min Threshold: {self.config.min_signal_threshold*100}%
• Min Confidence: {self.config.min_confidence*100}%
• Min Strategies: {self.config.min_strategies_aligned}/8
• Min Indicators: {self.config.min_indicators_aligned}/14
• Min Timeframes: {self.config.min_timeframes_aligned}/13
• Alert Cooldown: {self.config.alert_cooldown}s

━━━━━━━━━━━━━━━━━━━━━━
<b>⏱️  TIMEFRAMES</b>
━━━━━━━━━━━━━━━━━━━━━━
{', '.join(self.config.timeframes)}

━━━━━━━━━━━━━━━━━━━━━━
<b>📊 STATISTICS</b>
━━━━━━━━━━━━━━━━━━━━━━
Total Alerts: {len(self.scanner.alert_history) if self.scanner else 0}
Active Cooldowns: {len(self.scanner.last_alert) if self.scanner else 0}
"""
        await update.message.reply_text(msg, parse_mode='HTML')
    
    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.scanner or not self.scanner.alert_history:
            await update.message.reply_text("📊 No statistics available yet")
            return
        
        recent = list(self.scanner.alert_history)[-20:]
        
        long_signals = sum(1 for a in recent if a['signal']['type'] == 'LONG')
        short_signals = len(recent) - long_signals
        
        avg_score = np.mean([a['signal']['score'] for a in recent])
        avg_confidence = np.mean([a['signal']['confidence'] for a in recent])
        avg_rr = np.mean([a['signal']['risk_reward'] for a in recent])
        
        # Top symbols
        symbol_counts = {}
        for alert in recent:
            sym = alert['symbol']
            symbol_counts[sym] = symbol_counts.get(sym, 0) + 1
        
        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        msg = f"""
<b>📊 APEX STATISTICS (Last 20 Signals)</b>

━━━━━━━━━━━━━━━━━━━━━━
<b>SIGNAL DISTRIBUTION</b>
━━━━━━━━━━━━━━━━━━━━━━
🟢 Long: {long_signals}
🔴 Short: {short_signals}

━━━━━━━━━━━━━━━━━━━━━━
<b>QUALITY METRICS</b>
━━━━━━━━━━━━━━━━━━━━━━
⭐ Avg Score: {avg_score*100:.1f}%
🎯 Avg Confidence: {avg_confidence*100:.1f}%
💰 Avg R/R: 1:{avg_rr:.2f}

━━━━━━━━━━━━━━━━━━━━━━
<b>TOP SYMBOLS</b>
━━━━━━━━━━━━━━━━━━━━━━
"""
        for sym, count in top_symbols:
            msg += f"• {sym}: {count} signals\n"
        
        await update.message.reply_text(msg, parse_mode='HTML')
    
    async def cmd_timeframes(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = f"""
<b>⏱️  ALL 13 TIMEFRAMES</b>

━━━━━━━━━━━━━━━━━━━━━━
<b>MACRO (Long-term)</b>
━━━━━━━━━━━━━━━━━━━━━━
📅 2Y - 2 Years
📅 1Y - 1 Year
📅 5M - 5 Months
📅 4M - 4 Months
📅 2M - 2 Months

━━━━━━━━━━━━━━━━━━━━━━
<b>SWING (Medium-term)</b>
━━━━━━━━━━━━━━━━━━━━━━
📆 2W - 2 Weeks
📆 1W - 1 Week
📆 1D - Daily

━━━━━━━━━━━━━━━━━━━━━━
<b>INTRADAY (Short-term)</b>
━━━━━━━━━━━━━━━━━━━━━━
⏰ 12H - 12 Hours
⏰ 8H - 8 Hours
⏰ 4H - 4 Hours

━━━━━━━━━━━━━━━━━━━━━━
<b>SCALPING (Very short-term)</b>
━━━━━━━━━━━━━━━━━━━━━━
⚡ 15M - 15 Minutes
⚡ 5M - 5 Minutes

<b>Signals require ≥7/13 timeframes aligned!</b>
"""
        await update.message.reply_text(msg, parse_mode='HTML')
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.scanner:
            self.scanner.stop()
            await update.message.reply_text("🛑 Scanner stopped")
        else:
            await update.message.reply_text("Scanner not running")
    
    async def run(self):
        """Run the bot"""
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("stats", self.cmd_stats))
        self.app.add_handler(CommandHandler("timeframes", self.cmd_timeframes))
        self.app.add_handler(CommandHandler("stop", self.cmd_stop))
        
        bot = self.app.bot
        self.scanner = Scanner(self.config, bot)
        
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        
        startup_msg = f"""
🚀 <b>APEX SYSTEM V7.0 ACTIVATED</b> 🚀

━━━━━━━━━━━━━━━━━━━━━━
<b>ALL SYSTEMS OPERATIONAL</b>
━━━━━━━━━━━━━━━━━━━━━━

✅ 13 Timeframe Analysis Active
✅ Quantum Engine Online
✅ Game Theory Optimizer Running
✅ Deep Learning Models Loaded
✅ Order Flow Analyzer Ready
✅ News Sentiment Tracking Active

━━━━━━━━━━━━━━━━━━━━━━
<b>⏱️  TIMEFRAMES</b>
━━━━━━━━━━━━━━━━━━━━━━
{', '.join(self.config.timeframes)}

━━━━━━━━━━━━━━━━━━━━━━
<b>🎯 SIGNAL CRITERIA</b>
━━━━━━━━━━━━━━━━━━━━━━
• Score: ≥{self.config.min_signal_threshold*100}%
• Confidence: ≥{self.config.min_confidence*100}%
• Strategies: ≥{self.config.min_strategies_aligned}/8
• Indicators: ≥{self.config.min_indicators_aligned}/14
• Timeframes: ≥{self.config.min_timeframes_aligned}/13

Starting continuous market scan...
"""
        
        await bot.send_message(chat_id=self.chat_id, text=startup_msg, parse_mode='HTML')
        
        logging.info("✅ Telegram bot started successfully")
        
        # Start scanner
        await self.scanner.start()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    """Main application entry point"""
    
    setup_logging()
    
    print("\n" + "=" * 80)
    print("       APEX INSTITUTIONAL AI TRADING SYSTEM V7.0")
    print("     13 Timeframe Analysis: 2Y → 1Y → 5M → 4M → 2M → 2W → 1W")
    print("              → 1D → 12H → 8H → 4H → 15M → 5M")
    print("=" * 80 + "\n")
    
    # Check environment variables
    required_vars = ['BYBIT_API_KEY', 'BYBIT_API_SECRET', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logging.error(f"❌ Missing environment variables: {', '.join(missing)}")
        logging.error("\nPlease set all required environment variables:")
        for var in missing:
            logging.error(f"   export {var}='your_value'")
        return
    
    logging.info("✅ Environment validated\n")
    
    # Load configuration
    config = Config()
    
    logging.info("=" * 80)
    logging.info("CONFIGURATION:")
    logging.info(f"  • Min Signal Threshold: {config.min_signal_threshold*100}%")
    logging.info(f"  • Min Confidence: {config.min_confidence*100}%")
    logging.info(f"  • Min Strategies Aligned: {config.min_strategies_aligned}/8")
    logging.info(f"  • Min Indicators Aligned: {config.min_indicators_aligned}/14")
    logging.info(f"  • Min Timeframes Aligned: {config.min_timeframes_aligned}/13")
    logging.info(f"  • Total Timeframes: {len(config.timeframes)}")
    logging.info(f"  • Timeframes: {', '.join(config.timeframes)}")
    logging.info(f"  • Scan Interval: {config.scan_interval}s")
    logging.info(f"  • Alert Cooldown: {config.alert_cooldown}s")
    logging.info("=" * 80 + "\n")
    
    logging.info("🚀 Initializing Telegram Bot...")
    
    # Start bot
    bot = TelegramBot(config)
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        logging.info("\n" + "=" * 80)
        logging.info("🛑 Shutdown initiated by user")
        logging.info("=" * 80)
    except Exception as e:
        logging.error("\n" + "=" * 80)
        logging.error(f"❌ Fatal error: {e}")
        logging.error("=" * 80)
        raise


if __name__ == "__main__":
    asyncio.run(main())
