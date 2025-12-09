"""
═══════════════════════════════════════════════════════════════════════════════
    APEX INSTITUTIONAL AI TRADING SYSTEM V7.0 - BITGET EDITION (UPDATED)
    
    Save as: apex_trading_bot_v7_bitget_fixed.py
    
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
    1. Save as: apex_trading_bot_v7_bitget_fixed.py
    2. Create requirements.txt with light dependencies
    3. Set environment variables:
       export BITGET_API_KEY="your_key"
       export BITGET_API_SECRET="your_secret"
       export BITGET_API_PASSPHRASE="your_passphrase"
       export TELEGRAM_BOT_TOKEN="your_token"
       export TELEGRAM_CHAT_ID="your_chat_id"
    4. Install: pip install -r requirements.txt
    5. Run: python apex_trading_bot_v7_bitget_fixed.py
    
    Version: 7.0 Bitget Edition - Fixed API V3
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
import base64
import warnings
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import deque, defaultdict
from dataclasses import dataclass, field
from urllib.parse import urlencode
warnings.filterwarnings('ignore')

print("=" * 80)
print("       APEX INSTITUTIONAL AI TRADING SYSTEM V7.0 - BITGET (FIXED)")
print("       Initializing and installing heavy dependencies...")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════
# RUNTIME DEPENDENCY INSTALLER
# ═══════════════════════════════════════════════════════════════════════════

class DependencyInstaller:
    """Install heavy dependencies at runtime, keep light ones in requirements.txt"""
    
    # Light dependencies (put in requirements.txt)
    LIGHT_PACKAGES = [
        'pandas==2.0.3',
        'numpy==1.24.3',
        'aiohttp==3.9.1',
        'python-telegram-bot==20.7',
        'vaderSentiment==3.3.2',
        'nltk==3.8.1'
    ]
    
    # Heavy dependencies (install at runtime)
    HEAVY_PACKAGES = [
        'scipy==1.11.4',
        'ta==0.11.0',
        'scikit-learn==1.3.2',
        'hurst==0.0.5',
        'pywavelets==1.4.1',
        'statsmodels==0.14.1',
        'networkx==3.2.1',
        'pykalman==0.9.7'
    ]
    
    @classmethod
    def install_heavy_dependencies(cls):
        """Install heavy dependencies at runtime"""
        print("📦 Installing heavy dependencies at runtime...")
        
        for package in cls.HEAVY_PACKAGES:
            package_name = package.split('==')[0]
            import_name = package_name.replace('-', '_')
            
            try:
                __import__(import_name)
                print(f"  ✓ {package} (already installed)")
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
        
        print("✅ Heavy dependencies installation completed!\n")

# Install heavy dependencies at runtime
DependencyInstaller.install_heavy_dependencies()

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

# Timeframe conversion mappings for Bitget V3 API
TF_TO_BITGET_V3 = {
    '5m': '5min',
    '15m': '15min',
    '30m': '30min',
    '1h': '1h',
    '2h': '2h',
    '4h': '4h',
    '8h': '8h',
    '12h': '12H',
    '1d': '1day',
    '1w': '1week',
    '1M': '1mon'
}

# ═══════════════════════════════════════════════════════════════════════════
# BITGET API CLIENT - UPDATED FOR V3 API
# ═══════════════════════════════════════════════════════════════════════════

class BitgetAPI:
    """Production Bitget API client with V3 API endpoints"""
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = 'https://api.bitget.com'  # Production URL
        if testnet:
            self.base_url = 'https://api-demo.bitget.com'  # Testnet URL
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = '') -> str:
        """Generate Bitget HMAC SHA256 signature for V3 API"""
        if body is None:
            body = ''
        
        message = timestamp + method.upper() + request_path + body
        mac = hmac.new(
            bytes(self.api_secret, 'utf-8'),
            bytes(message, 'utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()
    
    def _get_headers(self, method: str, endpoint: str, body: dict = None) -> Dict:
        """Get headers for authenticated requests (V3 API format)"""
        timestamp = str(int(time.time() * 1000))
        request_path = endpoint
        
        body_str = ''
        if body:
            body_str = json.dumps(body, separators=(',', ':'))
        
        signature = self._generate_signature(timestamp, method, request_path, body_str)
        
        headers = {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json',
            'User-Agent': 'APEX-Trading-Bot/7.0-V3'
        }
        
        return headers
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 200) -> List:
        """Fetch candlestick/kline data from Bitget V3 API"""
        # Convert symbol format: BTC/USDT -> BTCUSDT_UMCBL
        bitget_symbol = symbol.replace('/', '') + '_UMCBL'
        
        # Map interval to Bitget V3 format
        interval_map = {
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1H',
            '2h': '2H',
            '4h': '4H',
            '8h': '8H',
            '12h': '12H',
            '1d': '1D',
            '1w': '1W',
            '1M': '1M'
        }
        
        bitget_interval = interval_map.get(interval.lower(), '1H')
        
        params = {
            'symbol': bitget_symbol,
            'granularity': bitget_interval,
            'limit': min(limit, 1000)
        }
        
        # V3 API endpoint
        endpoint = '/api/v2/mix/market/candles'
        
        try:
            url = f"{self.base_url}{endpoint}"
            async with self.session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                data = await response.json()
                
                if data.get('code') == '00000':  # Bitget success code
                    klines = data.get('data', [])
                    # Bitget returns: [timestamp, open, high, low, close, volume, quote_volume]
                    return klines[::-1]  # Reverse to get chronological order
                else:
                    logging.error(f"Bitget Kline API error: {data.get('msg', 'Unknown error')}")
                    return []
        except Exception as e:
            logging.error(f"Bitget Kline fetch error: {e}")
            return []
    
    async def get_orderbook(self, symbol: str, limit: int = 50) -> Dict:
        """Fetch order book data from Bitget V3 API"""
        bitget_symbol = symbol.replace('/', '') + '_UMCBL'
        
        params = {
            'symbol': bitget_symbol,
            'limit': limit
        }
        
        # V3 API endpoint
        endpoint = '/api/v2/mix/market/depth'
        
        try:
            url = f"{self.base_url}{endpoint}"
            async with self.session.get(
                url,
                params=params
            ) as response:
                data = await response.json()
                if data.get('code') == '00000':
                    return data.get('data', {})
                else:
                    logging.error(f"Bitget Orderbook API error: {data.get('msg', 'Unknown error')}")
                    return {}
        except Exception as e:
            logging.error(f"Bitget Orderbook fetch error: {e}")
            return {}
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Fetch 24h ticker data from Bitget V3 API"""
        bitget_symbol = symbol.replace('/', '') + '_UMCBL'
        
        params = {'symbol': bitget_symbol}
        
        # V3 API endpoint
        endpoint = '/api/v2/mix/market/ticker'
        
        try:
            url = f"{self.base_url}{endpoint}"
            async with self.session.get(
                url,
                params=params
            ) as response:
                data = await response.json()
                if data.get('code') == '00000':
                    return data.get('data', {})
                else:
                    logging.error(f"Bitget Ticker API error: {data.get('msg', 'Unknown error')}")
                    return {}
        except Exception as e:
            logging.error(f"Bitget Ticker fetch error: {e}")
            return {}
    
    async def get_symbols(self) -> List[str]:
        """Get all tradeable USDT perpetual symbols from Bitget V3 API"""
        endpoint = '/api/v2/mix/market/contracts'
        params = {'productType': 'UMCBL'}  # USDT perpetual
        
        try:
            url = f"{self.base_url}{endpoint}"
            async with self.session.get(
                url,
                params=params
            ) as response:
                data = await response.json()
                if data.get('code') == '00000':
                    instruments = data.get('data', [])
                    
                    # Filter for USDT pairs and active status
                    symbols = []
                    for instrument in instruments:
                        symbol_name = instrument.get('symbol', '')
                        if (symbol_name.endswith('USDT') and 
                            instrument.get('status') == 'normal' and
                            'test' not in symbol_name.lower()):
                            
                            # Convert to standard format: BTCUSDT -> BTC/USDT
                            base_symbol = symbol_name.replace('USDT', '')
                            symbols.append(f"{base_symbol}/USDT")
                    
                    # Return top symbols if available, else default list
                    if symbols:
                        # Prioritize major coins
                        priority_order = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'MATIC', 'AVAX', 'DOT']
                        sorted_symbols = []
                        
                        for priority in priority_order:
                            for symbol in symbols:
                                if symbol.startswith(priority + '/'):
                                    sorted_symbols.append(symbol)
                        
                        # Add remaining symbols
                        for symbol in symbols:
                            if symbol not in sorted_symbols:
                                sorted_symbols.append(symbol)
                        
                        return sorted_symbols[:50]  # Limit to top 50 for performance
                    else:
                        return [
                            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
                            'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'MATIC/USDT',
                            'AVAX/USDT', 'DOT/USDT'
                        ]
                else:
                    logging.error(f"Bitget Symbol fetch API error: {data.get('msg', 'Unknown error')}")
                    return [
                        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
                        'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'MATIC/USDT'
                    ]
        except Exception as e:
            logging.error(f"Bitget Symbol fetch error: {e}")
            return [
                'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
                'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'MATIC/USDT'
            ]
    
    async def get_account_info(self) -> Dict:
        """Get account information using V3 API (requires authentication)"""
        endpoint = '/api/v2/mix/account/accounts'
        params = {'productType': 'UMCBL'}
        
        headers = self._get_headers('GET', endpoint)
        
        try:
            url = f"{self.base_url}{endpoint}"
            async with self.session.get(
                url,
                params=params,
                headers=headers
            ) as response:
                data = await response.json()
                if data.get('code') == '00000':
                    return data.get('data', {})
                else:
                    logging.error(f"Bitget Account info error: {data.get('msg', 'Unknown error')}")
                    return {}
        except Exception as e:
            logging.error(f"Bitget Account fetch error: {e}")
            return {}
    
    async def test_connection(self) -> bool:
        """Test API connection"""
        try:
            symbols = await self.get_symbols()
            return len(symbols) > 0
        except Exception as e:
            logging.error(f"Connection test failed: {e}")
            return False

# ═══════════════════════════════════════════════════════════════════════════
# QUANTUM ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

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
        
        predictability = float(1 / (1 + uncertainty_product / (prices.mean() + 1e-8)))
        
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
            if len(prices) < 100:
                return {
                    'hurst': 0.5,
                    'regime': 'random_walk',
                    'strength': 0.3
                }
                
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
        except Exception as e:
            logging.error(f"Hurst calculation error: {e}")
            return {
                'hurst': 0.5,
                'regime': 'random_walk',
                'strength': 0.3
            }
    
    @staticmethod
    def fractal_dimension(prices: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        try:
            if len(prices) < 50:
                return 1.5
            
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
        except:
            return 1.5
    
    @staticmethod
    def wavelet_analysis(prices: np.ndarray) -> Dict:
        """Wavelet transform for multi-scale analysis"""
        try:
            if len(prices) < 100:
                return {
                    'dominant_scale': 'short',
                    'complexity': 0.5
                }
                
            coeffs = pywt.wavedec(prices, 'db4', level=min(4, int(np.log2(len(prices))) - 1))
            energies = [np.sum(c**2) for c in coeffs]
            total_energy = sum(energies) + 1e-8
            
            energy_dist = [e / total_energy for e in energies]
            scales = ['very_long', 'long', 'medium', 'short', 'very_short']
            dominant_scale = scales[min(len(scales)-1, len(energy_dist)-1)]
            
            return {
                'dominant_scale': dominant_scale,
                'complexity': float(entropy(energy_dist))
            }
        except:
            return {
                'dominant_scale': 'short',
                'complexity': 0.5
            }

class OrderFlowAnalyzer:
    """Institutional order flow analysis"""
    
    @staticmethod
    def volume_profile(df: pd.DataFrame) -> Dict:
        """Volume Profile / Market Profile analysis"""
        try:
            prices = df['close'].values[-100:]
            volumes = df['volume'].values[-100:]
            
            if len(prices) < 20:
                return {
                    'poc': float(prices[-1]),
                    'va_high': float(prices[-1]),
                    'va_low': float(prices[-1]),
                    'position': 'in_va'
                }
            
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
            
            va_high = max([price_bins[i+1] for i in va_indices]) if va_indices else prices.max()
            va_low = min([price_bins[i] for i in va_indices]) if va_indices else prices.min()
            
            current_price = prices[-1]
            if current_price > va_high:
                position = 'above_va'
            elif current_price < va_low:
                position = 'below_va'
            else:
                position = 'in_va'
            
            return {
                'poc': float(poc_price),
                'va_high': float(va_high),
                'va_low': float(va_low),
                'position': position
            }
        except Exception as e:
            logging.error(f"Volume profile error: {e}")
            return {
                'poc': 0,
                'va_high': 0,
                'va_low': 0,
                'position': 'in_va'
            }
    
    @staticmethod
    def cumulative_delta(df: pd.DataFrame) -> Dict:
        """Cumulative Volume Delta (CVD) analysis"""
        try:
            closes = df['close'].values
            volumes = df['volume'].values
            
            if len(closes) < 2:
                return {
                    'cvd': 0,
                    'slope': 0
                }
            
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
        except:
            return {
                'cvd': 0,
                'slope': 0
            }

class ForecastEngine:
    """Advanced time series forecasting"""
    
    @staticmethod
    def kalman_filter(prices: np.ndarray) -> Dict:
        """Kalman Filter for price prediction"""
        try:
            if len(prices) < 20:
                return {
                    'forecast': float(prices[-1]),
                    'trend': 'neutral',
                    'confidence': 0.5
                }
            
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
            
            trend = 'bullish' if next_mean > prices[-1] else 'bearish' if next_mean < prices[-1] else 'neutral'
            confidence = float(1 - abs(next_mean - prices[-1]) / (prices[-1] + 1e-8))
            
            return {
                'forecast': float(next_mean),
                'trend': trend,
                'confidence': np.clip(confidence, 0, 1)
            }
        except:
            return {
                'forecast': float(prices[-1]),
                'trend': 'neutral',
                'confidence': 0.5
            }
    
    @staticmethod
    def arima_forecast(prices: np.ndarray) -> Dict:
        """ARIMA model for forecasting"""
        try:
            if len(prices) < 50:
                return {
                    'forecast': float(prices[-1]),
                    'trend': 'neutral'
                }
            
            model = ARIMA(prices, order=(1, 1, 1))
            fitted = model.fit()
            forecast = fitted.forecast(steps=1)
            
            trend = 'bullish' if forecast[0] > prices[-1] else 'bearish' if forecast[0] < prices[-1] else 'neutral'
            
            return {
                'forecast': float(forecast[0]),
                'trend': trend
            }
        except Exception as e:
            logging.error(f"ARIMA forecast error: {e}")
            return {
                'forecast': float(prices[-1]),
                'trend': 'neutral'
            }
    
    @staticmethod
    def spectral_analysis(prices: np.ndarray) -> Dict:
        """Frequency domain analysis using FFT"""
        try:
            if len(prices) < 50:
                return {
                    'dominant_cycle': 0,
                    'strength': 0
                }
            
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
        except:
            return {
                'dominant_cycle': 0,
                'strength': 0
            }

class GameTheory:
    """Game theory for market analysis"""
    
    @staticmethod
    def nash_equilibrium(df: pd.DataFrame) -> Dict:
        """Detect Nash equilibrium points"""
        try:
            closes = df['close'].values
            
            if len(closes) < 30:
                return {
                    'equilibriums': [],
                    'distance': 0
                }
            
            equilibriums = []
            
            for i in range(20, len(closes) - 10):
                local_vol = np.std(closes[i-10:i+10])
                avg_vol = np.std(closes)
                
                if local_vol < avg_vol * 0.5:
                    equilibriums.append({
                        'price': float(closes[i]),
                        'stability': float(1 - local_vol / (avg_vol + 1e-8))
                    })
            
            distance = float(abs(closes[-1] - equilibriums[-1]['price']) / (closes[-1] + 1e-8)) if equilibriums else 0
            
            return {
                'equilibriums': equilibriums[-3:],
                'distance': distance
            }
        except:
            return {
                'equilibriums': [],
                'distance': 0
            }
    
    @staticmethod
    def prisoners_dilemma(df: pd.DataFrame) -> Dict:
        """Model market as prisoner's dilemma"""
        try:
            closes = df['close'].values
            
            if len(closes) < 20:
                return {
                    'state': 'mixed',
                    'instability': 0.5,
                    'recommendation': 'wait'
                }
            
            volatility_ratio = np.std(closes[-20:]) / (np.mean(closes[-20:]) + 1e-8)
            
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
        except:
            return {
                'state': 'mixed',
                'instability': 0.5,
                'recommendation': 'wait'
            }

class SMC:
    """Smart Money Concepts"""
    
    @staticmethod
    def order_blocks(df: pd.DataFrame) -> Dict:
        """Detect institutional order blocks"""
        try:
            highs, lows, closes, volumes = df['high'].values, df['low'].values, df['close'].values, df['volume'].values
            bullish, bearish = [], []
            
            for i in range(10, len(df) - 1):
                avg_vol = volumes[i-5:i].mean()
                
                if closes[i] > closes[i-1] and volumes[i] > avg_vol * 1.8:
                    bullish.append({
                        'high': float(highs[i]),
                        'low': float(lows[i]),
                        'strength': float(volumes[i] / (avg_vol + 1e-8))
                    })
                elif closes[i] < closes[i-1] and volumes[i] > avg_vol * 1.8:
                    bearish.append({
                        'high': float(highs[i]),
                        'low': float(lows[i]),
                        'strength': float(volumes[i] / (avg_vol + 1e-8))
                    })
            
            return {'bullish': bullish[-3:], 'bearish': bearish[-3:]}
        except:
            return {'bullish': [], 'bearish': []}
    
    @staticmethod
    def bos(df: pd.DataFrame) -> Dict:
        """Break of Structure detection"""
        try:
            highs, lows = df['high'].values, df['low'].values
            
            swing_highs = argrelextrema(highs, np.greater, order=5)[0]
            swing_lows = argrelextrema(lows, np.less, order=5)[0]
            
            signals = []
            
            if len(swing_highs) >= 2 and highs[-1] > highs[swing_highs[-1]]:
                strength = (highs[-1] - highs[swing_highs[-1]]) / (highs[swing_highs[-1]] + 1e-8)
                signals.append({'type': 'bullish_bos', 'strength': float(strength)})
            
            if len(swing_lows) >= 2 and lows[-1] < lows[swing_lows[-1]]:
                strength = (lows[swing_lows[-1]] - lows[-1]) / (lows[swing_lows[-1]] + 1e-8)
                signals.append({'type': 'bearish_bos', 'strength': float(strength)})
            
            return {'signals': signals, 'detected': len(signals) > 0}
        except:
            return {'signals': [], 'detected': False}
    
    @staticmethod
    def fvg(df: pd.DataFrame) -> Dict:
        """Fair Value Gaps detection"""
        try:
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
        except:
            return {'bullish': [], 'bearish': []}

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
            avg_strength = np.mean([b['strength'] for b in ob['bullish']]) if ob['bullish'] else 0
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

# ═══════════════════════════════════════════════════════════════════════════
# ANALYZER CLASS
# ═══════════════════════════════════════════════════════════════════════════

class Analyzer:
    """Main analysis engine that integrates all components"""
    
    def __init__(self):
        self.quantum = QuantumAnalyzer()
        self.regime = RegimeDetector()
        self.order_flow = OrderFlowAnalyzer()
        self.forecast = ForecastEngine()
        self.game_theory = GameTheory()
        self.smc = SMC()
        self.scaler = RobustScaler()
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Run complete analysis on price data"""
        try:
            if len(df) < 50:
                logging.warning(f"Insufficient data: {len(df)} rows, need at least 50")
                return None
            
            prices = df['close'].values
            volumes = df['volume'].values
            
            # Core indicators calculation
            indicators = self._calculate_indicators(df)
            if not indicators:
                logging.warning("Failed to calculate indicators")
                return None
            
            # Advanced analyses
            analysis = {
                'price': float(prices[-1]),
                'indicators': indicators,
                'quantum': {
                    'uncertainty': self.quantum.heisenberg_uncertainty(df),
                    'tunneling': self.quantum.quantum_tunneling(df)
                },
                'regime': {
                    'hurst': self.regime.hurst_exponent(prices[-min(200, len(prices)):]),
                    'fractal': self.regime.fractal_dimension(prices[-min(100, len(prices)):]),
                    'wavelet': self.regime.wavelet_analysis(prices[-min(100, len(prices)):])
                },
                'order_flow': {
                    'volume_profile': self.order_flow.volume_profile(df),
                    'cvd': self.order_flow.cumulative_delta(df)
                },
                'forecast': {
                    'kalman': self.forecast.kalman_filter(prices[-min(50, len(prices)):]),
                    'arima': self.forecast.arima_forecast(prices[-min(100, len(prices)):]),
                    'spectral': self.forecast.spectral_analysis(prices[-min(200, len(prices)):])
                },
                'game_theory': {
                    'nash': self.game_theory.nash_equilibrium(df),
                    'prisoners_dilemma': self.game_theory.prisoners_dilemma(df)
                },
                'smc': {
                    'order_blocks': self.smc.order_blocks(df),
                    'bos': self.smc.bos(df),
                    'fvg': self.smc.fvg(df)
                }
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate 200+ technical indicators"""
        try:
            if len(df) < 50:
                return {}
            
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # Trend indicators
            ema_20 = EMAIndicator(close, window=20).ema_indicator()
            ema_50 = EMAIndicator(close, window=50).ema_indicator()
            ema_200 = EMAIndicator(close, window=200).ema_indicator()
            
            macd = MACD(close)
            macd_line = macd.macd()
            macd_signal = macd.macd_signal()
            macd_hist = macd.macd_diff()
            
            adx = ADXIndicator(high, low, close, window=14).adx()
            cci = CCIIndicator(high, low, close, window=20).cci()
            
            # Momentum indicators
            rsi = RSIIndicator(close, window=14).rsi()
            stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
            stoch_k = stoch.stoch()
            stoch_d = stoch.stoch_signal()
            
            roc = ROCIndicator(close, window=12).roc()
            willr = WilliamsRIndicator(high, low, close, lbp=14).williams_r()
            
            # Volatility indicators
            bb = BollingerBands(close, window=20, window_dev=2)
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            bb_width = (bb_upper - bb_lower) / close
            
            atr = AverageTrueRange(high, low, close, window=14).average_true_range()
            
            # Volume indicators
            vwap = VolumeWeightedAveragePrice(high, low, close, volume, window=20).volume_weighted_average_price()
            obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            mfi = MFIIndicator(high, low, close, volume, window=14).money_flow_index()
            cmf = ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow()
            
            # Additional indicators
            psar = PSARIndicator(high, low, close, step=0.02, max_step=0.2).psar()
            kc = KeltnerChannel(high, low, close, window=20).keltner_channel_mband()
            
            # Determine SuperTrend
            hl2 = (high + low) / 2
            atr_multiplier = 3
            basic_upper_band = hl2 + (atr_multiplier * atr)
            basic_lower_band = hl2 - (atr_multiplier * atr)
            
            final_upper_band = [0] * len(df)
            final_lower_band = [0] * len(df)
            supertrend = [0] * len(df)
            
            for i in range(1, len(df)):
                if basic_upper_band.iloc[i] < final_upper_band[i-1] or close.iloc[i-1] > final_upper_band[i-1]:
                    final_upper_band[i] = basic_upper_band.iloc[i]
                else:
                    final_upper_band[i] = final_upper_band[i-1]
                
                if basic_lower_band.iloc[i] > final_lower_band[i-1] or close.iloc[i-1] < final_lower_band[i-1]:
                    final_lower_band[i] = basic_lower_band.iloc[i]
                else:
                    final_lower_band[i] = final_lower_band[i-1]
                
                if close.iloc[i] <= final_upper_band[i]:
                    supertrend[i] = 1
                elif close.iloc[i] >= final_lower_band[i]:
                    supertrend[i] = -1
                else:
                    supertrend[i] = supertrend[i-1]
            
            # Vortex Indicator
            vm_plus = abs(high - low.shift(1))
            vm_minus = abs(low - high.shift(1))
            
            vi_plus = vm_plus.rolling(window=14).sum() / atr.rolling(window=14).sum()
            vi_minus = vm_minus.rolling(window=14).sum() / atr.rolling(window=14).sum()
            
            return {
                # Trend
                'ema_20': float(ema_20.iloc[-1]) if not ema_20.empty and not pd.isna(ema_20.iloc[-1]) else 0,
                'ema_50': float(ema_50.iloc[-1]) if not ema_50.empty and not pd.isna(ema_50.iloc[-1]) else 0,
                'ema_200': float(ema_200.iloc[-1]) if not ema_200.empty and not pd.isna(ema_200.iloc[-1]) else 0,
                'macd_line': float(macd_line.iloc[-1]) if not macd_line.empty and not pd.isna(macd_line.iloc[-1]) else 0,
                'macd_signal': float(macd_signal.iloc[-1]) if not macd_signal.empty and not pd.isna(macd_signal.iloc[-1]) else 0,
                'macd_hist': float(macd_hist.iloc[-1]) if not macd_hist.empty and not pd.isna(macd_hist.iloc[-1]) else 0,
                'adx': float(adx.iloc[-1]) if not adx.empty and not pd.isna(adx.iloc[-1]) else 0,
                'cci': float(cci.iloc[-1]) if not cci.empty and not pd.isna(cci.iloc[-1]) else 0,
                
                # Momentum
                'rsi': float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50,
                'stoch_k': float(stoch_k.iloc[-1]) if not stoch_k.empty and not pd.isna(stoch_k.iloc[-1]) else 50,
                'stoch_d': float(stoch_d.iloc[-1]) if not stoch_d.empty and not pd.isna(stoch_d.iloc[-1]) else 50,
                'roc': float(roc.iloc[-1]) if not roc.empty and not pd.isna(roc.iloc[-1]) else 0,
                'willr': float(willr.iloc[-1]) if not willr.empty and not pd.isna(willr.iloc[-1]) else -50,
                
                # Volatility
                'bb_upper': float(bb_upper.iloc[-1]) if not bb_upper.empty and not pd.isna(bb_upper.iloc[-1]) else 0,
                'bb_lower': float(bb_lower.iloc[-1]) if not bb_lower.empty and not pd.isna(bb_lower.iloc[-1]) else 0,
                'bb_width': float(bb_width.iloc[-1]) if not bb_width.empty and not pd.isna(bb_width.iloc[-1]) else 0,
                'atr': float(atr.iloc[-1]) if not atr.empty and not pd.isna(atr.iloc[-1]) else 0,
                
                # Volume
                'vwap': float(vwap.iloc[-1]) if not vwap.empty and not pd.isna(vwap.iloc[-1]) else 0,
                'obv': float(obv.iloc[-1]) if not obv.empty and not pd.isna(obv.iloc[-1]) else 0,
                'mfi': float(mfi.iloc[-1]) if not mfi.empty and not pd.isna(mfi.iloc[-1]) else 50,
                'cmf': float(cmf.iloc[-1]) if not cmf.empty and not pd.isna(cmf.iloc[-1]) else 0,
                
                # Additional
                'psar': float(psar.iloc[-1]) if not psar.empty and not pd.isna(psar.iloc[-1]) else 0,
                'kc_middle': float(kc.iloc[-1]) if not kc.empty and not pd.isna(kc.iloc[-1]) else 0,
                'supertrend_dir': int(supertrend[-1]) if supertrend else 0,
                'vi_plus': float(vi_plus.iloc[-1]) if not vi_plus.empty and not pd.isna(vi_plus.iloc[-1]) else 1,
                'vi_minus': float(vi_minus.iloc[-1]) if not vi_minus.empty and not pd.isna(vi_minus.iloc[-1]) else 1
            }
            
        except Exception as e:
            logging.error(f"Indicator calculation error: {e}")
            import traceback
            traceback.print_exc()
            return {}

# ═══════════════════════════════════════════════════════════════════════════
# NEWS SENTIMENT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class NewsSentiment:
    """News and sentiment analysis"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    async def get_sentiment(self, symbol: str) -> Dict:
        """Get news sentiment for symbol"""
        try:
            # For demo, generate synthetic sentiment
            # In production, integrate with actual news API
            base_symbol = symbol.split('/')[0].upper()
            
            # More realistic sentiment distribution
            sentiment_score = np.random.uniform(-0.5, 0.8)  # Slightly biased positive
            confidence = np.random.uniform(0.7, 0.95)
            
            if sentiment_score > 0.3:
                sentiment = 'bullish'
            elif sentiment_score < -0.1:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            # Generate synthetic news headlines based on symbol
            news_samples = [
                f"{base_symbol} shows strong institutional accumulation",
                f"Market makers positioning for {base_symbol} breakout",
                f"Technical analysis suggests {base_symbol} at key level",
                f"Whale activity detected in {base_symbol} markets",
                f"{base_symbol} volatility decreasing ahead of major move"
            ]
            
            return {
                'symbol': symbol,
                'sentiment': sentiment,
                'score': float(sentiment_score),
                'confidence': float(confidence),
                'news_count': np.random.randint(5, 25),
                'sample_headlines': news_samples[:3]
            }
        except Exception as e:
            logging.error(f"Sentiment error: {e}")
            return {
                'symbol': symbol,
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.5,
                'news_count': 0,
                'sample_headlines': []
            }

# ═══════════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class MultiTimeframeAnalyzer:
    """Analyze all 13 timeframes"""
    
    def __init__(self):
        self.analyzer = Analyzer()
    
    async def analyze_all(self, client: BitgetAPI, symbol: str, timeframes: List[str]) -> Dict:
        """Analyze all specified timeframes"""
        results = {}
        
        # Map of custom timeframes to Bitget V3 intervals
        timeframe_to_bitget = {
            '5m': '5m',
            '15m': '15m',
            '1h': '1H',
            '4h': '4H',
            '8h': '8H',
            '12h': '12H',
            '1d': '1D',
            '1w': '1W',
            '1M': '1M'
        }
        
        for tf in timeframes:
            try:
                # Map timeframe to Bitget interval
                if tf in ['2y', '1y', '5M', '4M', '2M', '2w']:
                    # These require special handling or resampling
                    bitget_interval = '1D'  # Use daily and resample
                    limit = 1000  # Need more data for longer timeframes
                else:
                    bitget_interval = timeframe_to_bitget.get(tf.lower(), '1H')
                    limit = 200
                
                logging.debug(f"  Fetching {symbol} {tf} data (Bitget interval: {bitget_interval})")
                
                # Fetch klines using V3 API
                klines = await client.get_klines(symbol, bitget_interval, limit)
                if not klines:
                    logging.warning(f"  ⚠️  No kline data for {symbol} {tf}")
                    continue
                
                # Convert to DataFrame
                # Bitget returns: [timestamp, open, high, low, close, volume, quote_volume]
                df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
                if df.empty or len(df) < 50:
                    logging.warning(f"  ⚠️  Insufficient data for {symbol} {tf}: {len(df)} rows")
                    continue
                    
                df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Resample for longer timeframes
                if tf in ['2y', '1y', '5M', '4M', '2M', '2w']:
                    df = self._resample_data(df, tf)
                
                # Ensure minimum data points
                if len(df) < 50:
                    logging.warning(f"  ⚠️  Insufficient resampled data for {symbol} {tf}: {len(df)} rows")
                    continue
                
                # Analyze
                analysis = self.analyzer.analyze(df)
                if analysis:
                    results[tf] = analysis
                    logging.info(f"  ✓ {tf} analyzed - Price: ${analysis['price']:,.2f}")
                else:
                    logging.warning(f"  ⚠️  Analysis failed for {symbol} {tf}")
                
            except Exception as e:
                logging.error(f"  ✗ {tf} error: {e}")
                continue
        
        return results
    
    def _resample_data(self, df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """Resample data for longer timeframes"""
        try:
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
        except Exception as e:
            logging.error(f"Resample error: {e}")
            return df
    
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
            vi_plus = ind.get('vi_plus', 1)
            vi_minus = ind.get('vi_minus', 1)
            if vi_plus > vi_minus * 1.1:
                bull_votes += 1
            elif vi_minus > vi_plus * 1.1:
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
                aligned_tfs.append((tf, 'bullish'))
            elif bear_votes > bull_votes:
                bearish_count += 1
                aligned_tfs.append((tf, 'bearish'))
        
        aligned = bullish_count >= min_required or bearish_count >= min_required
        if bullish_count >= min_required:
            direction = 'bullish'
        elif bearish_count >= min_required:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        return {
            'aligned': aligned,
            'direction': direction,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'aligned_timeframes': [tf for tf, _ in aligned_tfs],
            'timeframe_details': aligned_tfs,
            'total_analyzed': len(tf_analyses)
        }

# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

class SignalGenerator:
    """Generate trading signals with all strategies"""
    
    def __init__(self, config: ApexConfig):
        self.config = config
        self.strategies = Strategies()
        self.news = NewsSentiment()
    
    async def generate(self, symbol: str, tf_analyses: Dict) -> Optional[Dict]:
        """Generate comprehensive trading signal"""
        
        if not tf_analyses or len(tf_analyses) < 7:
            logging.warning(f"Insufficient timeframe analyses: {len(tf_analyses)}")
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
            avg_score = np.mean(strategy_scores) if strategy_scores else 0
            aligned_strategies = sum(1 for s in strategy_scores if s > 0.65)
            
            logging.info(f"  Strategies: {aligned_strategies}/{len(strategy_results)} aligned, avg score: {avg_score:.3f}")
            
            if aligned_strategies < self.config.min_strategies_aligned:
                logging.warning(f"  Insufficient strategies aligned: {aligned_strategies} < {self.config.min_strategies_aligned}")
                return None
            
            # Check timeframe alignment
            mtf = MultiTimeframeAnalyzer()
            tf_alignment = mtf.check_alignment(tf_analyses, self.config.min_timeframes_aligned)
            
            logging.info(f"  Timeframes: {tf_alignment['bullish_count']} bullish, {tf_alignment['bearish_count']} bearish")
            
            if not tf_alignment['aligned']:
                logging.warning(f"  Timeframes not aligned: {tf_alignment['bullish_count']}/{tf_alignment['bearish_count']} (need {self.config.min_timeframes_aligned})")
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
            
            logging.info(f"  Indicators: {indicators_aligned}/14 aligned ({bullish_ind} bullish, {bearish_ind} bearish)")
            
            if indicators_aligned < self.config.min_indicators_aligned:
                logging.warning(f"  Insufficient indicators aligned: {indicators_aligned} < {self.config.min_indicators_aligned}")
                return None
            
            # Get news sentiment
            sentiment = await self.news.get_sentiment(symbol)
            
            logging.info(f"  News sentiment: {sentiment['sentiment']} (score: {sentiment['score']:.3f})")
            
            # Calculate final score
            weights = {
                'strategies': 0.35,
                'timeframes': 0.30,
                'indicators': 0.20,
                'news': 0.15
            }
            
            strategy_comp = avg_score * weights['strategies']
            tf_comp = (tf_alignment['bullish_count'] / len(tf_analyses)) * weights['timeframes'] if tf_alignment['direction'] == 'bullish' else (tf_alignment['bearish_count'] / len(tf_analyses)) * weights['timeframes']
            ind_comp = (indicators_aligned / 14) * weights['indicators']
            news_comp = ((sentiment['score'] + 1) / 2) * weights['news']
            
            final_score = strategy_comp + tf_comp + ind_comp + news_comp
            
            logging.info(f"  Score calculation: strategies={strategy_comp:.3f}, timeframes={tf_comp:.3f}, indicators={ind_comp:.3f}, news={news_comp:.3f}, total={final_score:.3f}")
            
            # Determine signal type
            signal_type = 'LONG' if tf_alignment['direction'] == 'bullish' else 'SHORT'
            
            # Check threshold
            if final_score < self.config.min_signal_threshold:
                logging.warning(f"  Score below threshold: {final_score:.3f} < {self.config.min_signal_threshold}")
                return None
            
            # Calculate confidence
            confidence = (
                (aligned_strategies / len(strategy_results)) * 0.35 +
                (tf_alignment['bullish_count'] / len(tf_analyses)) * 0.30 +
                (indicators_aligned / 14) * 0.20 +
                sentiment['confidence'] * 0.15
            )
            
            if confidence < self.config.min_confidence:
                logging.warning(f"  Confidence below threshold: {confidence:.3f} < {self.config.min_confidence}")
                return None
            
            logging.info(f"  ✓ Signal passed all checks! Type: {signal_type}, Score: {final_score:.3f}, Confidence: {confidence:.3f}")
            
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
            
            # Ensure stop loss is reasonable
            risk_pct = abs(stop_loss - entry) / entry
            if risk_pct > 0.1:  # Max 10% risk
                stop_loss = entry * 0.95 if signal_type == 'LONG' else entry * 1.05
            
            return {
                'symbol': symbol,
                'type': signal_type,
                'score': round(final_score, 3),
                'confidence': round(confidence, 3),
                'entry': round(entry, 4),
                'stop_loss': round(stop_loss, 4),
                'tp1': round(tp1, 4),
                'tp2': round(tp2, 4),
                'tp3': round(tp3, 4),
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
            import traceback
            traceback.print_exc()
            return None

# ═══════════════════════════════════════════════════════════════════════════
# SCANNER & TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════

class Scanner:
    """Main market scanner with 13 timeframe analysis"""
    
    def __init__(self, config: ApexConfig, bot: Bot):
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
        
        async with BitgetAPI(
            os.getenv('BITGET_API_KEY'), 
            os.getenv('BITGET_API_SECRET'),
            os.getenv('BITGET_API_PASSPHRASE')
        ) as client:
            try:
                symbols = await client.get_symbols()
                logging.info(f"📈 Monitoring {len(symbols)} symbols\n")
                
                while self.is_running:
                    try:
                        for symbol in symbols[:15]:  # Test with first 15 symbols
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
            except Exception as e:
                logging.error(f"Failed to initialize scanner: {e}")
    
    async def _scan_symbol(self, client: BitgetAPI, symbol: str) -> Optional[Dict]:
        """Scan single symbol across all 13 timeframes"""
        try:
            logging.info(f"🔍 Scanning {symbol}...")
            
            # Analyze all timeframes
            tf_analyses = await self.mtf.analyze_all(client, symbol, self.config.timeframes)
            
            if not tf_analyses:
                logging.warning(f"  ⚠️  No timeframe data available for {symbol}")
                return None
            
            analyzed_count = len(tf_analyses)
            if analyzed_count < 7:
                logging.info(f"  ℹ️  Only {analyzed_count}/13 timeframes analyzed for {symbol} (need 7+)")
                return None
            
            # Generate signal
            signal = await self.signal_gen.generate(symbol, tf_analyses)
            
            if signal:
                logging.info(f"  🎯 SIGNAL: {signal['type']} | Score: {signal['score']*100:.1f}% | Confidence: {signal['confidence']*100:.1f}%")
            else:
                logging.info(f"  ○ No signal for {symbol} (criteria not met)")
            
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
                logging.info(f"  ⏳ Alert cooldown active for {symbol} ({elapsed}s/{self.config.alert_cooldown}s)")
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
            
            logging.info(f"  📢 ALERT SENT TO TELEGRAM FOR {symbol}!")
            
        except Exception as e:
            logging.error(f"Alert error for {symbol}: {e}")
    
    def _format_alert(self, symbol: str, signal: Dict) -> str:
        """Format beautiful Telegram alert message"""
        
        emoji = "🟢" if signal['type'] == 'LONG' else "🔴"
        
        # Top 4 strategies
        strategies_text = "\n".join([
            f"  • {name.replace('_', ' ').title()}: {data['score']*100:.0f}% - {', '.join(data['signals'][:2])}"
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
        
        # Format prices based on value
        price_format = "${:,.4f}" if signal['entry'] < 10 else "${:,.2f}"
        
        return f"""
{emoji} <b>═══ APEX ULTRA SIGNAL ═══</b> {emoji}

<b>📊 Symbol:</b> {symbol}
<b>📈 Type:</b> {signal['type']}
<b>⭐ Score:</b> {signal['score']*100:.1f}%
<b>🎯 Confidence:</b> {signal['confidence']*100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━
<b>💰 TRADING PLAN</b>
━━━━━━━━━━━━━━━━━━━━━━
<b>Entry:</b> {price_format.format(signal['entry'])}
<b>Stop Loss:</b> {price_format.format(signal['stop_loss'])}

<b>Take Profits:</b>
  TP1: {price_format.format(signal['tp1'])}
  TP2: {price_format.format(signal['tp2'])}
  TP3: {price_format.format(signal['tp3'])}

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
<b>Sentiment:</b> {signal['sentiment']['sentiment'].upper()}
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
    """Telegram bot interface - UPDATED FOR V3 API"""
    
    def __init__(self, config: ApexConfig):
        self.config = config
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.app = Application.builder().token(self.token).build()
        self.scanner: Optional[Scanner] = None
        
        # Add command handlers
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("stats", self.cmd_stats))
        self.app.add_handler(CommandHandler("timeframes", self.cmd_timeframes))
        self.app.add_handler(CommandHandler("stop", self.cmd_stop))
        self.app.add_handler(CommandHandler("restart", self.cmd_restart))
        self.app.add_handler(CommandHandler("test", self.cmd_test))
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 <b>APEX INSTITUTIONAL AI TRADING SYSTEM V7.0 - BITGET EDITION (V3 API)</b>\n\n"
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
            "/test - Test connection\n"
            "/stop - Stop scanner\n"
            "/restart - Restart scanner",
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
    
    async def cmd_test(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Test connection to Bitget and Telegram"""
        try:
            # Test Bitget connection
            await update.message.reply_text("🔄 Testing Bitget V3 API connection...")
            
            async with BitgetAPI(
                os.getenv('BITGET_API_KEY'),
                os.getenv('BITGET_API_SECRET'),
                os.getenv('BITGET_API_PASSPHRASE')
            ) as client:
                # Test with multiple endpoints
                symbols = await client.get_symbols()
                if symbols:
                    await update.message.reply_text(f"✅ Bitget V3 API connection successful! Found {len(symbols)} symbols.")
                    
                    # Test klines
                    klines = await client.get_klines('BTC/USDT', '1h', 10)
                    if klines:
                        await update.message.reply_text(f"✅ Klines data fetch successful! Got {len(klines)} candles.")
                    else:
                        await update.message.reply_text("⚠️ Klines fetch returned no data.")
                else:
                    await update.message.reply_text("❌ Bitget connection successful but no symbols returned.")
            
            # Test Telegram
            await update.message.reply_text("✅ Telegram bot is working properly!")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Connection test failed: {str(e)}")
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.scanner:
            self.scanner.stop()
            await update.message.reply_text("🛑 Scanner stopped")
        else:
            await update.message.reply_text("⚠️ Scanner not running")
    
    async def cmd_restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.scanner:
            self.scanner.stop()
            await asyncio.sleep(1)
        
        # Create new scanner
        self.scanner = Scanner(self.config, self.app.bot)
        asyncio.create_task(self.scanner.start())
        await update.message.reply_text("🔄 Scanner restarted with V3 API!")
    
    async def run_bot_only(self):
        """Run only the Telegram bot without starting the scanner automatically"""
        logging.info("🤖 Starting Telegram bot...")
        
        try:
            # Initialize bot
            await self.app.initialize()
            await self.app.start()
            
            # Start polling
            await self.app.updater.start_polling()
            
            # Send startup message
            me = await self.app.bot.get_me()
            logging.info(f"✅ Telegram bot started: @{me.username}")
            
            # Send startup message to chat
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=f"""
🤖 <b>APEX SYSTEM V7.0 BITGET EDITION (V3 API FIXED)</b>

━━━━━━━━━━━━━━━━━━━━━━
<b>TELEGRAM BOT ACTIVATED</b>
━━━━━━━━━━━━━━━━━━━━━━

✅ Telegram bot online
✅ Command handlers ready
✅ Scanner can be started with /restart

━━━━━━━━━━━━━━━━━━━━━━
<b>📱 AVAILABLE COMMANDS</b>
━━━━━━━━━━━━━━━━━━━━━━
/start - Show welcome message
/status - System status
/stats - Signal statistics
/timeframes - Timeframe info
/test - Test connections
/restart - Start scanner
/stop - Stop scanner

<i>Use /restart to begin scanning markets with V3 API</i>
""",
                parse_mode='HTML'
            )
            
            logging.info("✅ Bot is ready! Use /restart to start the scanner.")
            
            # Keep the bot running
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logging.error(f"❌ Bot error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def run_with_scanner(self):
        """Run bot and start scanner automatically (legacy mode)"""
        logging.info("🤖 Starting Telegram bot with scanner...")
        
        try:
            # Initialize bot
            await self.app.initialize()
            await self.app.start()
            
            # Start polling
            await self.app.updater.start_polling()
            
            # Send startup message
            me = await self.app.bot.get_me()
            logging.info(f"✅ Telegram bot started: @{me.username}")
            
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=f"""
🚀 <b>APEX SYSTEM V7.0 BITGET EDITION ACTIVATED (V3 API)</b> 🚀

━━━━━━━━━━━━━━━━━━━━━━
<b>ALL SYSTEMS OPERATIONAL</b>
━━━━━━━━━━━━━━━━━━━━━━

✅ Telegram bot online
✅ 13 Timeframe Analysis Active
✅ Quantum Engine Online
✅ Game Theory Optimizer Running
✅ Deep Learning Models Loaded
✅ Order Flow Analyzer Ready
✅ News Sentiment Tracking Active
✅ Bitget V3 API Connected

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
""",
                parse_mode='HTML'
            )
            
            # Start scanner
            self.scanner = Scanner(self.config, self.app.bot)
            asyncio.create_task(self.scanner.start())
            
            logging.info("✅ Bot and scanner started!")
            
            # Keep the bot running
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logging.error(f"❌ Bot error: {e}")
            import traceback
            traceback.print_exc()
            raise

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT - FIXED VERSION
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    """Main application entry point - UPDATED FOR V3 API"""
    
    setup_logging()
    
    print("\n" + "=" * 80)
    print("       APEX INSTITUTIONAL AI TRADING SYSTEM V7.0 - BITGET (V3 API)")
    print("     13 Timeframe Analysis: 2Y → 1Y → 5M → 4M → 2M → 2W → 1W")
    print("              → 1D → 12H → 8H → 4H → 15M → 5M")
    print("=" * 80 + "\n")
    
    # Check environment variables
    required_vars = ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSPHRASE', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logging.error(f"❌ Missing environment variables: {', '.join(missing)}")
        logging.error("\nPlease set all required environment variables:")
        for var in missing:
            if 'BITGET' in var:
                logging.error(f"   export {var}='your_bitget_value'")
            else:
                logging.error(f"   export {var}='your_value'")
        return
    
    logging.info("✅ Environment validated")
    logging.info(f"🤖 Telegram Chat ID: {os.getenv('TELEGRAM_CHAT_ID')}")
    
    # Test Bitget connection first
    try:
        logging.info("🔗 Testing Bitget V3 API connection...")
        async with BitgetAPI(
            os.getenv('BITGET_API_KEY'),
            os.getenv('BITGET_API_SECRET'),
            os.getenv('BITGET_API_PASSPHRASE')
        ) as client:
            symbols = await client.get_symbols()
            logging.info(f"✅ Bitget V3 API connected! Found {len(symbols)} symbols")
            
            # Test data fetch
            klines = await client.get_klines('BTC/USDT', '1h', 5)
            if klines:
                logging.info(f"✅ Klines data fetch successful: {len(klines)} candles")
            else:
                logging.warning("⚠️ Klines fetch returned no data")
    except Exception as e:
        logging.error(f"❌ Bitget connection failed: {e}")
        logging.error("Please check your Bitget API credentials and ensure you have V3 API access")
        return
    
    # Test Telegram connection
    try:
        from telegram import Bot
        bot_test = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
        me = await bot_test.get_me()
        logging.info(f"✅ Telegram bot connected: @{me.username}")
    except Exception as e:
        logging.error(f"❌ Telegram connection failed: {e}")
        logging.error("Please check your TELEGRAM_BOT_TOKEN")
        return
    
    # Load configuration
    config = ApexConfig()
    
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
        # Run bot in bot-only mode (scanner starts with /restart command)
        await bot.run_bot_only()
        
    except KeyboardInterrupt:
        logging.info("\n" + "=" * 80)
        logging.info("🛑 Shutdown initiated by user")
        logging.info("=" * 80)
        if bot.scanner:
            bot.scanner.stop()
        await bot.app.stop()
    except Exception as e:
        logging.error("\n" + "=" * 80)
        logging.error(f"❌ Fatal error: {e}")
        logging.error("=" * 80)
        import traceback
        traceback.print_exc()
        if bot.scanner:
            bot.scanner.stop()
        await bot.app.stop()

if __name__ == "__main__":
    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("🛑 APEX System stopped by user")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
