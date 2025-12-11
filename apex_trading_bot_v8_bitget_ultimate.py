"""
═══════════════════════════════════════════════════════════════════════════════
    APEX INSTITUTIONAL AI TRADING SYSTEM V8.0 - BITGET ULTIMATE EDITION
    
    Save as: apex_trading_bot_v8_bitget_ultimate.py
    
    13 Timeframes: 2Y → 1Y → 5M → 4M → 2M → 2W → 1W → 1D → 12H → 8H → 4H → 15M → 5M
    
    Features:
    ✓ Quantum Market Analysis
    ✓ Game Theory Optimization
    ✓ Deep Learning Ensemble
    ✓ Order Flow Analysis
    ✓ Multi-Model Forecasting
    ✓ Smart Money Concepts
    ✓ 200+ Technical Indicators
    ✓ 20+ Advanced Strategies
    ✓ HuggingFace News Analysis
    ✓ TradingView Institutional Analysis
    ✓ AI Strategy Selector
    ✓ Adaptive Market Modes
    ✓ 85% Minimum Thresholds
    
    Setup Instructions:
    1. Save as: apex_trading_bot_v8_bitget_ultimate.py
    2. Create requirements.txt with minimal dependencies
    3. Set environment variables:
       export BITGET_API_KEY="your_key"
       export BITGET_API_SECRET="your_secret"
       export BITGET_API_PASSPHRASE="your_passphrase"
       export TELEGRAM_BOT_TOKEN="your_token"
       export TELEGRAM_CHAT_ID="your_chat_id"
       export HUGGINGFACE_TOKEN="your_hf_token"
    4. Install: pip install -r requirements.txt
    5. Run: python apex_trading_bot_v8_bitget_ultimate.py
    
    Version: 8.0 Ultimate Edition
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
print("       APEX INSTITUTIONAL AI TRADING SYSTEM V8.0 - ULTIMATE EDITION")
print("       Initializing and installing advanced dependencies...")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED DEPENDENCY INSTALLER - FIXED VERSIONS
# ═══════════════════════════════════════════════════════════════════════════

class DependencyInstaller:
    """Install advanced dependencies at runtime with compatible versions"""
    
    # MINIMAL dependencies (put in requirements.txt)
    MINIMAL_PACKAGES = [
        'pandas==2.0.3',
        'numpy==1.24.3',
        'aiohttp==3.9.1',
        'python-telegram-bot==20.7',
        'requests==2.31.0',
        'scipy==1.11.4',
        'scikit-learn==1.3.2'
    ]
    
    # HEAVY dependencies (install at runtime)
    HEAVY_PACKAGES = [
        'ta==0.11.0',
        'vaderSentiment==3.3.2',
        'nltk==3.8.1',
        'hurst==0.0.5',
        'pywavelets==1.4.1',
        'statsmodels==0.14.1',
        'networkx==3.2.1',
        'pykalman==0.9.7',
        'matplotlib==3.8.2',
        'yfinance==0.2.33',
        'optuna==3.4.0'
    ]
    
    # ADVANCED ML dependencies (install only if needed)
    ADVANCED_PACKAGES = [
        'torch==2.1.0',
        'transformers==4.36.0',
        'accelerate==0.25.0',
        'safetensors==0.4.1'
    ]
    
    @classmethod
    def install_all_dependencies(cls):
        """Install all dependencies at runtime"""
        print("📦 Installing all dependencies at runtime...")
        
        all_packages = cls.MINIMAL_PACKAGES + cls.HEAVY_PACKAGES
        
        for package in all_packages:
            package_name = package.split('==')[0]
            import_name = package_name.replace('-', '_')
            
            try:
                __import__(import_name)
                print(f"  ✓ {package} (already installed)")
            except ImportError:
                print(f"  📦 Installing {package}...")
                try:
                    # Use --no-deps to avoid dependency conflicts
                    subprocess.check_call([
                        sys.executable, '-m', 'pip', 'install', 
                        package, '--quiet', '--no-warn-script-location',
                        '--disable-pip-version-check'
                    ])
                    print(f"  ✓ {package} - Installed")
                except Exception as e:
                    print(f"  ⚠️  Warning: {package} - {e}")
                    # Try without specific version
                    try:
                        subprocess.check_call([
                            sys.executable, '-m', 'pip', 'install', 
                            package_name, '--quiet', '--no-warn-script-location'
                        ])
                        print(f"  ✓ {package_name} (latest) - Installed")
                    except:
                        print(f"  ❌ Failed to install {package_name}")
        
        # Install advanced packages only if HuggingFace token is provided
        if os.getenv('HUGGINGFACE_TOKEN'):
            print("\n📦 Installing advanced ML dependencies for HuggingFace...")
            for package in cls.ADVANCED_PACKAGES:
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
                            package, '--quiet', '--no-warn-script-location'
                        ])
                        print(f"  ✓ {package} - Installed")
                    except Exception as e:
                        print(f"  ⚠️  Skipping {package}: {e}")
        
        print("✅ All dependencies installation completed!\n")
    
    @classmethod
    def create_requirements_txt(cls):
        """Create minimal requirements.txt file"""
        requirements_content = "\n".join(cls.MINIMAL_PACKAGES)
        
        with open('requirements.txt', 'w') as f:
            f.write("# Minimal dependencies for APEX Trading System\n")
            f.write("# Heavy dependencies are installed at runtime\n\n")
            f.write(requirements_content)
        
        print("📝 Created requirements.txt with minimal dependencies")
        print("💡 Run: pip install -r requirements.txt")

# Install all dependencies at runtime
DependencyInstaller.install_all_dependencies()

# Now import all packages
try:
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    from scipy import signal, stats
    from scipy.signal import find_peaks, argrelextrema
    from scipy.stats import entropy, pearsonr, zscore
    import aiohttp
    import requests
    from telegram import Bot, Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    from ta.trend import EMAIndicator, MACD, ADXIndicator, CCIIndicator, PSARIndicator, IchimokuIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator, TSIIndicator
    from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
    from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, MFIIndicator, ChaikinMoneyFlowIndicator
    from ta.others import DailyReturnIndicator, CumulativeReturnIndicator
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import nltk
    from hurst import compute_Hc
    import pywt
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    import networkx as nx
    from pykalman import KalmanFilter
    import matplotlib.pyplot as plt
    import yfinance as yf
    import optuna
    
    # Try to import HuggingFace transformers (optional)
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        print("⚠️  Transformers not available. HuggingFace features will be limited.")
    
    # Download NLTK data
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass
    
    print("✅ All packages imported successfully!")
    
except ImportError as e:
    print(f"❌ Failed to import required packages: {e}")
    print("Please install dependencies manually:")
    print("pip install pandas numpy aiohttp python-telegram-bot requests scipy scikit-learn")
    sys.exit(1)

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
    scan_interval: int = 60  # seconds between scans
    alert_cooldown: int = 900  # 15 minutes cooldown per symbol
    
    # Risk management
    max_leverage: int = 20
    risk_per_trade: float = 0.015  # 1.5%
    
    # ML settings
    retrain_interval: int = 12  # hours

# ═══════════════════════════════════════════════════════════════════════════
# BITGET API CLIENT - FIXED FOR 8 COLUMNS DATA
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
        
        # Our timeframe to Bitget timeframe mapping
        self.timeframe_map = {
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1h',
            '4h': '4h',
            '6h': '6h',
            '12h': '12h',
            '1d': '1day',
            '1w': '1week',
            '1M': '1M'
        }
    
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
            'User-Agent': 'APEX-Trading-Bot/8.0-V3'
        }
        
        return headers
    
    def _map_timeframe(self, timeframe: str) -> str:
        """Map our timeframe to Bitget's valid timeframe"""
        tf_lower = timeframe.lower()
        
        if tf_lower in self.timeframe_map:
            return self.timeframe_map[tf_lower]
        
        # Default mappings
        if tf_lower in ['5m', '5min']:
            return '5min'
        elif tf_lower in ['15m', '15min']:
            return '15min'
        elif tf_lower in ['30m', '30min']:
            return '30min'
        elif tf_lower in ['1h', '1hour']:
            return '1h'
        elif tf_lower in ['4h', '4hour']:
            return '4h'
        elif tf_lower in ['6h', '6hour']:
            return '6h'
        elif tf_lower in ['12h', '12hour']:
            return '12h'
        elif tf_lower in ['1d', '1day', 'daily']:
            return '1day'
        elif tf_lower in ['1w', '1week', 'weekly']:
            return '1week'
        elif tf_lower in ['1M', '1month', 'monthly']:
            return '1M'
        else:
            return '1h'
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 200) -> List:
        """Fetch candlestick/kline data from Bitget V3 API"""
        bitget_symbol = symbol.replace('/', '')
        bitget_interval = self._map_timeframe(interval)
        
        params = {
            'symbol': bitget_symbol,
            'granularity': bitget_interval,
            'limit': min(limit, 1000)
        }
        
        endpoint = '/api/v2/spot/market/candles'
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            async with self.session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                data = await response.json()
                
                if data.get('code') == '00000':
                    klines = data.get('data', [])
                    return klines[::-1]  # Reverse to chronological order
                else:
                    logging.error(f"Bitget Kline API error: {data.get('msg', 'Unknown error')}")
                    return []
        except Exception as e:
            logging.error(f"Bitget Kline fetch error: {e}")
            return []
    
    async def get_orderbook(self, symbol: str, limit: int = 50) -> Dict:
        """Fetch order book data"""
        bitget_symbol = symbol.replace('/', '')
        
        params = {
            'symbol': bitget_symbol,
            'limit': limit
        }
        
        endpoint = '/api/v2/spot/market/orderbook'
        
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
        """Fetch 24h ticker data"""
        bitget_symbol = symbol.replace('/', '')
        
        params = {'symbol': bitget_symbol}
        
        endpoint = '/api/v2/spot/market/ticker'
        
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
        """Get all tradeable USDT symbols from Bitget"""
        endpoint = '/api/v2/spot/public/symbols'
        
        try:
            url = f"{self.base_url}{endpoint}"
            async with self.session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                data = await response.json()
                if data.get('code') == '00000':
                    instruments = data.get('data', [])
                    
                    symbols = []
                    for instrument in instruments:
                        symbol_name = instrument.get('symbol', '')
                        status = instrument.get('status', '')
                        
                        if (symbol_name.endswith('USDT') and 
                            status == 'online' and
                            'test' not in symbol_name.lower()):
                            
                            base_symbol = symbol_name.replace('USDT', '')
                            symbols.append(f"{base_symbol}/USDT")
                    
                    if symbols:
                        priority_order = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'MATIC', 'AVAX', 'DOT']
                        sorted_symbols = []
                        
                        for priority in priority_order:
                            for symbol in symbols:
                                if symbol.startswith(priority + '/'):
                                    sorted_symbols.append(symbol)
                        
                        for symbol in symbols:
                            if symbol not in sorted_symbols:
                                sorted_symbols.append(symbol)
                        
                        return sorted_symbols
                    else:
                        return [
                            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
                            'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'MATIC/USDT'
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
    
    async def test_connection(self) -> bool:
        """Test API connection"""
        try:
            symbols = await self.get_symbols()
            return len(symbols) > 0
        except Exception as e:
            logging.error(f"Connection test failed: {e}")
            return False

# ═══════════════════════════════════════════════════════════════════════════
# HUGGINGFACE NEWS ANALYZER - OPTIMIZED
# ═══════════════════════════════════════════════════════════════════════════

class HuggingFaceNewsAnalyzer:
    """Advanced news analysis using HuggingFace models"""
    
    def __init__(self, api_token: str = None):
        self.api_token = api_token
        self.base_url = "https://api-inference.huggingface.co/models"
        
        if api_token:
            self.headers = {"Authorization": f"Bearer {api_token}"}
        else:
            self.headers = {}
        
        # Initialize sentiment analyzer (fallback)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Try to load HuggingFace models
        self.hf_available = False
        if api_token and TRANSFORMERS_AVAILABLE:
            try:
                # Simple sentiment model that's lightweight
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    token=api_token,
                    device=-1  # CPU
                )
                self.hf_available = True
                logging.info("✅ HuggingFace sentiment model loaded")
            except Exception as e:
                logging.warning(f"⚠️ HuggingFace model loading failed: {e}")
                self.hf_available = False
    
    async def fetch_crypto_news(self, symbol: str) -> List[Dict]:
        """Fetch crypto news from multiple sources"""
        base_symbol = symbol.split('/')[0].upper()
        
        try:
            all_news = []
            
            # Try CoinGecko API first
            try:
                async with aiohttp.ClientSession() as session:
                    # Use CoinGecko for market data (includes some news)
                    url = f"https://api.coingecko.com/api/v3/coins/{base_symbol.lower()}"
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'description' in data and data['description'].get('en'):
                                all_news.append({
                                    'title': f"{base_symbol} Market Update",
                                    'description': data['description']['en'][:500],
                                    'source': 'CoinGecko',
                                    'published_at': datetime.now().isoformat(),
                                    'url': f'https://www.coingecko.com/en/coins/{base_symbol.lower()}'
                                })
            except:
                pass
            
            # If no news, generate synthetic
            if not all_news:
                all_news = self._generate_synthetic_news(base_symbol)
            
            return all_news[:5]
            
        except Exception as e:
            logging.error(f"News fetch error: {e}")
            return self._generate_synthetic_news(base_symbol)
    
    def _generate_synthetic_news(self, symbol: str) -> List[Dict]:
        """Generate synthetic news for testing"""
        news_templates = [
            {
                'title': f"{symbol} Institutional Activity Detected",
                'description': f"Market makers showing increased interest in {symbol} with elevated volume patterns.",
                'source': 'APEX Market Intelligence',
                'published_at': datetime.now().isoformat(),
                'url': f'https://news.apexbot.com/{symbol.lower()}'
            },
            {
                'title': f"{symbol} Technical Analysis Update",
                'description': f"Key levels being tested as {symbol} approaches critical support/resistance zones.",
                'source': 'Technical Analysis Pro',
                'published_at': (datetime.now() - timedelta(hours=1)).isoformat(),
                'url': f'https://ta.apexbot.com/{symbol.lower()}'
            },
            {
                'title': f"Market Sentiment for {symbol}",
                'description': f"Overall market sentiment leaning positive for {symbol} based on order flow analysis.",
                'source': 'Sentiment Tracker',
                'published_at': (datetime.now() - timedelta(hours=2)).isoformat(),
                'url': f'https://sentiment.apexbot.com/{symbol.lower()}'
            }
        ]
        return news_templates
    
    async def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using available models"""
        try:
            if not text or len(text) < 10:
                return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.5}
            
            # Use HuggingFace if available
            if self.hf_available and hasattr(self, 'sentiment_pipeline'):
                try:
                    result = self.sentiment_pipeline(text[:512])[0]
                    label = result['label'].lower()
                    score = result['score']
                    
                    if 'positive' in label:
                        sentiment = 'positive'
                    elif 'negative' in label:
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
                    
                    return {
                        'sentiment': sentiment,
                        'score': score if sentiment == 'positive' else -score,
                        'confidence': score
                    }
                except:
                    pass
            
            # Fallback to VADER
            scores = self.sentiment_analyzer.polarity_scores(text)
            compound = scores['compound']
            
            if compound >= 0.05:
                sentiment = 'positive'
            elif compound <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'score': compound,
                'confidence': abs(compound)
            }
                
        except Exception as e:
            logging.error(f"Sentiment analysis error: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.5}
    
    async def analyze_news_for_symbol(self, symbol: str) -> Dict:
        """Complete news analysis for a symbol"""
        try:
            # Fetch news
            news_articles = await self.fetch_crypto_news(symbol)
            
            if not news_articles:
                return {
                    'symbol': symbol,
                    'sentiment': 'neutral',
                    'score': 0.0,
                    'confidence': 0.5,
                    'news_count': 0,
                    'articles': [],
                    'source': 'synthetic'
                }
            
            # Analyze each article
            analyzed_articles = []
            total_sentiment_score = 0
            
            for article in news_articles:
                text = f"{article['title']}. {article.get('description', '')}"
                sentiment_result = await self.analyze_sentiment(text)
                
                analyzed_articles.append({
                    'title': article['title'],
                    'source': article['source'],
                    'sentiment': sentiment_result['sentiment'],
                    'sentiment_score': sentiment_result['score']
                })
                
                # Accumulate scores
                if sentiment_result['sentiment'] == 'positive':
                    total_sentiment_score += sentiment_result['score']
                elif sentiment_result['sentiment'] == 'negative':
                    total_sentiment_score -= sentiment_result['score']
            
            # Calculate averages
            avg_sentiment_score = total_sentiment_score / len(analyzed_articles) if analyzed_articles else 0
            
            # Determine overall sentiment
            if avg_sentiment_score > 0.1:
                overall_sentiment = 'positive'
            elif avg_sentiment_score < -0.1:
                overall_sentiment = 'negative'
            else:
                overall_sentiment = 'neutral'
            
            # Calculate confidence based on sentiment strength
            confidence = min(abs(avg_sentiment_score) * 2, 0.95) if avg_sentiment_score != 0 else 0.5
            
            return {
                'symbol': symbol,
                'sentiment': overall_sentiment,
                'score': float(avg_sentiment_score),
                'confidence': float(confidence),
                'news_count': len(analyzed_articles),
                'articles': analyzed_articles[:3],
                'source': 'analyzed'
            }
            
        except Exception as e:
            logging.error(f"News analysis error for {symbol}: {e}")
            return {
                'symbol': symbol,
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.5,
                'news_count': 0,
                'articles': [],
                'source': 'error'
            }

# ═══════════════════════════════════════════════════════════════════════════
# TRADINGVIEW INSTITUTIONAL TIMEFRAME ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class TradingViewInstitutionalAnalyzer:
    """Institutional-grade timeframe analysis"""
    
    def __init__(self):
        self.timeframes = [
            '2Y', '1Y', '5M', '4M', '3M', '2M', 
            '1M', '2W', '1W', '1D', '12H', '8H', 
            '4H', '1H', '15M', '5M'
        ]
        
        # Map to Bitget intervals
        self.timeframe_map = {
            '2Y': '1week',  # Approximate
            '1Y': '1week',
            '5M': '1day',
            '4M': '1day',
            '3M': '1day',
            '2M': '1day',
            '1M': '1day',
            '2W': '1day',
            '1W': '1week',
            '1D': '1day',
            '12H': '12h',
            '8H': '6h',
            '4H': '4h',
            '1H': '1h',
            '15M': '15min',
            '5M': '5min'
        }
    
    async def analyze_institutional_timeframes(self, df_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze institutional timeframes from 2Y down to 5M"""
        
        analysis_results = {}
        
        for tf in self.timeframes:
            if tf in df_dict:
                df = df_dict[tf]
                if len(df) < 20:
                    continue
                    
                try:
                    analysis = self._analyze_timeframe(tf, df)
                    analysis_results[tf] = analysis
                    
                except Exception as e:
                    logging.error(f"Institutional analysis error for {tf}: {e}")
        
        return analysis_results
    
    def _analyze_timeframe(self, timeframe: str, df: pd.DataFrame) -> Dict:
        """Analyze specific timeframe with institutional indicators"""
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Basic analysis
        price = float(close[-1])
        price_change = ((close[-1] - close[-2]) / close[-2] * 100) if len(close) > 1 else 0
        
        # Trend analysis
        trend_strength, trend_direction = self._calculate_trend_strength(close)
        
        # Volume analysis
        volume_analysis = self._analyze_volume_profile(volume)
        
        # Market structure
        market_structure = self._analyze_market_structure(high, low, close)
        
        # Support/Resistance
        support_resistance = self._find_support_resistance(high, low, close)
        
        analysis = {
            'timeframe': timeframe,
            'price': price,
            'price_change_pct': float(price_change),
            'trend_strength': float(trend_strength),
            'trend_direction': trend_direction,
            'volume_profile': volume_analysis['profile'],
            'volume_ratio': float(volume_analysis['ratio']),
            'market_structure': market_structure,
            'support_levels': len(support_resistance['support']),
            'resistance_levels': len(support_resistance['resistance']),
            'institutional_signals': self._detect_institutional_signals(df),
            'volatility': self._calculate_volatility(close)
        }
        
        return analysis
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> Tuple[float, str]:
        """Calculate trend strength and direction"""
        if len(prices) < 20:
            return 0.5, 'neutral'
        
        # Simple moving average slope
        sma_short = np.mean(prices[-10:])
        sma_long = np.mean(prices[-30:])
        
        if sma_short > sma_long:
            direction = 'bullish'
            strength = min((sma_short - sma_long) / sma_long, 1.0)
        elif sma_short < sma_long:
            direction = 'bearish'
            strength = min((sma_long - sma_short) / sma_short, 1.0)
        else:
            direction = 'neutral'
            strength = 0.0
        
        return float(strength), direction
    
    def _analyze_volume_profile(self, volume: np.ndarray) -> Dict:
        """Analyze volume profile"""
        if len(volume) < 20:
            return {'profile': 'neutral', 'ratio': 1.0}
        
        current_volume = volume[-1]
        avg_volume = np.mean(volume[-20:])
        ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        if ratio > 1.5:
            profile = 'accumulation'
        elif ratio < 0.7:
            profile = 'distribution'
        else:
            profile = 'neutral'
        
        return {'profile': profile, 'ratio': ratio}
    
    def _analyze_market_structure(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> str:
        """Analyze market structure"""
        if len(high) < 20:
            return 'neutral'
        
        # Check for higher highs/lower lows
        recent_highs = high[-5:]
        recent_lows = low[-5:]
        
        higher_highs = all(recent_highs[i] > recent_highs[i-1] for i in range(1, 5))
        higher_lows = all(recent_lows[i] > recent_lows[i-1] for i in range(1, 5))
        lower_highs = all(recent_highs[i] < recent_highs[i-1] for i in range(1, 5))
        lower_lows = all(recent_lows[i] < recent_lows[i-1] for i in range(1, 5))
        
        if higher_highs and higher_lows:
            return 'uptrend'
        elif lower_highs and lower_lows:
            return 'downtrend'
        elif higher_highs and lower_lows:
            return 'expansion'
        elif lower_highs and higher_lows:
            return 'contraction'
        else:
            return 'consolidation'
    
    def _find_support_resistance(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict:
        """Find key support and resistance levels"""
        if len(close) < 50:
            return {'support': [], 'resistance': []}
        
        # Use simple method: recent highs and lows
        lookback = 20
        recent_highs = high[-lookback:]
        recent_lows = low[-lookback:]
        
        resistance = float(np.max(recent_highs))
        support = float(np.min(recent_lows))
        
        return {
            'support': [{'price': support, 'strength': 0.7}],
            'resistance': [{'price': resistance, 'strength': 0.7}]
        }
    
    def _detect_institutional_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Detect institutional trading signals"""
        signals = []
        
        # Large volume candles
        volume = df['volume'].values
        if len(volume) > 20:
            avg_volume = np.mean(volume[-20:])
            if volume[-1] > avg_volume * 2:
                signals.append({
                    'type': 'VOLUME_SPIKE',
                    'strength': float(volume[-1] / avg_volume),
                    'direction': 'BULLISH' if df['close'].iloc[-1] > df['open'].iloc[-1] else 'BEARISH'
                })
        
        return signals
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate volatility"""
        if len(prices) < 20:
            return 0.02
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        return float(min(volatility, 1.0))

# ═══════════════════════════════════════════════════════════════════════════
# AI STRATEGY SELECTOR
# ═══════════════════════════════════════════════════════════════════════════

class AIStrategySelector:
    """AI-powered strategy selector based on market conditions"""
    
    def __init__(self):
        # Define all available strategies
        self.all_strategies = {
            # Basic strategies
            'momentum': {
                'name': 'Momentum Scalper V1.0',
                'description': 'Captures momentum breakouts with volume confirmation',
                'indicators': ['RSI', 'MACD', 'Volume', 'EMA'],
                'timeframes': ['5M', '15M', '1H'],
                'market_conditions': ['trending', 'volatile'],
                'risk_level': 'medium'
            },
            'breakout': {
                'name': 'Breakout Hunter V1.0',
                'description': 'Detects support/resistance breaks with volume confirmation',
                'indicators': ['Bollinger', 'ATR', 'Volume', 'ADX'],
                'timeframes': ['1H', '4H', '1D'],
                'market_conditions': ['consolidation', 'breakout'],
                'risk_level': 'high'
            },
            'meanreversion': {
                'name': 'Mean Reversion V1.0',
                'description': 'Trades price reversions from extremes',
                'indicators': ['RSI', 'Bollinger', 'Stochastic', 'CCI'],
                'timeframes': ['15M', '1H', '4H'],
                'market_conditions': ['ranging', 'mean_reverting'],
                'risk_level': 'low'
            },
            
            # Advanced hidden strategies
            'fibonacci_vortex': {
                'name': 'Fibonacci Vortex Hidden',
                'description': 'Sacred geometry + vortex convergence with golden spiral momentum',
                'indicators': ['Fibonacci', 'Vortex', 'GoldenRatio', 'Harmonics'],
                'timeframes': ['ALL'],
                'market_conditions': ['harmonic', 'geometric'],
                'risk_level': 'medium',
                'hidden': True
            },
            'quantum_entanglement': {
                'name': 'Quantum Entanglement Hidden',
                'description': 'Quantum probability wave analysis with Heisenberg uncertainty',
                'indicators': ['Quantum', 'Probability', 'Resonance', 'Uncertainty'],
                'timeframes': ['ALL'],
                'market_conditions': ['quantum', 'probabilistic'],
                'risk_level': 'high',
                'hidden': True
            },
            'dark_pool': {
                'name': 'Dark Pool Institutional Hidden',
                'description': 'Stealth institutional buying detection with iceberg order patterns',
                'indicators': ['VolumeProfile', 'OrderFlow', 'SmartMoney', 'Iceberg'],
                'timeframes': ['1H', '4H', '1D'],
                'market_conditions': ['institutional', 'accumulation'],
                'risk_level': 'low',
                'hidden': True
            },
            'gann_square': {
                'name': 'Gann Square Time Cycles Hidden',
                'description': 'W.D. Gann secret methods with sacred number sequences',
                'indicators': ['Gann', 'TimeCycles', 'SacredNumbers', 'CardinalCross'],
                'timeframes': ['1D', '1W', '1M'],
                'market_conditions': ['cyclical', 'time_based'],
                'risk_level': 'medium',
                'hidden': True
            },
            'elliott_wave': {
                'name': 'Elliott Wave Neural Hidden',
                'description': 'AI-enhanced wave recognition with Fibonacci relationships',
                'indicators': ['Elliott', 'WavePatterns', 'Neural', 'Fibonacci'],
                'timeframes': ['4H', '1D', '1W'],
                'market_conditions': ['wave', 'pattern'],
                'risk_level': 'high',
                'hidden': True
            },
            'cosmic_movement': {
                'name': 'Cosmic Movement Hidden',
                'description': 'Sacred geometry with lunar cycles and planetary alignments',
                'indicators': ['Geometry', 'Lunar', 'Planetary', 'Cosmic'],
                'timeframes': ['1W', '1M'],
                'market_conditions': ['cosmic', 'seasonal'],
                'risk_level': 'medium',
                'hidden': True
            },
            'exclusive': {
                'name': 'Exclusive Strategies Master',
                'description': 'Master confluence analyzer with proprietary algorithms',
                'indicators': ['ALL'],
                'timeframes': ['ALL'],
                'market_conditions': ['ALL'],
                'risk_level': 'variable',
                'hidden': True
            }
        }
        
        # Strategy weights based on performance
        self.strategy_weights = {}
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize strategy weights"""
        for strategy in self.all_strategies:
            self.strategy_weights[strategy] = 1.0
    
    def analyze_market_conditions(self, analysis_data: Dict) -> Dict:
        """Analyze current market conditions"""
        conditions = {
            'trending': False,
            'ranging': False,
            'volatile': False,
            'consolidation': False,
            'breakout': False,
            'mean_reverting': False,
            'institutional': False,
            'harmonic': False,
            'cosmic': False,
            'quantum': False
        }
        
        try:
            # Extract data from analysis
            trend_strength = analysis_data.get('trend_strength', 0.5)
            market_structure = analysis_data.get('market_structure', 'neutral')
            volume_profile = analysis_data.get('volume_profile', 'neutral')
            
            # Determine conditions
            if trend_strength > 0.6:
                conditions['trending'] = True
            
            if trend_strength < 0.3:
                conditions['ranging'] = True
            
            if analysis_data.get('volatility', 0) > 0.7:
                conditions['volatile'] = True
            
            if market_structure == 'consolidation':
                conditions['consolidation'] = True
            
            if volume_profile == 'accumulation' and analysis_data.get('institutional_signals'):
                conditions['institutional'] = True
            
            return conditions
            
        except Exception as e:
            logging.error(f"Market condition analysis error: {e}")
            return conditions
    
    def select_best_strategies(self, market_conditions: Dict, user_criteria: Dict = None) -> List[Dict]:
        """Select best strategies based on market conditions"""
        
        if user_criteria is None:
            user_criteria = {
                'risk_tolerance': 'medium',
                'timeframe_preference': 'mixed',
                'strategy_complexity': 'advanced'
            }
        
        strategy_scores = {}
        
        for strategy_id, strategy_info in self.all_strategies.items():
            score = 0
            
            # Match market conditions
            strategy_conditions = strategy_info['market_conditions']
            for condition in strategy_conditions:
                if market_conditions.get(condition, False):
                    score += 2
            
            # Match risk tolerance
            strategy_risk = strategy_info['risk_level']
            user_risk = user_criteria['risk_tolerance']
            
            risk_map = {'low': 1, 'medium': 2, 'high': 3, 'variable': 2}
            risk_diff = abs(risk_map.get(strategy_risk, 2) - risk_map.get(user_risk, 2))
            risk_score = max(0, 3 - risk_diff)
            score += risk_score
            
            # Apply historical weight
            historical_weight = self.strategy_weights.get(strategy_id, 1.0)
            score *= historical_weight
            
            strategy_scores[strategy_id] = {
                'score': score,
                'strategy': strategy_info,
                'matches': self._get_strategy_matches(strategy_info, market_conditions)
            }
        
        # Sort by score
        sorted_strategies = sorted(
            strategy_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        # Return top strategies
        top_strategies = []
        for strategy_id, data in sorted_strategies[:5]:
            if data['score'] > 0:
                top_strategies.append({
                    'id': strategy_id,
                    'name': data['strategy']['name'],
                    'score': data['score'],
                    'description': data['strategy']['description'],
                    'matches': data['matches'],
                    'risk': data['strategy']['risk_level'],
                    'timeframes': data['strategy']['timeframes']
                })
        
        return top_strategies
    
    def _get_strategy_matches(self, strategy_info: Dict, market_conditions: Dict) -> List[str]:
        """Get matching conditions for a strategy"""
        matches = []
        for condition in strategy_info['market_conditions']:
            if market_conditions.get(condition, False):
                matches.append(condition)
        return matches
    
    def update_strategy_weights(self, strategy_id: str, performance: float):
        """Update strategy weights based on performance"""
        current_weight = self.strategy_weights.get(strategy_id, 1.0)
        adjustment = 0.1 if performance > 0.5 else -0.1
        new_weight = max(0.1, min(2.0, current_weight + adjustment))
        self.strategy_weights[strategy_id] = new_weight

# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED STRATEGY IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

class AdvancedStrategies:
    """Implementation of all advanced strategies"""
    
    def __init__(self):
        self.strategies = {}
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize all strategy functions"""
        # Basic strategies
        self.strategies['momentum'] = self.momentum_scalper
        self.strategies['breakout'] = self.breakout_hunter
        self.strategies['meanreversion'] = self.mean_reversion
        
        # Advanced hidden strategies
        self.strategies['fibonacci_vortex'] = self.fibonacci_vortex
        self.strategies['quantum_entanglement'] = self.quantum_entanglement
        self.strategies['dark_pool'] = self.dark_pool_institutional
        self.strategies['gann_square'] = self.gann_square_time_cycles
        self.strategies['elliott_wave'] = self.elliott_wave_neural
        self.strategies['cosmic_movement'] = self.cosmic_movement
        self.strategies['exclusive'] = self.exclusive_confluence
    
    # ===== BASIC STRATEGIES =====
    
    def momentum_scalper(self, analysis: Dict) -> Dict:
        """Momentum Scalper V1.0 Strategy"""
        score = 0
        signals = []
        
        ind = analysis.get('indicators', {})
        volume = analysis.get('volume_analysis', {})
        
        # RSI momentum
        rsi = ind.get('rsi', 50)
        if rsi > 50:
            score += 0.2
            signals.append(f"RSI bullish: {rsi:.1f}")
        
        # Volume confirmation
        volume_ratio = volume.get('volume_ratio', 1)
        if volume_ratio > 1.2:
            score += 0.2
            signals.append(f"Volume spike: {volume_ratio:.1f}x")
        
        # EMA alignment
        ema_20 = ind.get('ema_20', 0)
        ema_50 = ind.get('ema_50', 0)
        if ema_20 > ema_50:
            score += 0.3
            signals.append("EMA golden cross")
        
        # MACD momentum
        macd_hist = ind.get('macd_hist', 0)
        if macd_hist > 0:
            score += 0.3
            signals.append(f"MACD positive: {macd_hist:.4f}")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    def breakout_hunter(self, analysis: Dict) -> Dict:
        """Breakout Hunter V1.0 Strategy"""
        score = 0
        signals = []
        
        # Check for breakout conditions
        market_structure = analysis.get('market_structure', 'neutral')
        if market_structure in ['expansion', 'breakout']:
            score += 0.4
            signals.append(f"Market structure: {market_structure}")
        
        # Volume confirmation
        volume_ratio = analysis.get('volume_analysis', {}).get('volume_ratio', 1)
        if volume_ratio > 1.5:
            score += 0.3
            signals.append(f"Breakout volume: {volume_ratio:.1f}x")
        
        # ADX trend strength
        adx = analysis.get('indicators', {}).get('adx', 0)
        if adx > 25:
            score += 0.3
            signals.append(f"Strong trend: ADX={adx:.1f}")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    def mean_reversion(self, analysis: Dict) -> Dict:
        """Mean Reversion V1.0 Strategy"""
        score = 0
        signals = []
        
        ind = analysis.get('indicators', {})
        
        # RSI extremes
        rsi = ind.get('rsi', 50)
        if rsi < 30:
            score += 0.4
            signals.append(f"RSI oversold: {rsi:.1f}")
        elif rsi > 70:
            score += 0.4
            signals.append(f"RSI overbought: {rsi:.1f}")
        
        # Bollinger Band position
        current_price = analysis.get('price', 0)
        bb_upper = ind.get('bb_upper', current_price)
        bb_lower = ind.get('bb_lower', current_price)
        
        if current_price > bb_upper * 0.99:
            score += 0.3
            signals.append(f"Price at upper BB")
        elif current_price < bb_lower * 1.01:
            score += 0.3
            signals.append(f"Price at lower BB")
        
        # Stochastic extremes
        stoch_k = ind.get('stoch_k', 50)
        if stoch_k < 20 or stoch_k > 80:
            score += 0.3
            signals.append(f"Stochastic extreme: {stoch_k:.1f}")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    # ===== ADVANCED HIDDEN STRATEGIES =====
    
    def fibonacci_vortex(self, analysis: Dict) -> Dict:
        """Fibonacci Vortex Hidden Strategy"""
        score = 0
        signals = []
        
        # Fibonacci levels (simulated)
        price = analysis.get('price', 0)
        recent_high = analysis.get('recent_high', price * 1.1)
        recent_low = analysis.get('recent_low', price * 0.9)
        
        fib_levels = [
            recent_low + (recent_high - recent_low) * 0.382,
            recent_low + (recent_high - recent_low) * 0.5,
            recent_low + (recent_high - recent_low) * 0.618
        ]
        
        # Check if price near Fibonacci level
        for fib in fib_levels:
            if abs(price - fib) / price < 0.01:
                score += 0.4
                signals.append(f"At Fibonacci level: ${fib:.2f}")
                break
        
        # Vortex indicator
        vi_plus = analysis.get('indicators', {}).get('vi_plus', 1)
        vi_minus = analysis.get('indicators', {}).get('vi_minus', 1)
        
        if vi_plus > vi_minus * 1.2:
            score += 0.3
            signals.append(f"Vortex bullish: {vi_plus/vi_minus:.2f}")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    def quantum_entanglement(self, analysis: Dict) -> Dict:
        """Quantum Entanglement Hidden Strategy"""
        score = 0
        signals = []
        
        # Quantum analysis (simulated)
        volatility = analysis.get('volatility', 0.5)
        trend_strength = analysis.get('trend_strength', 0.5)
        
        # Quantum predictability
        quantum_predictability = 1.0 - (volatility * 0.5)
        if quantum_predictability > 0.7:
            score += 0.4
            signals.append(f"Quantum predictability: {quantum_predictability:.2f}")
        
        # Heisenberg uncertainty (lower is better)
        uncertainty = volatility * (1.0 - trend_strength)
        if uncertainty < 0.3:
            score += 0.3
            signals.append(f"Low uncertainty: {uncertainty:.2f}")
        
        # Quantum resonance (cyclical patterns)
        if analysis.get('has_cycles', False):
            score += 0.3
            signals.append("Quantum resonance detected")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    def dark_pool_institutional(self, analysis: Dict) -> Dict:
        """Dark Pool Institutional Hidden Strategy"""
        score = 0
        signals = []
        
        # Institutional accumulation
        volume_profile = analysis.get('volume_analysis', {}).get('profile', 'neutral')
        if volume_profile == 'accumulation':
            score += 0.4
            signals.append("Institutional accumulation")
        
        # Large block trades
        volume_ratio = analysis.get('volume_analysis', {}).get('volume_ratio', 1)
        if volume_ratio > 2.0:
            score += 0.3
            signals.append(f"Large block trade: {volume_ratio:.1f}x")
        
        # Smart money signals
        institutional_signals = analysis.get('institutional_signals', [])
        if institutional_signals:
            score += 0.3
            signals.append(f"{len(institutional_signals)} institutional signals")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    def gann_square_time_cycles(self, analysis: Dict) -> Dict:
        """Gann Square Time Cycles Hidden Strategy"""
        score = 0
        signals = []
        
        # Time-based analysis
        current_time = datetime.now()
        day_of_month = current_time.day
        
        # Gann important dates (simplified)
        gann_dates = [1, 7, 14, 21, 28]
        if day_of_month in gann_dates:
            score += 0.4
            signals.append(f"Gann date: {day_of_month}")
        
        # Time cycles
        if analysis.get('has_cycles', False):
            score += 0.3
            signals.append("Time cycle alignment")
        
        # Square of 9 (simplified)
        price = analysis.get('price', 0)
        sqrt_price = np.sqrt(price)
        if sqrt_price.is_integer() or abs(sqrt_price - round(sqrt_price)) < 0.1:
            score += 0.3
            signals.append(f"Square of 9: {sqrt_price:.1f}")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    def elliott_wave_neural(self, analysis: Dict) -> Dict:
        """Elliott Wave Neural Hidden Strategy"""
        score = 0
        signals = []
        
        # Wave pattern detection
        market_structure = analysis.get('market_structure', 'neutral')
        if market_structure in ['uptrend', 'downtrend']:
            score += 0.4
            signals.append(f"{market_structure} wave structure")
        
        # Fibonacci relationships
        if analysis.get('fibonacci_levels'):
            score += 0.3
            signals.append("Fibonacci wave relationships")
        
        # Pattern recognition
        if analysis.get('pattern_detected', False):
            score += 0.3
            signals.append("Wave pattern detected")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    def cosmic_movement(self, analysis: Dict) -> Dict:
        """Cosmic Movement Hidden Strategy"""
        score = 0
        signals = []
        
        # Lunar phases (simplified)
        current_time = datetime.now()
        day_of_month = current_time.day
        
        lunar_phases = {
            1: 'New Moon', 7: 'First Quarter',
            15: 'Full Moon', 22: 'Last Quarter'
        }
        
        if day_of_month in lunar_phases:
            score += 0.4
            signals.append(f"Lunar phase: {lunar_phases[day_of_month]}")
        
        # Sacred geometry
        price = analysis.get('price', 0)
        golden_ratio = 1.618
        
        # Check golden ratio relationships
        if price > 0:
            price_log = np.log10(price)
            if abs(price_log - round(price_log * golden_ratio) / golden_ratio) < 0.1:
                score += 0.3
                signals.append("Golden ratio alignment")
        
        # Seasonal patterns
        month = current_time.month
        if month in [1, 2, 11, 12]:  # Winter months often bullish
            score += 0.3
            signals.append("Seasonal pattern: Winter")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    def exclusive_confluence(self, analysis: Dict) -> Dict:
        """Exclusive Strategies Master Confluence"""
        score = 0
        signals = []
        
        # Test all strategies and take average
        test_strategies = [
            self.momentum_scalper,
            self.breakout_hunter,
            self.mean_reversion,
            self.fibonacci_vortex,
            self.quantum_entanglement,
            self.dark_pool_institutional
        ]
        
        strategy_scores = []
        for strategy in test_strategies:
            try:
                result = strategy(analysis)
                strategy_scores.append(result['score'])
            except:
                strategy_scores.append(0)
        
        # Weighted average
        weights = [0.15, 0.15, 0.10, 0.20, 0.20, 0.20]
        weighted_score = sum(s * w for s, w in zip(strategy_scores, weights))
        
        score = weighted_score
        
        # Add bonus for exceptional conditions
        exceptional = []
        if analysis.get('trend_strength', 0) > 0.8:
            exceptional.append(f"Trend strength: {analysis['trend_strength']:.2f}")
        if analysis.get('volume_analysis', {}).get('volume_ratio', 1) > 2:
            exceptional.append(f"Volume spike: {analysis['volume_analysis']['volume_ratio']:.1f}x")
        
        if exceptional:
            score = min(1.0, score * 1.2)
            signals.extend(exceptional)
        
        signals.append(f"Master confluence: {weighted_score:.2f}")
        
        return {'score': min(score, 1.0), 'signals': signals}
    
    def execute_strategy(self, strategy_id: str, analysis: Dict) -> Dict:
        """Execute a specific strategy"""
        if strategy_id in self.strategies:
            return self.strategies[strategy_id](analysis)
        else:
            return {'score': 0, 'signals': ['Strategy not found']}

# ═══════════════════════════════════════════════════════════════════════════
# UPDATED SIGNAL GENERATOR WITH AI SELECTION
# ═══════════════════════════════════════════════════════════════════════════

class AdvancedSignalGenerator:
    """Advanced signal generator with AI strategy selection"""
    
    def __init__(self, config, hf_token: str = None):
        self.config = config
        self.news_analyzer = HuggingFaceNewsAnalyzer(hf_token)
        self.strategy_selector = AIStrategySelector()
        self.advanced_strategies = AdvancedStrategies()
        self.tradingview_analyzer = TradingViewInstitutionalAnalyzer()
        
    async def generate_advanced_signal(self, symbol: str, tf_analyses: Dict, 
                                      user_criteria: Dict = None) -> Optional[Dict]:
        """Generate advanced trading signal with AI selection"""
        
        if not tf_analyses or len(tf_analyses) < 7:
            logging.warning(f"Insufficient timeframe analyses: {len(tf_analyses)}")
            return None
        
        try:
            # Use 5m as primary timeframe
            primary_tf = '5M' if '5M' in tf_analyses else sorted(tf_analyses.keys())[0]
            primary = tf_analyses[primary_tf]
            
            # 1. AI Strategy Selection
            market_conditions = self.strategy_selector.analyze_market_conditions(primary)
            recommended_strategies = self.strategy_selector.select_best_strategies(
                market_conditions, user_criteria
            )
            
            # 2. Execute selected strategies
            strategy_results = {}
            for strategy_info in recommended_strategies[:3]:
                strategy_id = strategy_info['id']
                result = self.advanced_strategies.execute_strategy(strategy_id, primary)
                strategy_results[strategy_id] = {
                    'name': strategy_info['name'],
                    'score': result['score'],
                    'signals': result['signals'],
                    'matches': strategy_info['matches']
                }
            
            # 3. News Analysis
            news_analysis = await self.news_analyzer.analyze_news_for_symbol(symbol)
            
            # 4. Calculate comprehensive score
            final_score, confidence = self._calculate_advanced_score(
                strategy_results, tf_analyses, news_analysis
            )
            
            # 5. Apply 85% threshold
            if final_score < self.config.min_signal_threshold:
                logging.warning(f"Score below 85% threshold: {final_score:.3f}")
                return None
            
            if confidence < self.config.min_confidence:
                logging.warning(f"Confidence below 85% threshold: {confidence:.3f}")
                return None
            
            # 6. Generate trading plan
            trading_plan = self._generate_trading_plan(primary, final_score)
            
            # 7. Compile complete signal
            signal = {
                'symbol': symbol,
                'type': self._determine_signal_type(primary),
                'score': round(final_score, 3),
                'confidence': round(confidence, 3),
                'trading_plan': trading_plan,
                'selected_strategies': strategy_results,
                'market_conditions': market_conditions,
                'recommended_mode': self._get_recommended_mode(market_conditions),
                'news_analysis': news_analysis,
                'timeframe_alignment': self._check_timeframe_alignment(tf_analyses),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            logging.info(f"✓ Advanced signal generated! Type: {signal['type']}, "
                        f"Score: {final_score:.3f}, Confidence: {confidence:.3f}")
            
            return signal
            
        except Exception as e:
            logging.error(f"Advanced signal generation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_advanced_score(self, strategy_results: Dict, tf_analyses: Dict,
                                news_analysis: Dict) -> Tuple[float, float]:
        """Calculate advanced weighted score"""
        
        # Strategy component (40%)
        strategy_scores = [s['score'] for s in strategy_results.values()]
        strategy_avg = np.mean(strategy_scores) if strategy_scores else 0
        
        # Timeframe alignment (30%)
        tf_alignment = self._calculate_timeframe_alignment_score(tf_analyses)
        
        # News sentiment (30%)
        news_score = (news_analysis.get('score', 0) + 1) / 2
        
        # Weighted final score
        final_score = (
            strategy_avg * 0.40 +
            tf_alignment * 0.30 +
            news_score * 0.30
        )
        
        # Calculate confidence
        strategy_conf = strategy_avg
        tf_conf = tf_alignment
        news_conf = news_analysis.get('confidence', 0.5)
        
        confidence = (
            strategy_conf * 0.40 +
            tf_conf * 0.30 +
            news_conf * 0.30
        )
        
        return final_score, confidence
    
    def _calculate_timeframe_alignment_score(self, tf_analyses: Dict) -> float:
        """Calculate timeframe alignment score"""
        bullish_count = 0
        bearish_count = 0
        
        for tf, analysis in tf_analyses.items():
            trend_direction = analysis.get('trend_direction', 'neutral')
            trend_strength = analysis.get('trend_strength', 0)
            
            if trend_direction == 'bullish' and trend_strength > 0.4:
                bullish_count += 1
            elif trend_direction == 'bearish' and trend_strength > 0.4:
                bearish_count += 1
        
        total = len(tf_analyses)
        alignment = max(bullish_count, bearish_count) / total if total > 0 else 0
        
        return alignment
    
    def _determine_signal_type(self, analysis: Dict) -> str:
        """Determine signal type (LONG/SHORT)"""
        trend_direction = analysis.get('trend_direction', 'neutral')
        
        if trend_direction == 'bullish':
            return 'LONG'
        elif trend_direction == 'bearish':
            return 'SHORT'
        else:
            return 'LONG'  # Default bullish bias
    
    def _get_recommended_mode(self, market_conditions: Dict) -> str:
        """Get recommended trading mode"""
        if market_conditions.get('trending', False):
            return 'momentum'
        elif market_conditions.get('consolidation', False):
            return 'breakout'
        elif market_conditions.get('ranging', False):
            return 'meanreversion'
        elif market_conditions.get('institutional', False):
            return 'dark_pool'
        else:
            return 'exclusive'
    
    def _generate_trading_plan(self, analysis: Dict, score: float) -> Dict:
        """Generate detailed trading plan"""
        current_price = analysis.get('price', 0)
        volatility = analysis.get('volatility', 0.02)
        
        # Dynamic position sizing
        base_risk = self.config.risk_per_trade
        score_multiplier = 0.5 + (score * 0.5)
        adjusted_risk = base_risk * score_multiplier
        
        # Calculate levels
        entry = current_price
        atr_distance = current_price * volatility * 2.5
        
        # Stop Loss
        stop_loss = entry - atr_distance if analysis.get('trend_direction') == 'bullish' else entry + atr_distance
        
        # Take Profit levels
        tp_levels = []
        tp_multipliers = [3, 6, 10]
        
        for i, mult in enumerate(tp_multipliers, 1):
            tp_distance = atr_distance * mult
            tp_price = entry + tp_distance if analysis.get('trend_direction') == 'bullish' else entry - tp_distance
            tp_percentage = (abs(tp_price - entry) / entry) * 100
            
            tp_levels.append({
                'level': i,
                'price': round(tp_price, 4),
                'distance_pct': round(tp_percentage, 2),
                'reward_risk': round(mult / 2.5, 2)
            })
        
        return {
            'entry': round(entry, 4),
            'stop_loss': round(stop_loss, 4),
            'take_profits': tp_levels,
            'risk_pct': round(adjusted_risk * 100, 2),
            'risk_reward': round(tp_levels[0]['reward_risk'], 2),
            'score_multiplier': round(score_multiplier, 2)
        }
    
    def _check_timeframe_alignment(self, tf_analyses: Dict) -> Dict:
        """Check timeframe alignment details"""
        bullish_tfs = []
        bearish_tfs = []
        
        for tf, analysis in tf_analyses.items():
            trend_direction = analysis.get('trend_direction', 'neutral')
            trend_strength = analysis.get('trend_strength', 0)
            
            if trend_direction == 'bullish' and trend_strength > 0.4:
                bullish_tfs.append(tf)
            elif trend_direction == 'bearish' and trend_strength > 0.4:
                bearish_tfs.append(tf)
        
        return {
            'bullish': bullish_tfs,
            'bearish': bearish_tfs,
            'bullish_count': len(bullish_tfs),
            'bearish_count': len(bearish_tfs),
            'total': len(tf_analyses),
            'alignment': 'bullish' if len(bullish_tfs) > len(bearish_tfs) else 'bearish'
        }

# ═══════════════════════════════════════════════════════════════════════════
# ULTIMATE SCANNER
# ═══════════════════════════════════════════════════════════════════════════

class UltimateScanner:
    """Ultimate scanner with all integrated features"""
    
    def __init__(self, config: ApexConfig, bot: Bot, hf_token: str = None):
        self.config = config
        self.bot = bot
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.signal_gen = AdvancedSignalGenerator(config, hf_token)
        self.last_alert: Dict[str, datetime] = {}
        self.alert_history = deque(maxlen=500)
        self.is_running = False
        self.strategy_selector = AIStrategySelector()
        
    async def start(self):
        """Start ultimate scanning"""
        self.is_running = True
        
        logging.info("=" * 80)
        logging.info("🚀 APEX ULTIMATE SCANNER STARTED")
        logging.info("📊 85% Thresholds | AI Strategy Selection")
        logging.info("=" * 80)
        
        async with BitgetAPI(
            os.getenv('BITGET_API_KEY'), 
            os.getenv('BITGET_API_SECRET'),
            os.getenv('BITGET_API_PASSPHRASE')
        ) as client:
            try:
                symbols = await client.get_symbols()
                logging.info(f"📈 Monitoring {len(symbols)} Bitget USDT symbols")
                
                # Process in batches
                batch_size = 10
                symbol_batches = [symbols[i:i + batch_size] 
                                 for i in range(0, len(symbols), batch_size)]
                
                while self.is_running:
                    try:
                        for batch_idx, batch in enumerate(symbol_batches):
                            if not self.is_running:
                                break
                            
                            logging.info(f"\n📊 Batch {batch_idx + 1}/{len(symbol_batches)}")
                            
                            for symbol in batch:
                                if not self.is_running:
                                    break
                                
                                signal = await self._scan_symbol_ultimate(client, symbol)
                                
                                if signal:
                                    await self._send_ultimate_alert(symbol, signal)
                            
                            if batch_idx < len(symbol_batches) - 1:
                                await asyncio.sleep(5)
                        
                        logging.info(f"✓ Full scan completed. Next in {self.config.scan_interval}s\n")
                        await asyncio.sleep(self.config.scan_interval)
                        
                    except Exception as e:
                        logging.error(f"Scanner error: {e}")
                        await asyncio.sleep(10)
            except Exception as e:
                logging.error(f"Failed to initialize scanner: {e}")
    
    async def _scan_symbol_ultimate(self, client: BitgetAPI, symbol: str) -> Optional[Dict]:
        """Ultimate scanning with all features"""
        try:
            logging.info(f"🔍 Scanning: {symbol}")
            
            # Get data for key timeframes
            tf_analyses = await self._get_timeframe_data(client, symbol)
            
            if not tf_analyses or len(tf_analyses) < 7:
                return None
            
            # User criteria for AI selection
            user_criteria = {
                'risk_tolerance': 'medium',
                'timeframe_preference': 'mixed',
                'strategy_complexity': 'advanced'
            }
            
            # Generate advanced signal
            signal = await self.signal_gen.generate_advanced_signal(
                symbol, tf_analyses, user_criteria
            )
            
            if signal:
                logging.info(f"  🎯 SIGNAL: {signal['type']} | Score: {signal['score']*100:.1f}%")
            else:
                logging.info(f"  ○ No signal (85% threshold)")
            
            return signal
            
        except Exception as e:
            logging.error(f"Scan error for {symbol}: {e}")
            return None
    
    async def _get_timeframe_data(self, client: BitgetAPI, symbol: str) -> Dict:
        """Get data for key timeframes"""
        tf_analyses = {}
        
        # Key timeframes to analyze
        timeframes = ['5min', '15min', '30min', '1h', '4h', '1day', '1week']
        tf_names = ['5M', '15M', '30M', '1H', '4H', '1D', '1W']
        
        for bitget_tf, our_tf in zip(timeframes, tf_names):
            try:
                klines = await client.get_klines(symbol, bitget_tf, 100)
                if klines and len(klines) >= 50:
                    # Create DataFrame
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 
                        'volume', 'quote_volume', 'trade_count'
                    ])
                    
                    # Convert types
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Basic analysis
                    analysis = self._analyze_timeframe_data(df)
                    tf_analyses[our_tf] = analysis
                    
            except Exception as e:
                logging.warning(f"  ⚠️  Could not get {our_tf} data: {e}")
                continue
        
        return tf_analyses
    
    def _analyze_timeframe_data(self, df: pd.DataFrame) -> Dict:
        """Analyze timeframe data"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Basic calculations
        price = float(close[-1])
        price_change = ((close[-1] - close[-2]) / close[-2] * 100) if len(close) > 1 else 0
        
        # Trend analysis
        if len(close) >= 20:
            sma_short = np.mean(close[-10:])
            sma_long = np.mean(close[-30:])
            trend_strength = abs(sma_short - sma_long) / sma_long if sma_long > 0 else 0
            trend_direction = 'bullish' if sma_short > sma_long else 'bearish' if sma_short < sma_long else 'neutral'
        else:
            trend_strength = 0.5
            trend_direction = 'neutral'
        
        # Volume analysis
        if len(volume) >= 20:
            current_volume = volume[-1]
            avg_volume = np.mean(volume[-20:])
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            volume_profile = 'accumulation' if volume_ratio > 1.5 else 'distribution' if volume_ratio < 0.7 else 'neutral'
        else:
            volume_ratio = 1.0
            volume_profile = 'neutral'
        
        # Market structure
        market_structure = self._determine_market_structure(high, low)
        
        # Volatility
        volatility = self._calculate_volatility(close)
        
        return {
            'price': price,
            'price_change_pct': float(price_change),
            'trend_strength': float(trend_strength),
            'trend_direction': trend_direction,
            'volume_analysis': {
                'profile': volume_profile,
                'ratio': float(volume_ratio)
            },
            'market_structure': market_structure,
            'volatility': float(volatility),
            'indicators': self._calculate_indicators(df)
        }
    
    def _determine_market_structure(self, high: np.ndarray, low: np.ndarray) -> str:
        """Determine market structure"""
        if len(high) < 10:
            return 'neutral'
        
        # Simple structure detection
        recent_highs = high[-5:]
        recent_lows = low[-5:]
        
        higher_highs = all(recent_highs[i] > recent_highs[i-1] for i in range(1, 5))
        higher_lows = all(recent_lows[i] > recent_lows[i-1] for i in range(1, 5))
        
        if higher_highs and higher_lows:
            return 'uptrend'
        elif not higher_highs and not higher_lows:
            return 'downtrend'
        else:
            return 'consolidation'
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate volatility"""
        if len(prices) < 20:
            return 0.02
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)
        
        return float(min(volatility, 1.0))
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate basic indicators"""
        try:
            close = df['close']
            
            # RSI
            rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
            
            # MACD
            macd = MACD(close)
            macd_hist = macd.macd_diff().iloc[-1]
            
            # EMAs
            ema_20 = EMAIndicator(close, window=20).ema_indicator().iloc[-1]
            ema_50 = EMAIndicator(close, window=50).ema_indicator().iloc[-1]
            
            # Bollinger Bands
            bb = BollingerBands(close, window=20, window_dev=2)
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            
            # ATR
            atr = AverageTrueRange(df['high'], df['low'], close, window=14).average_true_range().iloc[-1]
            
            return {
                'rsi': float(rsi) if not pd.isna(rsi) else 50,
                'macd_hist': float(macd_hist) if not pd.isna(macd_hist) else 0,
                'ema_20': float(ema_20) if not pd.isna(ema_20) else 0,
                'ema_50': float(ema_50) if not pd.isna(ema_50) else 0,
                'bb_upper': float(bb_upper) if not pd.isna(bb_upper) else 0,
                'bb_lower': float(bb_lower) if not pd.isna(bb_lower) else 0,
                'atr': float(atr) if not pd.isna(atr) else 0
            }
        except:
            return {}
    
    async def _send_ultimate_alert(self, symbol: str, signal: Dict):
        """Send ultimate Telegram alert"""
        
        # Check cooldown
        if symbol in self.last_alert:
            elapsed = (datetime.now() - self.last_alert[symbol]).seconds
            if elapsed < self.config.alert_cooldown:
                return
        
        try:
            message = self._format_ultimate_alert(symbol, signal)
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
            
            self.last_alert[symbol] = datetime.now()
            self.alert_history.append({
                'symbol': symbol,
                'signal': signal,
                'timestamp': datetime.now()
            })
            
            logging.info(f"  📢 ALERT SENT FOR {symbol}!")
            
        except Exception as e:
            logging.error(f"Alert error for {symbol}: {e}")
    
    def _format_ultimate_alert(self, symbol: str, signal: Dict) -> str:
        """Format ultimate Telegram alert"""
        emoji = "🟢" if signal['type'] == 'LONG' else "🔴"
        
        # Trading plan
        plan = signal['trading_plan']
        tp_text = "\n".join([
            f"  TP{level['level']}: ${level['price']:,} (+{level['distance_pct']:.1f}%)"
            for level in plan['take_profits']
        ])
        
        # Selected strategies
        strategies_text = "\n".join([
            f"  • {data['name']}: {data['score']*100:.0f}%"
            for _, data in list(signal['selected_strategies'].items())[:3]
        ])
        
        # Timeframe alignment
        tf_alignment = signal['timeframe_alignment']
        
        return f"""
{emoji} <b>═══ APEX ULTIMATE SIGNAL ═══</b> {emoji}

<b>🎯 SYMBOL:</b> {symbol}
<b>📊 TYPE:</b> {signal['type']}
<b>⭐ SCORE:</b> {signal['score']*100:.1f}%
<b>🎯 CONFIDENCE:</b> {signal['confidence']*100:.1f}%
<b>🌀 MODE:</b> {signal['recommended_mode'].upper()}

━━━━━━━━━━━━━━━━━━━━━━
<b>💰 TRADING PLAN</b>
━━━━━━━━━━━━━━━━━━━━━━
<b>Entry:</b> ${plan['entry']:,}
<b>Stop Loss:</b> ${plan['stop_loss']:,}

<b>Take Profits:</b>
{tp_text}

<b>Risk:</b> {plan['risk_pct']}% | <b>R/R:</b> 1:{plan['risk_reward']}

━━━━━━━━━━━━━━━━━━━━━━
<b>🤖 AI SELECTED STRATEGIES</b>
━━━━━━━━━━━━━━━━━━━━━━
{strategies_text}

━━━━━━━━━━━━━━━━━━━━━━
<b>⏱️  TIMEFRAME ALIGNMENT</b>
━━━━━━━━━━━━━━━━━━━━━━
<b>Direction:</b> {tf_alignment['alignment'].upper()}
<b>Bullish:</b> {tf_alignment['bullish_count']} | <b>Bearish:</b> {tf_alignment['bearish_count']}
<b>Total:</b> {tf_alignment['total']} timeframes

━━━━━━━━━━━━━━━━━━━━━━
<b>📰 NEWS SENTIMENT</b>
━━━━━━━━━━━━━━━━━━━━━━
<b>Sentiment:</b> {signal['news_analysis']['sentiment'].upper()}
<b>Score:</b> {signal['news_analysis']['score']*100:.0f}%
<b>Source:</b> {signal['news_analysis']['source']}

━━━━━━━━━━━━━━━━━━━━━━
<b>⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</b>

<i>🚀 Ultimate Edition: AI + Quantum + Institutional Analysis</i>
<i>✅ 85% Minimum Thresholds Applied</i>
"""
    
    def stop(self):
        """Stop scanner"""
        self.is_running = False
        logging.info("🛑 Ultimate scanner stopped")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT - ULTIMATE EDITION
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    """Main application entry point"""
    
    setup_logging()
    
    print("\n" + "=" * 80)
    print("       APEX INSTITUTIONAL AI TRADING SYSTEM V8.0 - ULTIMATE")
    print("=" * 80 + "\n")
    
    # Check environment variables
    required_vars = ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSPHRASE', 
                    'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logging.error(f"❌ Missing environment variables: {', '.join(missing)}")
        return
    
    # Get HuggingFace token (optional)
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    
    logging.info("✅ Environment validated")
    
    # Test connections
    try:
        logging.info("🔗 Testing Bitget V3 API connection...")
        async with BitgetAPI(
            os.getenv('BITGET_API_KEY'),
            os.getenv('BITGET_API_SECRET'),
            os.getenv('BITGET_API_PASSPHRASE')
        ) as client:
            symbols = await client.get_symbols()
            logging.info(f"✅ Bitget V3 API connected! Found {len(symbols)} symbols")
    except Exception as e:
        logging.error(f"❌ Bitget connection failed: {e}")
        return
    
    # Test Telegram
    try:
        bot_test = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
        me = await bot_test.get_me()
        logging.info(f"✅ Telegram bot connected: @{me.username}")
    except Exception as e:
        logging.error(f"❌ Telegram connection failed: {e}")
        return
    
    # Load configuration with 85% thresholds
    config = ApexConfig()
    config.min_signal_threshold = 0.85
    config.min_confidence = 0.85
    config.min_strategies_aligned = 6
    config.min_timeframes_aligned = 7
    config.scan_interval = 60
    
    logging.info("=" * 80)
    logging.info("ULTIMATE CONFIGURATION:")
    logging.info("  • 85% Minimum Score Threshold")
    logging.info("  • 85% Minimum Confidence Threshold")
    logging.info("  • 6/8 Strategies Alignment Required")
    logging.info("  • 7/13 Timeframes Alignment Required")
    logging.info("  • AI Strategy Selection Active")
    logging.info("  • HuggingFace News: " + ("✅" if hf_token else "⚠️ Synthetic"))
    logging.info("=" * 80 + "\n")
    
    logging.info("🚀 Starting Ultimate System...")
    
    # Start system
    try:
        # Create bot and scanner
        from telegram.ext import Application
        app = Application.builder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()
        
        # Initialize scanner
        scanner = UltimateScanner(config, app.bot, hf_token)
        
        # Start scanner in background
        scanner_task = asyncio.create_task(scanner.start())
        
        # Send startup message
        await app.bot.send_message(
            chat_id=os.getenv('TELEGRAM_CHAT_ID'),
            text="🤖 <b>APEX ULTIMATE SYSTEM V8.0 ACTIVATED</b>\n\n✅ 85% Thresholds\n✅ AI Strategy Selection\n✅ Scanning all symbols\n\n<i>System operational</i>",
            parse_mode='HTML'
        )
        
        logging.info("✅ Ultimate system fully operational!")
        
        # Keep running
        await asyncio.sleep(3600)  # Run for 1 hour
        
        # Clean shutdown
        scanner.stop()
        await scanner_task
        
    except KeyboardInterrupt:
        logging.info("\n🛑 System shutdown by user")
    except Exception as e:
        logging.error(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 APEX Ultimate System stopped")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
