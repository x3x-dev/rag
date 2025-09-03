# Market Data Processing Module

## Overview

The `minglib.market_data` module provides comprehensive tools for processing, cleaning, and analyzing financial market data from multiple sources. This module handles real-time data feeds, historical data retrieval, data quality checks, and advanced market microstructure analysis.

## Installation

```python
from minglib.market_data import (
    DataFeed,
    HistoricalDataRetriever,
    MarketDataCleaner,
    VolumeAnalyzer,
    TickDataProcessor,
    CrossAssetCorrelations
)
```

## Core Classes

### DataFeed

Real-time market data feed handler with support for multiple data providers.

#### Syntax

```python
class DataFeed:
    def __init__(
        self,
        provider: str,
        api_key: str = None,
        symbols: list = None,
        data_types: list = ["trades", "quotes", "bars"]
    )
```

#### Parameters

- **provider** (str): Data provider name ("bloomberg", "refinitiv", "alpha_vantage", "yfinance")
- **api_key** (str, optional): API authentication key. Default: None
- **symbols** (list, optional): List of symbols to subscribe to. Default: None
- **data_types** (list, optional): Types of data to receive. Default: ["trades", "quotes", "bars"]

#### Methods

##### connect()

Establish connection to data provider.

```python
def connect(self) -> bool:
    """
    Establish connection to the data provider.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
```

##### subscribe()

Subscribe to real-time data for specified symbols.

```python
def subscribe(
    self,
    symbols: list,
    data_types: list = ["trades"],
    frequency: str = "tick"
) -> dict:
    """
    Subscribe to real-time market data.
    
    Parameters:
        symbols (list): List of symbols to subscribe to
        data_types (list): Types of data to receive
        frequency (str): Data frequency ("tick", "1s", "1min", "5min")
    
    Returns:
        dict: Subscription status and details
    """
```

#### Example

```python
from minglib.market_data import DataFeed
import pandas as pd

# Initialize data feed
feed = DataFeed(
    provider="alpha_vantage",
    api_key="your_api_key_here",
    symbols=["AAPL", "MSFT", "GOOGL"]
)

# Connect to data provider
if feed.connect():
    print("Successfully connected to data provider")
    
    # Subscribe to real-time trades and quotes
    subscription = feed.subscribe(
        symbols=["AAPL", "MSFT"],
        data_types=["trades", "quotes"],
        frequency="tick"
    )
    
    print(f"Subscription status: {subscription['status']}")
    print(f"Subscribed symbols: {subscription['symbols']}")
    
    # Get real-time data
    while True:
        data = feed.get_latest_data()
        if data:
            print(f"Latest trade: {data['trades']}")
            print(f"Latest quote: {data['quotes']}")
        
        # Process data or break loop
        break
else:
    print("Failed to connect to data provider")
```

### HistoricalDataRetriever

Retrieve and manage historical market data with advanced filtering and aggregation.

#### Syntax

```python
class HistoricalDataRetriever:
    def __init__(
        self,
        provider: str = "yfinance",
        cache_data: bool = True,
        cache_directory: str = "./data_cache"
    )
```

#### Methods

##### get_price_data()

Retrieve historical price data for specified symbols.

```python
def get_price_data(
    self,
    symbols: list,
    start_date: str,
    end_date: str,
    frequency: str = "1d",
    adjusted: bool = True,
    include_volume: bool = True
) -> pd.DataFrame:
    """
    Retrieve historical price data.
    
    Parameters:
        symbols (list): List of stock symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        frequency (str): Data frequency ('1d', '1h', '5m', '1m')
        adjusted (bool): Whether to use adjusted prices
        include_volume (bool): Whether to include volume data
    
    Returns:
        pd.DataFrame: Historical price data with MultiIndex columns
    """
```

#### Example

```python
from minglib.market_data import HistoricalDataRetriever
from datetime import datetime, timedelta

# Initialize data retriever
retriever = HistoricalDataRetriever(
    provider="yfinance",
    cache_data=True
)

# Get historical data for multiple stocks
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

price_data = retriever.get_price_data(
    symbols=symbols,
    start_date=start_date.strftime("%Y-%m-%d"),
    end_date=end_date.strftime("%Y-%m-%d"),
    frequency="1d",
    adjusted=True
)

print(f"Retrieved data shape: {price_data.shape}")
print(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")
print("\nSample data:")
print(price_data.head())

# Calculate returns
returns = price_data.pct_change().dropna()
print(f"\nDaily returns statistics:")
print(returns.describe())
```

##### get_fundamental_data()

Retrieve fundamental data for financial analysis.

```python
def get_fundamental_data(
    self,
    symbols: list,
    metrics: list = ["market_cap", "pe_ratio", "debt_to_equity"],
    period: str = "annual"
) -> pd.DataFrame:
    """
    Retrieve fundamental data for specified symbols.
    
    Parameters:
        symbols (list): List of stock symbols
        metrics (list): Fundamental metrics to retrieve
        period (str): Data period ('annual', 'quarterly')
    
    Returns:
        pd.DataFrame: Fundamental data
    """
```

### MarketDataCleaner

Advanced data cleaning and quality assurance for market data.

#### Syntax

```python
class MarketDataCleaner:
    def __init__(
        self,
        outlier_method: str = "iqr",
        fill_method: str = "forward_fill",
        validation_rules: dict = None
    )
```

#### Methods

##### clean_price_data()

Clean and validate price data with multiple quality checks.

```python
def clean_price_data(
    self,
    data: pd.DataFrame,
    remove_outliers: bool = True,
    fill_missing: bool = True,
    validate_prices: bool = True
) -> dict:
    """
    Clean price data with comprehensive quality checks.
    
    Parameters:
        data (pd.DataFrame): Raw price data
        remove_outliers (bool): Whether to remove outliers
        fill_missing (bool): Whether to fill missing values
        validate_prices (bool): Whether to validate price relationships
    
    Returns:
        dict: Cleaned data and quality report
    """
```

#### Example

```python
from minglib.market_data import MarketDataCleaner
import numpy as np

# Initialize data cleaner
cleaner = MarketDataCleaner(
    outlier_method="iqr",
    fill_method="interpolation"
)

# Simulate noisy price data
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
clean_prices = np.cumsum(np.random.normal(0.001, 0.02, len(dates))) + 100

# Add some data quality issues
noisy_prices = clean_prices.copy()
noisy_prices[50] = np.nan  # Missing value
noisy_prices[100] = clean_prices[100] * 3  # Outlier
noisy_prices[150] = 0  # Invalid price

# Create DataFrame
raw_data = pd.DataFrame({
    'AAPL': noisy_prices,
    'Close': noisy_prices
}, index=dates)

# Clean the data
cleaning_result = cleaner.clean_price_data(
    data=raw_data,
    remove_outliers=True,
    fill_missing=True,
    validate_prices=True
)

cleaned_data = cleaning_result['cleaned_data']
quality_report = cleaning_result['quality_report']

print("Data Quality Report:")
print(f"Missing values found: {quality_report['missing_values']}")
print(f"Outliers detected: {quality_report['outliers_detected']}")
print(f"Invalid prices: {quality_report['invalid_prices']}")
print(f"Data completeness: {quality_report['completeness_score']:.2%}")
```

### VolumeAnalyzer

Analyze trading volume patterns and market microstructure.

#### Syntax

```python
class VolumeAnalyzer:
    def __init__(
        self,
        volume_profile_bins: int = 50,
        time_window: str = "1d"
    )
```

#### Methods

##### calculate_vwap()

Calculate Volume Weighted Average Price.

```python
def calculate_vwap(
    self,
    price_data: pd.DataFrame,
    volume_data: pd.DataFrame,
    window: str = "1d"
) -> pd.DataFrame:
    """
    Calculate Volume Weighted Average Price.
    
    Parameters:
        price_data (pd.DataFrame): Price data (OHLC)
        volume_data (pd.DataFrame): Volume data
        window (str): VWAP calculation window
    
    Returns:
        pd.DataFrame: VWAP values
    """
```

##### volume_profile()

Generate volume profile analysis.

```python
def volume_profile(
    self,
    price_data: pd.DataFrame,
    volume_data: pd.DataFrame,
    bins: int = 50
) -> dict:
    """
    Generate volume profile for price levels.
    
    Parameters:
        price_data (pd.DataFrame): Price data
        volume_data (pd.DataFrame): Volume data
        bins (int): Number of price bins
    
    Returns:
        dict: Volume profile data
    """
```

#### Example

```python
from minglib.market_data import VolumeAnalyzer

# Initialize volume analyzer
analyzer = VolumeAnalyzer(volume_profile_bins=100)

# Sample price and volume data
dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
prices = pd.DataFrame({
    'Open': np.random.normal(100, 2, 252),
    'High': np.random.normal(102, 2, 252),
    'Low': np.random.normal(98, 2, 252),
    'Close': np.random.normal(100, 2, 252)
}, index=dates)

volumes = pd.DataFrame({
    'Volume': np.random.lognormal(15, 0.5, 252)
}, index=dates)

# Calculate VWAP
vwap = analyzer.calculate_vwap(
    price_data=prices,
    volume_data=volumes,
    window="5d"
)

print("VWAP Analysis:")
print(vwap.head())

# Generate volume profile
vol_profile = analyzer.volume_profile(
    price_data=prices,
    volume_data=volumes,
    bins=50
)

print(f"\nVolume Profile:")
print(f"Price of maximum volume: {vol_profile['poc_price']:.2f}")
print(f"Value area high: {vol_profile['value_area_high']:.2f}")
print(f"Value area low: {vol_profile['value_area_low']:.2f}")
```

### CrossAssetCorrelations

Analyze correlations and relationships across different asset classes.

#### Syntax

```python
class CrossAssetCorrelations:
    def __init__(
        self,
        correlation_method: str = "pearson",
        rolling_window: int = 252
    )
```

#### Methods

##### dynamic_correlation()

Calculate time-varying correlations between assets.

```python
def dynamic_correlation(
    self,
    returns_data: pd.DataFrame,
    window: int = 252,
    min_periods: int = 100
) -> pd.DataFrame:
    """
    Calculate rolling correlations between assets.
    
    Parameters:
        returns_data (pd.DataFrame): Asset returns data
        window (int): Rolling window size
        min_periods (int): Minimum periods for calculation
    
    Returns:
        pd.DataFrame: Time-varying correlation matrix
    """
```

#### Example

```python
from minglib.market_data import CrossAssetCorrelations

# Initialize correlation analyzer
corr_analyzer = CrossAssetCorrelations(
    correlation_method="pearson",
    rolling_window=60
)

# Sample returns data for different asset classes
np.random.seed(42)
asset_returns = pd.DataFrame({
    'Stocks': np.random.normal(0.0008, 0.015, 252),
    'Bonds': np.random.normal(0.0003, 0.005, 252),
    'Commodities': np.random.normal(0.0005, 0.020, 252),
    'FX': np.random.normal(0.0001, 0.008, 252),
    'Crypto': np.random.normal(0.0010, 0.040, 252)
}, index=pd.date_range('2023-01-01', periods=252, freq='D'))

# Calculate dynamic correlations
dynamic_corr = corr_analyzer.dynamic_correlation(
    returns_data=asset_returns,
    window=60,
    min_periods=30
)

print("Dynamic Correlation Analysis:")
print(f"Average correlation between Stocks and Bonds: {dynamic_corr['Stocks']['Bonds'].mean():.3f}")
print(f"Maximum correlation between Stocks and Crypto: {dynamic_corr['Stocks']['Crypto'].max():.3f}")
print(f"Minimum correlation between Bonds and Commodities: {dynamic_corr['Bonds']['Commodities'].min():.3f}")
```

## Utility Functions

### data_quality_score()

Calculate comprehensive data quality scores.

```python
from minglib.market_data import data_quality_score

quality_score = data_quality_score(
    data=price_data,
    checks=["completeness", "accuracy", "consistency", "timeliness"]
)
```

### resample_data()

Resample market data to different frequencies.

```python
from minglib.market_data import resample_data

daily_data = resample_data(
    data=minute_data,
    frequency="1d",
    aggregation_method="ohlc"
)
```

### merge_multiple_sources()

Merge data from multiple sources with conflict resolution.

```python
from minglib.market_data import merge_multiple_sources

merged_data = merge_multiple_sources(
    data_sources=[bloomberg_data, reuters_data, yahoo_data],
    conflict_resolution="weighted_average",
    quality_weights=[0.5, 0.3, 0.2]
)
```

## Performance Considerations

- Use vectorized operations for large datasets
- Implement data caching for frequently accessed historical data
- Consider using HDF5 format for large time series storage
- Use streaming processing for real-time data feeds
- Implement connection pooling for multiple data sources

## Error Handling

```python
try:
    feed = DataFeed(provider="bloomberg", api_key="invalid_key")
    feed.connect()
except ConnectionError as e:
    print(f"Connection failed: {e}")
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except DataProviderError as e:
    print(f"Data provider error: {e}")
```

## Configuration

```python
# Configuration for data providers
config = {
    'bloomberg': {
        'session_options': {
            'serverHost': 'localhost',
            'serverPort': 8194
        }
    },
    'alpha_vantage': {
        'rate_limit': 5,  # calls per minute
        'premium': False
    },
    'cache_settings': {
        'enabled': True,
        'ttl': 3600,  # seconds
        'max_size': '1GB'
    }
}
```

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.19.0
- requests >= 2.25.0
- websockets >= 8.1 (for real-time feeds)
- pytz >= 2021.1 (for timezone handling)
- h5py >= 3.1.0 (for HDF5 storage)

## See Also

- [Data Validation](data_validation.md)
- [Performance Analytics](performance_analytics.md)
- [Risk Management](risk_management.md)
