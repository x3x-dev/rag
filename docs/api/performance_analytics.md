# Performance Analytics Module

## Overview

The `minglib.performance` module provides comprehensive portfolio and fund performance analysis tools including return calculations, risk-adjusted metrics, attribution analysis, and benchmark comparisons used by institutional asset managers and performance analysts.

## Installation

```python
from minglib.performance import (
    PerformanceCalculator,
    AttributionAnalyzer,
    BenchmarkComparison,
    RiskAdjustedMetrics,
    DrawdownAnalyzer,
    FactorPerformanceAnalysis
)
```

## Core Classes

### PerformanceCalculator

Calculate various performance metrics for portfolios and individual securities.

#### Syntax

```python
class PerformanceCalculator:
    def __init__(
        self,
        return_frequency: str = "daily",
        annualization_factor: int = 252,
        risk_free_rate: float = 0.02
    )
```

#### Parameters

- **return_frequency** (str, optional): Frequency of return data. Options: "daily", "monthly", "quarterly". Default: "daily"
- **annualization_factor** (int, optional): Periods per year for annualization. Default: 252
- **risk_free_rate** (float, optional): Annual risk-free rate. Default: 0.02

#### Methods

##### calculate_returns()

Calculate portfolio returns from price or NAV data.

```python
def calculate_returns(
    self,
    price_data: pd.DataFrame,
    method: str = "simple",
    periods: int = 1
) -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Parameters:
        price_data (pd.DataFrame): Price or NAV time series
        method (str): Return calculation method ("simple", "log", "total")
        periods (int): Periods for return calculation
    
    Returns:
        pd.DataFrame: Calculated returns
    """
```

##### performance_summary()

Generate comprehensive performance summary statistics.

```python
def performance_summary(
    self,
    returns: pd.Series,
    benchmark_returns: pd.Series = None,
    inception_date: str = None
) -> dict:
    """
    Calculate comprehensive performance metrics.
    
    Parameters:
        returns (pd.Series): Portfolio returns time series
        benchmark_returns (pd.Series): Benchmark returns for comparison
        inception_date (str): Portfolio inception date
    
    Returns:
        dict: Performance summary with all key metrics
    """
```

#### Example

```python
from minglib.performance import PerformanceCalculator
import pandas as pd
import numpy as np

# Generate sample portfolio and benchmark data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')

# Portfolio returns (slightly higher mean, higher volatility)
portfolio_returns = pd.Series(
    np.random.normal(0.0008, 0.015, len(dates)),
    index=dates,
    name='Portfolio'
)

# Benchmark returns (market index)
benchmark_returns = pd.Series(
    np.random.normal(0.0006, 0.012, len(dates)),
    index=dates,
    name='Benchmark'
)

# Add some correlation between portfolio and benchmark
correlation_factor = 0.7
benchmark_returns = (correlation_factor * portfolio_returns + 
                    (1 - correlation_factor) * benchmark_returns)

# Initialize performance calculator
perf_calc = PerformanceCalculator(
    return_frequency="daily",
    annualization_factor=252,
    risk_free_rate=0.025
)

# Calculate performance summary
performance_summary = perf_calc.performance_summary(
    returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    inception_date='2020-01-01'
)

print("Portfolio Performance Summary (2020-2023):")
print("=" * 50)
print(f"Total Return: {performance_summary['total_return']:.2%}")
print(f"Annualized Return: {performance_summary['annualized_return']:.2%}")
print(f"Annualized Volatility: {performance_summary['annualized_volatility']:.2%}")
print(f"Sharpe Ratio: {performance_summary['sharpe_ratio']:.4f}")
print(f"Information Ratio: {performance_summary['information_ratio']:.4f}")
print(f"Maximum Drawdown: {performance_summary['max_drawdown']:.2%}")
print(f"Calmar Ratio: {performance_summary['calmar_ratio']:.4f}")

print(f"\nBenchmark Comparison:")
print(f"Benchmark Total Return: {performance_summary['benchmark_total_return']:.2%}")
print(f"Active Return: {performance_summary['active_return']:.2%}")
print(f"Tracking Error: {performance_summary['tracking_error']:.2%}")
print(f"Beta: {performance_summary['beta']:.4f}")
print(f"Alpha: {performance_summary['alpha']:.4f}")

print(f"\nRisk Metrics:")
print(f"VaR (95%): {performance_summary['var_95']:.2%}")
print(f"CVaR (95%): {performance_summary['cvar_95']:.2%}")
print(f"Skewness: {performance_summary['skewness']:.4f}")
print(f"Kurtosis: {performance_summary['kurtosis']:.4f}")

# Rolling performance analysis
rolling_metrics = perf_calc.rolling_performance(
    returns=portfolio_returns,
    window=252,  # 1-year rolling windows
    metrics=['sharpe_ratio', 'volatility', 'max_drawdown']
)

print(f"\nRolling Performance (1-Year Windows):")
print(f"Average Rolling Sharpe: {rolling_metrics['sharpe_ratio'].mean():.4f}")
print(f"Rolling Sharpe Std: {rolling_metrics['sharpe_ratio'].std():.4f}")
print(f"Best 1Y Sharpe: {rolling_metrics['sharpe_ratio'].max():.4f}")
print(f"Worst 1Y Sharpe: {rolling_metrics['sharpe_ratio'].min():.4f}")
```

### AttributionAnalyzer

Perform detailed performance attribution analysis including Brinson-Fachler attribution.

#### Syntax

```python
class AttributionAnalyzer:
    def __init__(
        self,
        attribution_method: str = "brinson_fachler",
        currency: str = "USD"
    )
```

#### Methods

##### sector_attribution()

Analyze performance attribution by sector or asset class.

```python
def sector_attribution(
    self,
    portfolio_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    sector_returns: pd.DataFrame,
    rebalancing_frequency: str = "monthly"
) -> dict:
    """
    Calculate sector-based performance attribution.
    
    Parameters:
        portfolio_weights (pd.DataFrame): Portfolio weights by sector over time
        benchmark_weights (pd.DataFrame): Benchmark weights by sector
        sector_returns (pd.DataFrame): Sector returns
        rebalancing_frequency (str): Portfolio rebalancing frequency
    
    Returns:
        dict: Attribution analysis results
    """
```

##### security_attribution()

Perform security-level attribution analysis.

```python
def security_attribution(
    self,
    portfolio_data: pd.DataFrame,
    benchmark_data: pd.DataFrame,
    return_data: pd.DataFrame
) -> dict:
    """
    Calculate security-level performance attribution.
    
    Parameters:
        portfolio_data (pd.DataFrame): Portfolio holdings data
        benchmark_data (pd.DataFrame): Benchmark composition
        return_data (pd.DataFrame): Security returns
    
    Returns:
        dict: Security attribution results
    """
```

#### Example

```python
from minglib.performance import AttributionAnalyzer

# Sample portfolio and benchmark sector allocations
sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer_Goods', 'Energy']
dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')

# Portfolio weights (overweight tech, underweight energy)
portfolio_weights = pd.DataFrame({
    'Technology': [0.35, 0.36, 0.34, 0.37, 0.35, 0.38, 0.36, 0.35, 0.37, 0.36, 0.35, 0.36],
    'Healthcare': [0.18, 0.17, 0.19, 0.18, 0.19, 0.17, 0.18, 0.19, 0.18, 0.17, 0.18, 0.19],
    'Financials': [0.22, 0.23, 0.21, 0.22, 0.21, 0.23, 0.22, 0.21, 0.22, 0.23, 0.22, 0.21],
    'Consumer_Goods': [0.20, 0.19, 0.21, 0.18, 0.20, 0.18, 0.19, 0.20, 0.18, 0.19, 0.20, 0.19],
    'Energy': [0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
}, index=dates)

# Benchmark weights (market cap weighted)
benchmark_weights = pd.DataFrame({
    'Technology': [0.28] * 12,
    'Healthcare': [0.16] * 12,
    'Financials': [0.24] * 12,
    'Consumer_Goods': [0.22] * 12,
    'Energy': [0.10] * 12
}, index=dates)

# Sector returns (technology outperformed, energy underperformed)
np.random.seed(42)
sector_returns = pd.DataFrame({
    'Technology': np.random.normal(0.015, 0.06, 12),
    'Healthcare': np.random.normal(0.008, 0.04, 12),
    'Financials': np.random.normal(0.010, 0.05, 12),
    'Consumer_Goods': np.random.normal(0.007, 0.04, 12),
    'Energy': np.random.normal(-0.005, 0.08, 12)
}, index=dates)

# Initialize attribution analyzer
attribution = AttributionAnalyzer(
    attribution_method="brinson_fachler",
    currency="USD"
)

# Perform sector attribution
sector_attr = attribution.sector_attribution(
    portfolio_weights=portfolio_weights,
    benchmark_weights=benchmark_weights,
    sector_returns=sector_returns,
    rebalancing_frequency="monthly"
)

print("Sector Attribution Analysis (2023):")
print("=" * 60)
print("Sector\t\tAllocation\tSelection\tInteraction\tTotal")
print("-" * 60)

total_attribution = 0
for sector in sectors:
    allocation = sector_attr['allocation_effect'][sector]
    selection = sector_attr['selection_effect'][sector]  
    interaction = sector_attr['interaction_effect'][sector]
    total = allocation + selection + interaction
    total_attribution += total
    
    print(f"{sector:15s}\t{allocation:8.2%}\t{selection:8.2%}\t{interaction:8.2%}\t{total:8.2%}")

print("-" * 60)
print(f"{'Total Attribution':15s}\t\t\t\t\t{total_attribution:8.2%}")

print(f"\nAttribution Summary:")
print(f"Total Allocation Effect: {sector_attr['total_allocation_effect']:.2%}")
print(f"Total Selection Effect: {sector_attr['total_selection_effect']:.2%}")
print(f"Total Interaction Effect: {sector_attr['total_interaction_effect']:.2%}")
print(f"Portfolio Return: {sector_attr['portfolio_return']:.2%}")
print(f"Benchmark Return: {sector_attr['benchmark_return']:.2%}")
print(f"Active Return: {sector_attr['active_return']:.2%}")
```

### BenchmarkComparison

Compare portfolio performance against multiple benchmarks and peer groups.

#### Syntax

```python
class BenchmarkComparison:
    def __init__(
        self,
        comparison_metrics: list = ["return", "volatility", "sharpe", "max_drawdown"],
        significance_level: float = 0.05
    )
```

#### Methods

##### multi_benchmark_analysis()

Compare performance against multiple benchmarks.

```python
def multi_benchmark_analysis(
    self,
    portfolio_returns: pd.Series,
    benchmark_returns: dict,
    analysis_periods: list = ["1Y", "3Y", "5Y", "ITD"]
) -> dict:
    """
    Compare portfolio against multiple benchmarks.
    
    Parameters:
        portfolio_returns (pd.Series): Portfolio return time series
        benchmark_returns (dict): Dictionary of benchmark return series
        analysis_periods (list): Analysis time periods
    
    Returns:
        dict: Multi-benchmark comparison results
    """
```

#### Example

```python
from minglib.performance import BenchmarkComparison

# Define multiple benchmarks
benchmarks = {
    'S&P 500': pd.Series(np.random.normal(0.0008, 0.012, len(dates)), index=dates),
    'Russell 2000': pd.Series(np.random.normal(0.0006, 0.018, len(dates)), index=dates),
    'MSCI World': pd.Series(np.random.normal(0.0007, 0.014, len(dates)), index=dates),
    'Peer Average': pd.Series(np.random.normal(0.0005, 0.013, len(dates)), index=dates)
}

# Initialize benchmark comparison
benchmark_comp = BenchmarkComparison(
    comparison_metrics=["return", "volatility", "sharpe", "max_drawdown", "alpha"],
    significance_level=0.05
)

# Perform multi-benchmark analysis
multi_bench_results = benchmark_comp.multi_benchmark_analysis(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmarks,
    analysis_periods=["1Y", "2Y", "3Y", "ITD"]
)

print("Multi-Benchmark Comparison:")
print("=" * 80)

for period in ["1Y", "2Y", "3Y", "ITD"]:
    print(f"\n{period} Performance:")
    print("Benchmark\t\tReturn\tVol\tSharpe\tMax DD\tAlpha")
    print("-" * 65)
    
    portfolio_metrics = multi_bench_results[period]['portfolio']
    print(f"{'Portfolio':15s}\t{portfolio_metrics['return']:6.1%}\t{portfolio_metrics['volatility']:5.1%}\t{portfolio_metrics['sharpe']:6.2f}\t{portfolio_metrics['max_drawdown']:6.1%}\t{'N/A':>6s}")
    
    for bench_name, bench_metrics in multi_bench_results[period]['benchmarks'].items():
        alpha = multi_bench_results[period]['alphas'][bench_name]
        print(f"{bench_name:15s}\t{bench_metrics['return']:6.1%}\t{bench_metrics['volatility']:5.1%}\t{bench_metrics['sharpe']:6.2f}\t{bench_metrics['max_drawdown']:6.1%}\t{alpha:6.2%}")

# Rank analysis
rankings = benchmark_comp.percentile_rankings(
    portfolio_returns=portfolio_returns,
    peer_universe=benchmarks,
    ranking_periods=["1Y", "3Y", "ITD"]
)

print(f"\nPercentile Rankings:")
for period, ranking in rankings.items():
    print(f"{period}: {ranking:.0f}th percentile")
```

### RiskAdjustedMetrics

Calculate advanced risk-adjusted performance metrics.

#### Example

```python
from minglib.performance import RiskAdjustedMetrics

risk_metrics = RiskAdjustedMetrics(
    risk_free_rate=0.025,
    confidence_levels=[0.95, 0.99]
)

# Calculate comprehensive risk-adjusted metrics
advanced_metrics = risk_metrics.calculate_metrics(
    returns=portfolio_returns,
    benchmark_returns=benchmark_returns['S&P 500']
)

print("Advanced Risk-Adjusted Metrics:")
print(f"Treynor Ratio: {advanced_metrics['treynor_ratio']:.4f}")
print(f"Jensen's Alpha: {advanced_metrics['jensens_alpha']:.4f}")
print(f"Modigliani M²: {advanced_metrics['m_squared']:.4f}")
print(f"Sortino Ratio: {advanced_metrics['sortino_ratio']:.4f}")
print(f"Omega Ratio: {advanced_metrics['omega_ratio']:.4f}")
print(f"Ulcer Index: {advanced_metrics['ulcer_index']:.4f}")
print(f"Sterling Ratio: {advanced_metrics['sterling_ratio']:.4f}")
```

### DrawdownAnalyzer

Comprehensive drawdown analysis and recovery statistics.

#### Example

```python
from minglib.performance import DrawdownAnalyzer

dd_analyzer = DrawdownAnalyzer()

drawdown_analysis = dd_analyzer.analyze_drawdowns(
    returns=portfolio_returns,
    min_drawdown_threshold=-0.02  # Minimum 2% drawdown
)

print("Drawdown Analysis:")
print(f"Maximum Drawdown: {drawdown_analysis['max_drawdown']:.2%}")
print(f"Average Drawdown: {drawdown_analysis['avg_drawdown']:.2%}")
print(f"Number of Drawdowns > 5%: {drawdown_analysis['num_significant_drawdowns']}")
print(f"Average Recovery Time: {drawdown_analysis['avg_recovery_days']:.0f} days")
print(f"Longest Recovery Time: {drawdown_analysis['max_recovery_days']:.0f} days")
print(f"Drawdown Frequency: {drawdown_analysis['drawdown_frequency']:.1f} per year")

# Top 5 worst drawdowns
print(f"\nTop 5 Worst Drawdowns:")
for i, dd in enumerate(drawdown_analysis['worst_drawdowns'][:5], 1):
    print(f"{i}. {dd['magnitude']:.2%} from {dd['start_date']} to {dd['end_date']} (Recovery: {dd['recovery_days']} days)")
```

## Utility Functions

### performance_attribution_waterfall()

Create waterfall chart data for attribution analysis.

```python
from minglib.performance import performance_attribution_waterfall

waterfall_data = performance_attribution_waterfall(
    attribution_results=sector_attr,
    chart_type="sector_attribution"
)
```

### rolling_correlation()

Calculate rolling correlations between portfolio and benchmarks.

```python
from minglib.performance import rolling_correlation

rolling_corr = rolling_correlation(
    returns1=portfolio_returns,
    returns2=benchmark_returns['S&P 500'],
    window=252
)
```

### style_analysis()

Perform returns-based style analysis (Sharpe style analysis).

```python
from minglib.performance import style_analysis

style_results = style_analysis(
    portfolio_returns=portfolio_returns,
    factor_returns=benchmark_returns,
    analysis_window=252
)
```

## Performance Considerations

- Use vectorized operations for large time series calculations
- Implement caching for frequently accessed benchmark data
- Consider using approximate methods for real-time performance monitoring
- Use parallel processing for multi-portfolio analysis
- Optimize memory usage when analyzing long historical periods

## Error Handling

```python
try:
    performance_metrics = perf_calc.performance_summary(returns)
except ValueError as e:
    print(f"Invalid return data: {e}")
except InsufficientDataError as e:
    print(f"Insufficient data for calculation: {e}")
except BenchmarkMismatchError as e:
    print(f"Portfolio and benchmark periods don't match: {e}")
```

## Dependencies

- NumPy >= 1.19.0
- Pandas >= 1.3.0
- SciPy >= 1.7.0
- matplotlib >= 3.3.0 (for visualization)
- seaborn >= 0.11.0 (for enhanced visualization)

## See Also

- [Portfolio Optimization](portfolio_optimization.md)
- [Risk Management](risk_management.md)
- [Backtesting Framework](backtesting.md)
