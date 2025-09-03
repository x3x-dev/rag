# Portfolio Construction Tutorial

## Overview

This tutorial demonstrates how to construct and optimize a portfolio using MingLib's portfolio optimization and risk management modules. We'll walk through the complete workflow from data preparation to performance analysis.

## Prerequisites

```python
import minglib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Required modules
from minglib.market_data import HistoricalDataRetriever
from minglib.portfolio import mean_variance_optimizer, black_litterman_optimizer
from minglib.risk import calculate_var, stress_test_portfolio
from minglib.performance import PerformanceCalculator
from minglib.backtesting import BacktestEngine
```

## Step 1: Data Collection

First, let's collect historical market data for our universe of assets:

```python
# Initialize data retriever
data_retriever = HistoricalDataRetriever(
    provider="yfinance",  # Using Yahoo Finance for this example
    cache_data=True
)

# Define our investment universe
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'PG', 'XOM', 'BRK-B']
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)  # 3 years of data

# Retrieve historical price data
price_data = data_retriever.get_price_data(
    symbols=symbols,
    start_date=start_date.strftime("%Y-%m-%d"),
    end_date=end_date.strftime("%Y-%m-%d"),
    frequency="1d",
    adjusted=True
)

# Calculate returns
returns = price_data.pct_change().dropna()

print(f"Data retrieved for {len(symbols)} assets")
print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
print(f"Total observations: {len(returns)}")
```

## Step 2: Expected Returns Estimation

Calculate expected returns using multiple methods:

```python
# Method 1: Historical mean returns
historical_returns = returns.mean() * 252  # Annualized

# Method 2: CAPM-based expected returns
market_returns = returns.mean(axis=1)  # Equal-weighted market proxy
risk_free_rate = 0.025  # 2.5% risk-free rate

# Calculate beta for each asset
betas = {}
for symbol in symbols:
    covariance = np.cov(returns[symbol].dropna(), market_returns)[0, 1]
    market_variance = np.var(market_returns)
    betas[symbol] = covariance / market_variance

# CAPM expected returns
market_premium = historical_returns.mean() - risk_free_rate
capm_returns = pd.Series({
    symbol: risk_free_rate + beta * market_premium 
    for symbol, beta in betas.items()
}, index=symbols)

# Method 3: Shrinkage estimator (James-Stein)
def shrinkage_estimator(returns, shrinkage_factor=0.3):
    """Apply shrinkage to expected returns"""
    grand_mean = returns.mean().mean()
    shrunk_returns = (1 - shrinkage_factor) * returns.mean() + shrinkage_factor * grand_mean
    return shrunk_returns * 252

shrunk_returns = shrinkage_estimator(returns)

print("Expected Returns Comparison:")
print("-" * 50)
comparison_df = pd.DataFrame({
    'Historical': historical_returns,
    'CAPM': camp_returns,
    'Shrinkage': shrunk_returns
})
print(comparison_df.round(4))
```

## Step 3: Covariance Matrix Estimation

Estimate the covariance matrix with various approaches:

```python
# Method 1: Sample covariance matrix
sample_cov = returns.cov() * 252  # Annualized

# Method 2: Exponentially weighted moving average
def ewma_covariance(returns, lambda_decay=0.94):
    """Calculate EWMA covariance matrix"""
    # Implementation simplified for tutorial
    weights = np.array([(1 - lambda_decay) * lambda_decay**i 
                       for i in range(len(returns))][::-1])
    weights = weights / weights.sum()
    
    centered_returns = returns - returns.mean()
    weighted_cov = np.cov(centered_returns.T, aweights=weights)
    return pd.DataFrame(weighted_cov, index=returns.columns, columns=returns.columns) * 252

ewma_cov = ewma_covariance(returns)

# Method 3: Ledoit-Wolf shrinkage
from sklearn.covariance import LedoitWolf

lw = LedoitWolf()
lw_cov_array = lw.fit(returns.fillna(0)).covariance_ * 252
lw_cov = pd.DataFrame(lw_cov_array, index=returns.columns, columns=returns.columns)

print(f"Covariance Matrix Comparison:")
print(f"Sample covariance condition number: {np.linalg.cond(sample_cov):.2f}")
print(f"EWMA covariance condition number: {np.linalg.cond(ewma_cov):.2f}")
print(f"Ledoit-Wolf covariance condition number: {np.linalg.cond(lw_cov):.2f}")
```

## Step 4: Portfolio Optimization

Now let's optimize portfolios using different approaches:

```python
# Optimization 1: Mean-Variance (Markowitz)
mv_portfolio = mean_variance_optimizer(
    expected_returns=shrunk_returns.values,
    covariance_matrix=lw_cov.values,
    risk_aversion=1.0,
    weight_bounds=(0.0, 0.15),  # Max 15% per position
    constraints=[
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
    ]
)

# Optimization 2: Risk Parity
from minglib.portfolio import risk_parity_optimizer

rp_portfolio = risk_parity_optimizer(
    covariance_matrix=lw_cov.values,
    method="equal_risk"
)

# Optimization 3: Black-Litterman with views
# Define investor views
views_matrix = np.array([
    [1, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # AAPL outperforms MSFT
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]    # Bullish on TSLA
])
view_returns = np.array([0.03, 0.20])  # 3% and 20% expected outperformance
view_confidences = np.array([0.7, 0.5])  # Confidence levels

# Market cap weights for Black-Litterman
market_caps = np.array([2.5e12, 1.8e12, 1.2e12, 1.0e12, 0.8e12, 
                       0.4e12, 0.5e12, 0.4e12, 0.3e12, 0.6e12])  # Sample market caps

bl_portfolio = black_litterman_optimizer(
    market_caps=market_caps,
    price_data=price_data.values,
    views_matrix=views_matrix,
    view_returns=view_returns,
    view_confidences=view_confidences,
    risk_aversion=2.0
)

# Compare portfolio weights
portfolio_comparison = pd.DataFrame({
    'Mean-Variance': mv_portfolio['weights'],
    'Risk Parity': rp_portfolio['weights'],
    'Black-Litterman': bl_portfolio['optimal_weights']
}, index=symbols)

print("Portfolio Weights Comparison:")
print("=" * 50)
print(portfolio_comparison.round(4))

# Portfolio characteristics
print(f"\nPortfolio Characteristics:")
print(f"Mean-Variance - Expected Return: {mv_portfolio['expected_return']:.4f}, "
      f"Volatility: {mv_portfolio['volatility']:.4f}")
print(f"Risk Parity - Total Risk: {rp_portfolio['total_risk']:.4f}")
```

## Step 5: Risk Analysis

Analyze the risk characteristics of our optimized portfolios:

```python
# Calculate VaR for each portfolio
portfolios = {
    'Mean-Variance': mv_portfolio['weights'],
    'Risk Parity': rp_portfolio['weights'],
    'Black-Litterman': bl_portfolio['optimal_weights']
}

risk_analysis = {}

for name, weights in portfolios.items():
    # Portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # VaR calculation
    var_result = calculate_var(
        returns=portfolio_returns.values,
        confidence_level=0.95,
        method="historical"
    )
    
    # Stress testing
    stress_scenarios = {
        'market_crash': {'equity_shock': -0.30},
        'interest_rate_spike': {'rate_shock': 0.02},
        'volatility_spike': {'vol_shock': 2.0}
    }
    
    # Create portfolio dictionary for stress testing
    portfolio_dict = {
        'positions': {symbol: {'weight': weight} for symbol, weight in zip(symbols, weights)},
        'total_value': 1000000  # $1M portfolio
    }
    
    stress_result = stress_test_portfolio(
        portfolio=portfolio_dict,
        stress_scenarios=stress_scenarios
    )
    
    risk_analysis[name] = {
        'var_95': var_result['var_value'],
        'volatility': portfolio_returns.std() * np.sqrt(252),
        'worst_stress_loss': stress_result['worst_case_loss']
    }

print("Risk Analysis Summary:")
print("=" * 40)
risk_df = pd.DataFrame(risk_analysis).T
print(risk_df.round(4))
```

## Step 6: Backtesting

Test our portfolios using historical simulation:

```python
# Set up backtesting engine
backtest_engine = BacktestEngine(
    start_date=(start_date + timedelta(days=365)).strftime("%Y-%m-%d"),  # Use last 2 years
    end_date=end_date.strftime("%Y-%m-%d"),
    initial_capital=1000000.0,
    execution_mode="vectorized"
)

# Define a simple rebalancing strategy
class MonthlyRebalanceStrategy:
    def __init__(self, target_weights, name):
        self.target_weights = target_weights
        self.name = name
    
    def generate_signals(self, data):
        # Rebalance monthly to target weights
        signals = pd.DataFrame(index=data.index, columns=symbols, data=0.0)
        
        # Monthly rebalancing dates
        monthly_dates = pd.date_range(
            start=data.index[0], 
            end=data.index[-1], 
            freq='MS'  # Month start
        )
        
        for date in monthly_dates:
            if date in signals.index:
                signals.loc[date] = self.target_weights
        
        return signals.ffill()  # Forward fill between rebalancing dates

# Create strategy instances
strategies = {
    'Mean-Variance': MonthlyRebalanceStrategy(mv_portfolio['weights'], 'Mean-Variance'),
    'Risk Parity': MonthlyRebalanceStrategy(rp_portfolio['weights'], 'Risk Parity'),
    'Black-Litterman': MonthlyRebalanceStrategy(bl_portfolio['optimal_weights'], 'Black-Litterman')
}

# Run backtests
backtest_results = {}
for name, strategy in strategies.items():
    results = backtest_engine.run_backtest(
        strategy=strategy,
        data=price_data
    )
    backtest_results[name] = results

# Performance comparison
print("Backtest Performance Summary:")
print("=" * 50)
performance_summary = pd.DataFrame({
    name: {
        'Total Return': results['total_return'],
        'Annual Return': results['annualized_return'],
        'Volatility': results['annualized_volatility'],
        'Sharpe Ratio': results['sharpe_ratio'],
        'Max Drawdown': results['max_drawdown']
    }
    for name, results in backtest_results.items()
}).T

print(performance_summary.round(4))
```

## Step 7: Performance Attribution

Analyze what drove performance differences:

```python
from minglib.performance import AttributionAnalyzer

# Sector mapping (simplified)
sector_mapping = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
    'JPM': 'Financials', 'JNJ': 'Healthcare', 'PG': 'Consumer Staples',
    'XOM': 'Energy', 'BRK-B': 'Financials'
}

# Create sector allocation data
def create_sector_allocation(weights, sector_mapping):
    sector_weights = {}
    for symbol, weight in zip(symbols, weights):
        sector = sector_mapping[symbol]
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    return sector_weights

# Benchmark (equal weight)
benchmark_weights = np.array([1/len(symbols)] * len(symbols))

attribution_analyzer = AttributionAnalyzer()

for name, weights in portfolios.items():
    portfolio_sectors = create_sector_allocation(weights, sector_mapping)
    benchmark_sectors = create_sector_allocation(benchmark_weights, sector_mapping)
    
    print(f"\n{name} Sector Allocation vs Benchmark:")
    for sector in set(sector_mapping.values()):
        port_weight = portfolio_sectors.get(sector, 0)
        bench_weight = benchmark_sectors.get(sector, 0)
        active_weight = port_weight - bench_weight
        print(f"  {sector}: {port_weight:.3f} vs {bench_weight:.3f} (Active: {active_weight:+.3f})")
```

## Step 8: Reporting

Generate a comprehensive report:

```python
from minglib.reporting import ReportGenerator

# Prepare report data
report_data = {
    'analysis_date': datetime.now().strftime("%Y-%m-%d"),
    'investment_universe': symbols,
    'optimization_methods': list(portfolios.keys()),
    'portfolio_weights': portfolio_comparison.to_dict(),
    'risk_metrics': risk_analysis,
    'backtest_performance': {name: results for name, results in backtest_results.items()},
    'recommendations': {
        'preferred_strategy': max(backtest_results.keys(), 
                                key=lambda x: backtest_results[x]['sharpe_ratio']),
        'risk_considerations': [
            'Market volatility remains elevated',
            'Consider reducing concentration risk',
            'Monitor correlation breakdown in stress scenarios'
        ]
    }
}

# Generate report
report_gen = ReportGenerator(
    output_formats=['html', 'pdf'],
    branding={'company_name': 'Asset Management Firm'}
)

report = report_gen.generate_report(
    report_type='portfolio_analysis',
    data=report_data,
    output_filename='portfolio_construction_analysis'
)

print(f"\nReport generated: {report['output_files']}")
```

## Key Takeaways

1. **Data Quality Matters**: Always validate and clean your input data
2. **Multiple Approaches**: Different optimization methods yield different results
3. **Risk Management**: Consider multiple risk metrics beyond just volatility
4. **Backtesting**: Historical simulation helps validate theoretical results
5. **Regular Review**: Portfolio optimization is an ongoing process

## Next Steps

- Explore transaction cost modeling
- Implement dynamic rebalancing strategies
- Add factor exposure constraints
- Consider ESG scoring integration
- Develop custom risk models

This tutorial provides a comprehensive framework for portfolio construction using MingLib. Adapt the code to your specific requirements and investment constraints.
