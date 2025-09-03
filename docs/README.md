# MingLib API Documentation

## Overview

MingLib is a comprehensive Python library for quantitative finance and investment banking applications. It provides a full suite of tools for portfolio optimization, risk management, derivatives pricing, fixed income analysis, and regulatory compliance.

## Module Overview

### [Risk Management](risk_management.md)
Comprehensive risk assessment tools including VaR calculation, stress testing, and portfolio risk analytics.

**Key Features:**
- Value at Risk (Historical, Parametric, Monte Carlo)
- Expected Shortfall calculations
- Stress testing framework
- Risk attribution analysis
- Correlation matrix analysis

**Example Usage:**
```python
from minglib.risk import calculate_var
var_result = calculate_var(returns, confidence_level=0.95, method="historical")
```

### [Portfolio Optimization](portfolio_optimization.md)
Advanced portfolio optimization algorithms for institutional asset management.

**Key Features:**
- Mean-variance optimization (Markowitz)
- Black-Litterman model implementation
- Risk parity strategies
- Hierarchical Risk Parity (HRP)
- Robust optimization techniques
- Multi-period optimization

**Example Usage:**
```python
from minglib.portfolio import mean_variance_optimizer
optimal_portfolio = mean_variance_optimizer(expected_returns, covariance_matrix)
```

### [Market Data Processing](market_data.md)
Real-time and historical market data processing with quality assurance.

**Key Features:**
- Multi-provider data feeds (Bloomberg, Refinitiv, Alpha Vantage)
- Data cleaning and validation
- Volume analysis and market microstructure
- Cross-asset correlation analysis
- Time series resampling

**Example Usage:**
```python
from minglib.market_data import DataFeed
feed = DataFeed(provider="bloomberg", api_key="your_key")
```

### [Options Pricing Models](options_pricing.md)
Comprehensive options pricing framework with support for vanilla and exotic derivatives.

**Key Features:**
- Black-Scholes-Merton model
- Binomial tree models for American options
- Monte Carlo simulation engine
- Implied volatility solver
- Volatility surface construction
- Exotic options calculator

**Example Usage:**
```python
from minglib.options import BlackScholesModel
bs_model = BlackScholesModel(spot=100, strike=105, time=0.25, rate=0.05, vol=0.20)
price = bs_model.option_price("call")
```

### [Fixed Income Calculations](fixed_income.md)
Bond pricing, yield curve construction, and interest rate derivatives.

**Key Features:**
- Bond pricing and yield calculations
- Yield curve bootstrapping
- Duration and convexity analysis
- Interest rate swap valuation
- Credit spread analysis
- Inflation-linked bonds

**Example Usage:**
```python
from minglib.fixed_income import BondPricer
bond = BondPricer(face_value=1000, coupon_rate=0.04, maturity_years=10)
price = bond.price_bond(yield_to_maturity=0.035)
```

### [Credit Risk Assessment](credit_risk.md)
Advanced credit risk modeling for regulatory and economic capital calculations.

**Key Features:**
- Probability of Default (PD) models
- Loss Given Default (LGD) estimation
- Exposure at Default (EAD) calculations
- Portfolio credit risk analytics
- Stress testing engine
- Regulatory capital calculations

**Example Usage:**
```python
from minglib.credit import DefaultProbabilityModel
pd_model = DefaultProbabilityModel(model_type="logistic")
pd_model.fit_model(features, default_indicators)
```

### [Performance Analytics](performance_analytics.md)
Portfolio and fund performance analysis with attribution capabilities.

**Key Features:**
- Return calculations and risk metrics
- Brinson-Fachler attribution analysis
- Multi-benchmark comparisons
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Style analysis

**Example Usage:**
```python
from minglib.performance import PerformanceCalculator
perf_calc = PerformanceCalculator()
summary = perf_calc.performance_summary(returns, benchmark_returns)
```

### [Data Validation](data_validation.md)
Comprehensive data quality assessment and reconciliation tools.

**Key Features:**
- Schema validation
- Business rule validation
- Data quality scoring
- Multi-source reconciliation
- Time series validation
- Anomaly detection

**Example Usage:**
```python
from minglib.validation import DataValidator
validator = DataValidator()
results = validator.validate_dataset(data, schema, business_rules)
```

### [Backtesting Framework](backtesting.md)
Professional backtesting framework for quantitative strategies.

**Key Features:**
- Vectorized and event-driven backtesting
- Transaction cost modeling
- Risk management integration
- Parameter optimization
- Walk-forward analysis
- Multi-strategy backtesting

**Example Usage:**
```python
from minglib.backtesting import BacktestEngine
engine = BacktestEngine(start_date="2020-01-01", end_date="2023-12-31")
results = engine.run_backtest(strategy, data)
```

### [Reporting Generators](reporting.md)
Automated report generation for investment management and compliance.

**Key Features:**
- Multi-format output (PDF, HTML, Excel, PowerPoint)
- Template management system
- Automated scheduling
- Risk reporting
- Performance reporting
- Regulatory compliance reports

**Example Usage:**
```python
from minglib.reporting import ReportGenerator
report_gen = ReportGenerator()
report = report_gen.generate_report("monthly_performance", data)
```

## Installation

```bash
pip install minglib
```

## Quick Start

```python
import minglib
import pandas as pd
import numpy as np

# Example: Basic portfolio optimization
from minglib.portfolio import mean_variance_optimizer
from minglib.risk import calculate_var

# Sample expected returns and covariance matrix
expected_returns = np.array([0.08, 0.12, 0.15])
cov_matrix = np.array([
    [0.04, 0.01, 0.02],
    [0.01, 0.09, 0.03],
    [0.02, 0.03, 0.16]
])

# Optimize portfolio
optimal_portfolio = mean_variance_optimizer(
    expected_returns=expected_returns,
    covariance_matrix=cov_matrix,
    risk_aversion=1.0
)

print(f"Optimal weights: {optimal_portfolio['weights']}")
print(f"Expected return: {optimal_portfolio['expected_return']:.4f}")
print(f"Portfolio volatility: {optimal_portfolio['volatility']:.4f}")
```

## Configuration

MingLib supports various configuration options through environment variables or configuration files:

```python
# Set up configuration
import minglib.config as config

config.set_data_provider("bloomberg")
config.set_risk_free_rate(0.025)
config.set_default_currency("USD")
config.set_parallel_processing(True)
```

## Dependencies

- **Core:** NumPy >= 1.19.0, Pandas >= 1.3.0, SciPy >= 1.7.0
- **Optimization:** CVXPY >= 1.1.0, scikit-learn >= 0.24.0
- **Data:** requests >= 2.25.0, websockets >= 8.1
- **Reporting:** matplotlib >= 3.3.0, jinja2 >= 2.11.0, reportlab >= 3.5.0
- **Optional:** QuantLib >= 1.20 (for advanced derivatives), Bloomberg API

## Support and Documentation

- **API Reference:** Complete function and class documentation
- **Tutorials:** Step-by-step guides for common workflows
- **Examples:** Real-world use cases and implementation patterns
- **Community:** GitHub issues and discussions

## License

MingLib is released under the MIT License. See LICENSE file for details.

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines on:
- Bug reports and feature requests
- Code contributions and pull requests
- Documentation improvements
- Testing and quality assurance

## Changelog

### Version 1.0.0 (Latest)
- Initial release with full module suite
- Comprehensive documentation
- Performance optimizations
- Production-ready APIs

For detailed release notes, see CHANGELOG.md.
