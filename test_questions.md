# Test Questions

## MingLib RAG Test Questions

These questions are designed to test the RAG system's knowledge of the fictional minglib financial library. Since minglib doesn't exist in the LLM's training data, these questions help demonstrate the value of RAG for proprietary/new documentation.

### Portfolio and Risk Management

**1. How do I calculate portfolio VaR using minglib?**

*Expected Output:*
To calculate portfolio VaR using minglib, use the `calculate_var()` function from the `minglib.risk` module:

```python
from minglib.risk import calculate_var
import numpy as np

# Generate sample returns data
returns = np.random.normal(0.001, 0.02, 1000)

# Calculate 95% VaR using historical simulation
var_result = calculate_var(
    returns=returns,
    confidence_level=0.95,
    method="historical"
)

print(f"95% VaR: {var_result['var_value']:.4f}")
```

The function supports three methods: "historical", "parametric", and "monte_carlo". It returns a dictionary containing the VaR value, method used, confidence level, and sample size.

**Reference:** risk_management.md

---

**2. List all optimization methods in the minglib.portfolio module?**

*Expected Output:*
The minglib.portfolio module provides several optimization methods:

- **mean_variance_optimizer()** - Markowitz mean-variance optimization
- **black_litterman_optimizer()** - Black-Litterman model for incorporating investor views
- **risk_parity_optimizer()** - Risk parity strategies including equal risk and budgeted risk
- **hierarchical_risk_parity()** - HRP algorithm using machine learning clustering
- **robust_optimizer()** - Robust optimization for parameter uncertainty
- **multi_period_optimizer()** - Multi-period optimization with transaction costs

Example usage:
```python
from minglib.portfolio import mean_variance_optimizer

optimal_portfolio = mean_variance_optimizer(
    expected_returns=expected_returns,
    covariance_matrix=cov_matrix,
    risk_aversion=1.0
)
```

**Reference:** portfolio_optimization.md

---

**3. What's risk parity optimization in minglib and How is it implemented?**

*Expected Output:*
Risk parity optimization in minglib is implemented through the `risk_parity_optimizer()` function, which creates portfolios where each asset contributes equally to portfolio risk or according to specified risk budgets.

```python
from minglib.portfolio import risk_parity_optimizer

# Equal Risk Contribution portfolio
rp_portfolio = risk_parity_optimizer(
    covariance_matrix=cov_matrix,
    method="equal_risk"
)

# Budgeted Risk Contribution portfolio
risk_budgets = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
rp_budgeted = risk_parity_optimizer(
    covariance_matrix=cov_matrix,
    risk_budgets=risk_budgets,
    method="budgeted_risk"
)
```

The function supports both "equal_risk" and "budgeted_risk" methods and returns optimal weights, risk contributions, and total portfolio risk.

**Reference:** portfolio_optimization.md

---

**4. What functions are in minglib.risk for stress testing?**

*Expected Output:*
The minglib.risk module provides `stress_test_portfolio()` function for comprehensive stress testing:

```python
from minglib.risk import stress_test_portfolio

# Define stress scenarios
stress_scenarios = {
    'market_crash': {'equity_shock': -0.30, 'vol_shock': 2.0},
    'interest_rate_spike': {'rate_shock': 0.02, 'bond_shock': -0.15},
    'liquidity_crisis': {'bid_ask_widening': 3.0, 'correlation_shock': 0.8}
}

# Run stress tests
stress_results = stress_test_portfolio(
    portfolio=portfolio_dict,
    stress_scenarios=stress_scenarios,
    shock_magnitude=0.025
)
```

The function returns scenario results with portfolio losses, worst-case scenarios, and risk breakdowns by asset class.

**Reference:** risk_management.md

---

**5. What's the difference between minglib.portfolio.mean_variance_optimizer and minglib.portfolio.black_litterman_optimizer?**

*Expected Output:*
**mean_variance_optimizer()** implements the classic Markowitz mean-variance optimization:
- Uses historical expected returns and covariance matrix
- Optimizes for maximum Sharpe ratio or target return/volatility
- Returns optimal weights, expected return, volatility, and Sharpe ratio

**black_litterman_optimizer()** implements the Black-Litterman model:
- Starts with market-implied returns from market cap weights
- Incorporates investor views and confidence levels
- Combines market equilibrium with subjective views
- Returns Black-Litterman adjusted returns, covariance, and optimal weights

Black-Litterman is better for incorporating market views and avoiding extreme allocations, while mean-variance is simpler and more direct.

**Reference:** portfolio_optimization.md

### Options and Fixed Income

**6. How do I price a call option with Black-Scholes in minglib?**

*Expected Output:*
To price a call option using Black-Scholes in minglib, use the `BlackScholesModel` class:

```python
from minglib.options import BlackScholesModel

# Initialize Black-Scholes model
bs_model = BlackScholesModel(
    spot_price=100.0,
    strike_price=105.0,
    time_to_expiry=0.25,  # 3 months
    risk_free_rate=0.05,   # 5% risk-free rate
    volatility=0.20,       # 20% implied volatility
    dividend_yield=0.02    # 2% dividend yield
)

# Calculate call option price
call_price = bs_model.option_price(option_type="call")
print(f"Call Option Price: ${call_price:.4f}")

# Calculate Greeks
call_greeks = bs_model.greeks(option_type="call")
print(f"Delta: {call_greeks['delta']:.4f}")
```

The model also supports put options and calculates all Greeks (delta, gamma, theta, vega, rho).

**Reference:** options_pricing.md

---

**7. How do I build a yield curve using minglib.fixed_income?**

*Expected Output:*
To build a yield curve in minglib, use the `YieldCurveBuilder` class:

```python
from minglib.fixed_income import YieldCurveBuilder
import pandas as pd

# Create market data
market_data = pd.DataFrame({
    'maturity_years': [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
    'yield': [0.015, 0.018, 0.022, 0.025, 0.028, 0.032, 0.035, 0.038, 0.042, 0.045],
    'bond_type': ['Treasury'] * 10
})

# Initialize curve builder
curve_builder = YieldCurveBuilder(
    interpolation_method="cubic_spline",
    currency="USD",
    curve_type="par_yield"
)

# Build yield curve
yield_curve = curve_builder.build_curve(
    market_data=market_data,
    reference_date="2024-01-15",
    bootstrap_method="standard"
)
```

The builder supports bootstrapping zero curves from par yields and can calculate forward rates.

**Reference:** fixed_income.md

### Backtesting and Analytics

**8. How do I use minglib.backtesting for trading strategies?**

*Expected Output:*
To backtest trading strategies with minglib, use the `BacktestEngine` class:

```python
from minglib.backtesting import BacktestEngine

# Initialize backtest engine
engine = BacktestEngine(
    start_date="2020-01-01",
    end_date="2023-12-31",
    initial_capital=1000000.0,
    execution_mode="vectorized",
    benchmark="SPY"
)

# Run backtest
results = engine.run_backtest(
    strategy=your_strategy,
    data=market_data,
    transaction_costs=cost_model,
    risk_manager=risk_mgr
)
```

The engine supports vectorized and event-driven backtesting, transaction cost modeling, parameter optimization, and walk-forward analysis. Results include performance metrics, drawdowns, and benchmark comparisons.

**Reference:** backtesting.md

### Data Management

**9. How do I validate data using minglib.validation.DataValidator?**

*Expected Output:*
To validate data using minglib, use the `DataValidator` class:

```python
from minglib.validation import DataValidator

# Initialize validator
validator = DataValidator(
    error_threshold=0.05,
    auto_correction=True
)

# Define schema
schema = {
    'price': {'type': 'float', 'required': True, 'min_value': 0.01},
    'volume': {'type': 'float', 'required': False, 'min_value': 0}
}

# Validate dataset
results = validator.validate_dataset(
    data=your_dataframe,
    schema=schema,
    business_rules=your_rules,
    cross_validation=True
)
```

The validator returns comprehensive results including error counts, quality scores, and detailed error descriptions. It supports schema validation, business rules, and data reconciliation.

**Reference:** data_validation.md

### Reporting

**10. What reporting capabilities does minglib.reporting provide?**

*Expected Output:*
The minglib.reporting module provides comprehensive report generation capabilities:

**Main Classes:**
- **ReportGenerator** - Multi-format output (PDF, HTML, Excel, PowerPoint)
- **RiskReportBuilder** - VaR reports and stress testing
- **PerformanceReportBuilder** - Performance attribution analysis
- **RegulatoryReportBuilder** - Compliance reports (UCITS, AIFMD, SEC)
- **ClientReportBuilder** - Client-facing reports and factsheets

```python
from minglib.reporting import ReportGenerator

report_gen = ReportGenerator(
    output_formats=["pdf", "html", "excel"],
    template_directory="./templates",
    branding={'company_name': 'ABC Asset Management'}
)

report = report_gen.generate_report(
    report_type="monthly_performance",
    data=portfolio_data,
    output_filename="monthly_report"
)
```

The module supports automated scheduling, template management, and distribution via email or portals.

**Reference:** reporting.md