# Risk Management Module

## Overview

The `minglib.risk` module provides comprehensive risk management tools for quantitative finance applications. This module includes functions for calculating Value at Risk (VaR), Expected Shortfall, stress testing, and portfolio risk metrics.

## Installation

```python
from minglib.risk import (
    calculate_var,
    calculate_expected_shortfall,
    monte_carlo_simulation,
    stress_test_portfolio,
    risk_attribution,
    correlation_matrix_analysis
)
```

## Functions

### calculate_var()

Calculate Value at Risk using historical simulation, parametric, or Monte Carlo methods.

#### Syntax

```python
calculate_var(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    method: str = "historical",
    window_size: int = 252,
    distribution: str = "normal"
) -> dict
```

#### Parameters

- **returns** (np.ndarray): Array of historical returns
- **confidence_level** (float, optional): Confidence level for VaR calculation. Default: 0.95
- **method** (str, optional): Calculation method. Options: "historical", "parametric", "monte_carlo". Default: "historical"
- **window_size** (int, optional): Rolling window size for calculations. Default: 252
- **distribution** (str, optional): Distribution assumption for parametric method. Default: "normal"

#### Returns

- **dict**: Dictionary containing VaR values and metadata
  - `var_value`: The calculated VaR
  - `method_used`: Method used for calculation
  - `confidence_level`: Confidence level used
  - `sample_size`: Number of observations used

#### Example

```python
import numpy as np
from minglib.risk import calculate_var

# Generate sample returns data
returns = np.random.normal(0.001, 0.02, 1000)

# Calculate 95% VaR using historical simulation
var_result = calculate_var(
    returns=returns,
    confidence_level=0.95,
    method="historical"
)

print(f"95% VaR: {var_result['var_value']:.4f}")
print(f"Method: {var_result['method_used']}")

# Calculate 99% VaR using parametric method
var_99 = calculate_var(
    returns=returns,
    confidence_level=0.99,
    method="parametric",
    distribution="t"
)

print(f"99% VaR (t-distribution): {var_99['var_value']:.4f}")
```

#### Error Handling

```python
try:
    var_result = calculate_var(returns, confidence_level=0.95)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Calculation error: {e}")
```

### calculate_expected_shortfall()

Calculate Expected Shortfall (Conditional VaR) for portfolio risk assessment.

#### Syntax

```python
calculate_expected_shortfall(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    method: str = "historical"
) -> dict
```

#### Parameters

- **returns** (np.ndarray): Array of historical returns
- **confidence_level** (float, optional): Confidence level for ES calculation. Default: 0.95
- **method** (str, optional): Calculation method. Options: "historical", "parametric". Default: "historical"

#### Returns

- **dict**: Dictionary containing Expected Shortfall results
  - `es_value`: The calculated Expected Shortfall
  - `var_value`: Corresponding VaR value
  - `tail_observations`: Number of tail observations used

#### Example

```python
# Calculate Expected Shortfall
es_result = calculate_expected_shortfall(
    returns=returns,
    confidence_level=0.95
)

print(f"Expected Shortfall: {es_result['es_value']:.4f}")
print(f"Corresponding VaR: {es_result['var_value']:.4f}")
print(f"Tail observations: {es_result['tail_observations']}")
```

### monte_carlo_simulation()

Perform Monte Carlo simulation for portfolio risk assessment.

#### Syntax

```python
monte_carlo_simulation(
    portfolio_weights: np.ndarray,
    asset_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    num_simulations: int = 10000,
    time_horizon: int = 1,
    random_seed: int = None
) -> dict
```

#### Parameters

- **portfolio_weights** (np.ndarray): Portfolio weights for each asset
- **asset_returns** (np.ndarray): Expected returns for each asset
- **covariance_matrix** (np.ndarray): Covariance matrix of asset returns
- **num_simulations** (int, optional): Number of Monte Carlo simulations. Default: 10000
- **time_horizon** (int, optional): Time horizon in days. Default: 1
- **random_seed** (int, optional): Random seed for reproducibility. Default: None

#### Returns

- **dict**: Simulation results
  - `simulated_returns`: Array of simulated portfolio returns
  - `var_95`: 95% VaR from simulation
  - `var_99`: 99% VaR from simulation
  - `expected_return`: Expected portfolio return
  - `volatility`: Portfolio volatility

#### Example

```python
import numpy as np

# Portfolio setup
weights = np.array([0.4, 0.3, 0.3])
expected_returns = np.array([0.08, 0.12, 0.15])
cov_matrix = np.array([
    [0.04, 0.01, 0.02],
    [0.01, 0.09, 0.03],
    [0.02, 0.03, 0.16]
])

# Run Monte Carlo simulation
mc_results = monte_carlo_simulation(
    portfolio_weights=weights,
    asset_returns=expected_returns,
    covariance_matrix=cov_matrix,
    num_simulations=50000,
    time_horizon=10,
    random_seed=42
)

print(f"Expected Return: {mc_results['expected_return']:.4f}")
print(f"Portfolio Volatility: {mc_results['volatility']:.4f}")
print(f"95% VaR (10-day): {mc_results['var_95']:.4f}")
print(f"99% VaR (10-day): {mc_results['var_99']:.4f}")
```

### stress_test_portfolio()

Perform stress testing on portfolio under various market scenarios.

#### Syntax

```python
stress_test_portfolio(
    portfolio: dict,
    stress_scenarios: dict,
    shock_magnitude: float = 0.02,
    correlation_shock: bool = True
) -> dict
```

#### Parameters

- **portfolio** (dict): Portfolio configuration with positions and weights
- **stress_scenarios** (dict): Dictionary of stress scenarios to test
- **shock_magnitude** (float, optional): Magnitude of market shocks. Default: 0.02
- **correlation_shock** (bool, optional): Whether to include correlation breakdown scenarios. Default: True

#### Returns

- **dict**: Stress test results for each scenario
  - `scenario_results`: Results for each stress scenario
  - `worst_case_loss`: Maximum potential loss across all scenarios
  - `scenario_ranking`: Scenarios ranked by impact

#### Example

```python
# Define portfolio
portfolio = {
    'positions': {
        'AAPL': {'quantity': 1000, 'current_price': 150.0},
        'GOOGL': {'quantity': 500, 'current_price': 2800.0},
        'TSLA': {'quantity': 200, 'current_price': 800.0}
    },
    'total_value': 1850000
}

# Define stress scenarios
scenarios = {
    'market_crash': {'equity_shock': -0.30, 'vol_shock': 2.0},
    'interest_rate_spike': {'rate_shock': 0.02, 'bond_shock': -0.15},
    'tech_selloff': {'tech_shock': -0.40, 'broad_market': -0.10},
    'liquidity_crisis': {'bid_ask_widening': 3.0, 'correlation_shock': 0.8}
}

# Run stress tests
stress_results = stress_test_portfolio(
    portfolio=portfolio,
    stress_scenarios=scenarios,
    shock_magnitude=0.025
)

print("Stress Test Results:")
for scenario, result in stress_results['scenario_results'].items():
    print(f"{scenario}: ${result['portfolio_loss']:,.2f} loss")

print(f"\nWorst case loss: ${stress_results['worst_case_loss']:,.2f}")
```

### risk_attribution()

Perform risk attribution analysis to identify sources of portfolio risk.

#### Syntax

```python
risk_attribution(
    portfolio_weights: np.ndarray,
    factor_exposures: np.ndarray,
    factor_covariance: np.ndarray,
    specific_risk: np.ndarray = None
) -> dict
```

#### Parameters

- **portfolio_weights** (np.ndarray): Portfolio weights
- **factor_exposures** (np.ndarray): Factor exposures matrix
- **factor_covariance** (np.ndarray): Factor covariance matrix
- **specific_risk** (np.ndarray, optional): Specific risk for each asset. Default: None

#### Returns

- **dict**: Risk attribution results
  - `total_risk`: Total portfolio risk
  - `factor_contributions`: Risk contribution by factor
  - `specific_risk_contribution`: Contribution from specific risk
  - `risk_decomposition`: Detailed risk breakdown

#### Example

```python
# Risk attribution example
weights = np.array([0.25, 0.25, 0.25, 0.25])
factor_exp = np.random.normal(0, 1, (4, 3))  # 4 assets, 3 factors
factor_cov = np.array([
    [0.04, 0.01, 0.02],
    [0.01, 0.09, 0.01],
    [0.02, 0.01, 0.16]
])

attribution = risk_attribution(
    portfolio_weights=weights,
    factor_exposures=factor_exp,
    factor_covariance=factor_cov
)

print(f"Total Portfolio Risk: {attribution['total_risk']:.4f}")
print("Factor Contributions:")
for i, contrib in enumerate(attribution['factor_contributions']):
    print(f"  Factor {i+1}: {contrib:.4f}")
```

## Performance Considerations

- For large portfolios (>1000 assets), use the `parallel_processing=True` parameter
- Monte Carlo simulations with >100,000 iterations may require significant memory
- Historical VaR calculations are fastest but may not capture tail risks effectively
- Use appropriate window sizes based on market volatility regimes

## Error Handling

All functions include comprehensive error handling for:
- Invalid input data formats
- Insufficient historical data
- Numerical convergence issues
- Memory allocation problems

## Dependencies

- NumPy >= 1.19.0
- SciPy >= 1.7.0
- Pandas >= 1.3.0
- scikit-learn >= 0.24.0

## See Also

- [Portfolio Optimization](portfolio_optimization.md)
- [Performance Analytics](performance_analytics.md)
- [Credit Risk Assessment](credit_risk.md)
