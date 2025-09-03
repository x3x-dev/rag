# Portfolio Optimization Module

## Overview

The `minglib.portfolio` module provides advanced portfolio optimization algorithms including mean-variance optimization, Black-Litterman model, risk parity strategies, and multi-objective optimization frameworks used in institutional asset management.

## Installation

```python
from minglib.portfolio import (
    mean_variance_optimizer,
    black_litterman_optimizer,
    risk_parity_optimizer,
    hierarchical_risk_parity,
    robust_optimizer,
    multi_period_optimizer
)
```

## Core Functions

### mean_variance_optimizer()

Implements Markowitz mean-variance optimization with various constraint options.

#### Syntax

```python
mean_variance_optimizer(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_aversion: float = 1.0,
    weight_bounds: tuple = (0.0, 1.0),
    target_return: float = None,
    target_volatility: float = None,
    constraints: list = None
) -> dict
```

#### Parameters

- **expected_returns** (np.ndarray): Expected returns for each asset
- **covariance_matrix** (np.ndarray): Asset return covariance matrix
- **risk_aversion** (float, optional): Risk aversion parameter. Default: 1.0
- **weight_bounds** (tuple, optional): Min/max weight bounds per asset. Default: (0.0, 1.0)
- **target_return** (float, optional): Target portfolio return. Default: None
- **target_volatility** (float, optional): Target portfolio volatility. Default: None
- **constraints** (list, optional): Additional constraints. Default: None

#### Returns

- **dict**: Optimization results
  - `weights`: Optimal portfolio weights
  - `expected_return`: Portfolio expected return
  - `volatility`: Portfolio volatility
  - `sharpe_ratio`: Portfolio Sharpe ratio
  - `optimization_status`: Solver status

#### Example

```python
import numpy as np
from minglib.portfolio import mean_variance_optimizer

# Define expected returns and covariance matrix
expected_returns = np.array([0.08, 0.12, 0.15, 0.18, 0.10])
cov_matrix = np.array([
    [0.0400, 0.0100, 0.0200, 0.0050, 0.0075],
    [0.0100, 0.0900, 0.0300, 0.0150, 0.0200],
    [0.0200, 0.0300, 0.1600, 0.0400, 0.0350],
    [0.0050, 0.0150, 0.0400, 0.2500, 0.0100],
    [0.0075, 0.0200, 0.0350, 0.0100, 0.0625]
])

# Optimize for maximum Sharpe ratio
optimal_portfolio = mean_variance_optimizer(
    expected_returns=expected_returns,
    covariance_matrix=cov_matrix,
    risk_aversion=1.0,
    weight_bounds=(0.0, 0.4)  # Maximum 40% allocation per asset
)

print("Optimal Portfolio Weights:")
for i, weight in enumerate(optimal_portfolio['weights']):
    print(f"Asset {i+1}: {weight:.3f}")

print(f"\nExpected Return: {optimal_portfolio['expected_return']:.4f}")
print(f"Volatility: {optimal_portfolio['volatility']:.4f}")
print(f"Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.4f}")

# Optimize for specific target return
target_return_portfolio = mean_variance_optimizer(
    expected_returns=expected_returns,
    covariance_matrix=cov_matrix,
    target_return=0.12,
    weight_bounds=(0.0, 0.5)
)

print(f"\nTarget Return Portfolio (12%):")
print(f"Achieved Return: {target_return_portfolio['expected_return']:.4f}")
print(f"Risk (Volatility): {target_return_portfolio['volatility']:.4f}")
```

### black_litterman_optimizer()

Implements the Black-Litterman model for incorporating investor views into portfolio optimization.

#### Syntax

```python
black_litterman_optimizer(
    market_caps: np.ndarray,
    price_data: np.ndarray,
    views_matrix: np.ndarray = None,
    view_returns: np.ndarray = None,
    view_confidences: np.ndarray = None,
    risk_aversion: float = 3.0,
    tau: float = 0.025
) -> dict
```

#### Parameters

- **market_caps** (np.ndarray): Market capitalizations for each asset
- **price_data** (np.ndarray): Historical price data matrix
- **views_matrix** (np.ndarray, optional): Matrix encoding investor views. Default: None
- **view_returns** (np.ndarray, optional): Expected returns from views. Default: None
- **view_confidences** (np.ndarray, optional): Confidence levels for views. Default: None
- **risk_aversion** (float, optional): Market risk aversion parameter. Default: 3.0
- **tau** (float, optional): Scaling factor for uncertainty. Default: 0.025

#### Returns

- **dict**: Black-Litterman optimization results
  - `bl_returns`: Black-Litterman adjusted expected returns
  - `bl_covariance`: Adjusted covariance matrix
  - `optimal_weights`: Optimal portfolio weights
  - `implied_returns`: Market-implied returns
  - `posterior_returns`: Posterior expected returns

#### Example

```python
# Market cap data (in millions)
market_caps = np.array([2500000, 1800000, 1200000, 800000, 600000])

# Historical price data (252 days x 5 assets)
np.random.seed(42)
price_data = np.random.multivariate_normal(
    mean=[100, 80, 120, 60, 40],
    cov=cov_matrix,
    size=252
)

# Define investor views
# View 1: Asset 1 will outperform Asset 2 by 2%
# View 2: Asset 3 will return 18%
views_matrix = np.array([
    [1, -1, 0, 0, 0],   # Asset 1 - Asset 2
    [0, 0, 1, 0, 0]     # Asset 3
])
view_returns = np.array([0.02, 0.18])
view_confidences = np.array([0.7, 0.8])  # 70% and 80% confidence

# Run Black-Litterman optimization
bl_result = black_litterman_optimizer(
    market_caps=market_caps,
    price_data=price_data,
    views_matrix=views_matrix,
    view_returns=view_returns,
    view_confidences=view_confidences,
    risk_aversion=2.5
)

print("Black-Litterman Results:")
print(f"Implied Returns: {bl_result['implied_returns']}")
print(f"Posterior Returns: {bl_result['posterior_returns']}")
print(f"Optimal Weights: {bl_result['optimal_weights']}")
```

### risk_parity_optimizer()

Implements risk parity optimization strategies including equal risk contribution and budgeted risk.

#### Syntax

```python
risk_parity_optimizer(
    covariance_matrix: np.ndarray,
    risk_budgets: np.ndarray = None,
    method: str = "equal_risk",
    max_iterations: int = 1000,
    tolerance: float = 1e-8
) -> dict
```

#### Parameters

- **covariance_matrix** (np.ndarray): Asset return covariance matrix
- **risk_budgets** (np.ndarray, optional): Target risk budgets for each asset. Default: None
- **method** (str, optional): Risk parity method. Options: "equal_risk", "budgeted_risk". Default: "equal_risk"
- **max_iterations** (int, optional): Maximum optimization iterations. Default: 1000
- **tolerance** (float, optional): Convergence tolerance. Default: 1e-8

#### Returns

- **dict**: Risk parity optimization results
  - `weights`: Risk parity portfolio weights
  - `risk_contributions`: Risk contribution of each asset
  - `total_risk`: Total portfolio risk
  - `convergence_achieved`: Whether optimization converged

#### Example

```python
# Equal Risk Contribution portfolio
rp_equal = risk_parity_optimizer(
    covariance_matrix=cov_matrix,
    method="equal_risk"
)

print("Equal Risk Contribution Portfolio:")
print(f"Weights: {rp_equal['weights']}")
print(f"Risk Contributions: {rp_equal['risk_contributions']}")
print(f"Total Risk: {rp_equal['total_risk']:.4f}")

# Budgeted Risk Contribution portfolio
risk_budgets = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
rp_budgeted = risk_parity_optimizer(
    covariance_matrix=cov_matrix,
    risk_budgets=risk_budgets,
    method="budgeted_risk"
)

print("\nBudgeted Risk Contribution Portfolio:")
print(f"Target Budgets: {risk_budgets}")
print(f"Actual Contributions: {rp_budgeted['risk_contributions']}")
print(f"Weights: {rp_budgeted['weights']}")
```

### hierarchical_risk_parity()

Implements Hierarchical Risk Parity (HRP) algorithm using machine learning clustering techniques.

#### Syntax

```python
hierarchical_risk_parity(
    returns_data: np.ndarray,
    linkage_method: str = "ward",
    distance_metric: str = "correlation",
    num_clusters: int = None
) -> dict
```

#### Parameters

- **returns_data** (np.ndarray): Historical returns matrix (time x assets)
- **linkage_method** (str, optional): Hierarchical clustering linkage method. Default: "ward"
- **distance_metric** (str, optional): Distance metric for clustering. Default: "correlation"
- **num_clusters** (int, optional): Number of clusters for allocation. Default: None

#### Returns

- **dict**: HRP optimization results
  - `weights`: HRP portfolio weights
  - `dendrogram_order`: Asset ordering from clustering
  - `cluster_allocations`: Allocation within each cluster
  - `cluster_weights`: Weight allocation between clusters

#### Example

```python
# Generate sample returns data
np.random.seed(42)
returns_data = np.random.multivariate_normal(
    mean=np.zeros(5),
    cov=cov_matrix / 252,  # Daily returns
    size=252
)

# Apply Hierarchical Risk Parity
hrp_result = hierarchical_risk_parity(
    returns_data=returns_data,
    linkage_method="ward",
    distance_metric="correlation"
)

print("Hierarchical Risk Parity Results:")
print(f"HRP Weights: {hrp_result['weights']}")
print(f"Asset Order: {hrp_result['dendrogram_order']}")
print(f"Cluster Allocations: {hrp_result['cluster_allocations']}")
```

### robust_optimizer()

Implements robust optimization techniques to handle parameter uncertainty.

#### Syntax

```python
robust_optimizer(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    uncertainty_set: dict,
    robust_method: str = "worst_case",
    confidence_level: float = 0.95
) -> dict
```

#### Parameters

- **expected_returns** (np.ndarray): Expected returns estimate
- **covariance_matrix** (np.ndarray): Covariance matrix estimate
- **uncertainty_set** (dict): Definition of parameter uncertainty
- **robust_method** (str, optional): Robust optimization method. Default: "worst_case"
- **confidence_level** (float, optional): Confidence level for robust solution. Default: 0.95

#### Returns

- **dict**: Robust optimization results
  - `robust_weights`: Robust portfolio weights
  - `worst_case_return`: Worst-case portfolio return
  - `robust_sharpe`: Robust Sharpe ratio
  - `sensitivity_analysis`: Parameter sensitivity results

#### Example

```python
# Define uncertainty around expected returns
uncertainty_set = {
    'return_uncertainty': 0.02,  # 2% uncertainty in expected returns
    'covariance_uncertainty': 0.1,  # 10% uncertainty in covariance
    'correlation_uncertainty': 0.05  # 5% uncertainty in correlations
}

# Optimize robust portfolio
robust_result = robust_optimizer(
    expected_returns=expected_returns,
    covariance_matrix=cov_matrix,
    uncertainty_set=uncertainty_set,
    robust_method="worst_case",
    confidence_level=0.95
)

print("Robust Portfolio Results:")
print(f"Robust Weights: {robust_result['robust_weights']}")
print(f"Worst-case Return: {robust_result['worst_case_return']:.4f}")
print(f"Robust Sharpe Ratio: {robust_result['robust_sharpe']:.4f}")
```

### multi_period_optimizer()

Implements multi-period portfolio optimization with transaction costs and rebalancing constraints.

#### Syntax

```python
multi_period_optimizer(
    returns_forecasts: np.ndarray,
    covariance_forecasts: np.ndarray,
    transaction_costs: np.ndarray,
    initial_weights: np.ndarray,
    periods: int = 12,
    rebalancing_frequency: str = "monthly"
) -> dict
```

#### Parameters

- **returns_forecasts** (np.ndarray): Multi-period return forecasts
- **covariance_forecasts** (np.ndarray): Multi-period covariance forecasts
- **transaction_costs** (np.ndarray): Transaction cost matrix
- **initial_weights** (np.ndarray): Starting portfolio weights
- **periods** (int, optional): Number of optimization periods. Default: 12
- **rebalancing_frequency** (str, optional): Rebalancing frequency. Default: "monthly"

#### Returns

- **dict**: Multi-period optimization results
  - `optimal_path`: Optimal weight path over time
  - `transaction_costs_incurred`: Total transaction costs
  - `cumulative_returns`: Cumulative portfolio returns
  - `turnover_analysis`: Portfolio turnover statistics

#### Example

```python
# Multi-period optimization setup
periods = 12
returns_forecasts = np.random.normal(0.01, 0.02, (periods, 5))
cov_forecasts = np.tile(cov_matrix, (periods, 1, 1))
transaction_costs = np.full((5, 5), 0.001)  # 10 bps transaction costs
initial_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Run multi-period optimization
mp_result = multi_period_optimizer(
    returns_forecasts=returns_forecasts,
    covariance_forecasts=cov_forecasts,
    transaction_costs=transaction_costs,
    initial_weights=initial_weights,
    periods=periods
)

print("Multi-Period Optimization Results:")
print(f"Total Transaction Costs: {mp_result['transaction_costs_incurred']:.4f}")
print(f"Final Cumulative Return: {mp_result['cumulative_returns'][-1]:.4f}")
print(f"Average Monthly Turnover: {np.mean(mp_result['turnover_analysis']):.4f}")
```

## Utility Functions

### efficient_frontier()

Generate the efficient frontier for mean-variance optimization.

#### Syntax

```python
from minglib.portfolio import efficient_frontier

frontier_data = efficient_frontier(
    expected_returns=expected_returns,
    covariance_matrix=cov_matrix,
    num_portfolios=100,
    risk_free_rate=0.02
)
```

### portfolio_performance()

Calculate comprehensive portfolio performance metrics.

#### Syntax

```python
from minglib.portfolio import portfolio_performance

performance = portfolio_performance(
    weights=optimal_portfolio['weights'],
    expected_returns=expected_returns,
    covariance_matrix=cov_matrix,
    risk_free_rate=0.02
)
```

## Performance Considerations

- Use sparse matrices for large-scale optimization problems (>500 assets)
- Consider using approximate methods for very high-dimensional problems
- Monte Carlo simulation may be required for non-convex constraints
- Parallel processing available for multi-period optimization

## Error Handling

```python
try:
    result = mean_variance_optimizer(expected_returns, cov_matrix)
except ValueError as e:
    print(f"Input validation error: {e}")
except RuntimeError as e:
    print(f"Optimization failed: {e}")
except np.linalg.LinAlgError as e:
    print(f"Matrix computation error: {e}")
```

## Dependencies

- NumPy >= 1.19.0
- SciPy >= 1.7.0
- scikit-learn >= 0.24.0
- CVXPY >= 1.1.0 (for convex optimization)
- matplotlib >= 3.3.0 (for visualization)

## See Also

- [Risk Management](risk_management.md)
- [Performance Analytics](performance_analytics.md)
- [Backtesting Framework](backtesting.md)
