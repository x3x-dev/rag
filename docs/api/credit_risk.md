# Credit Risk Assessment Module

## Overview

The `minglib.credit` module provides advanced credit risk modeling tools including probability of default (PD) models, loss given default (LGD) estimation, exposure at default (EAD) calculations, and portfolio credit risk analytics for regulatory and economic capital calculations.

## Installation

```python
from minglib.credit import (
    DefaultProbabilityModel,
    LossGivenDefaultEstimator,
    ExposureAtDefaultCalculator,
    CreditPortfolioAnalyzer,
    StressTestingEngine,
    Regulatory CapitalCalculator
)
```

## Core Models

### DefaultProbabilityModel

Comprehensive PD modeling using various methodologies including structural, reduced-form, and machine learning approaches.

#### Syntax

```python
class DefaultProbabilityModel:
    def __init__(
        self,
        model_type: str = "logistic",
        time_horizon: float = 1.0,
        calibration_method: str = "maximum_likelihood"
    )
```

#### Parameters

- **model_type** (str, optional): PD model type. Options: "logistic", "probit", "merton", "creditmetrics". Default: "logistic"
- **time_horizon** (float, optional): Time horizon in years. Default: 1.0
- **calibration_method** (str, optional): Model calibration method. Default: "maximum_likelihood"

#### Methods

##### fit_model()

Train the PD model on historical data.

```python
def fit_model(
    self,
    features: pd.DataFrame,
    default_indicators: pd.Series,
    validation_split: float = 0.2
) -> dict:
    """
    Train probability of default model.
    
    Parameters:
        features (pd.DataFrame): Financial ratios and macro variables
        default_indicators (pd.Series): Binary default indicators
        validation_split (float): Validation data proportion
    
    Returns:
        dict: Model training results and performance metrics
    """
```

##### predict_pd()

Predict default probabilities for new observations.

```python
def predict_pd(
    self,
    features: pd.DataFrame,
    confidence_intervals: bool = True
) -> dict:
    """
    Predict default probabilities.
    
    Parameters:
        features (pd.DataFrame): Input features for prediction
        confidence_intervals (bool): Whether to return confidence intervals
    
    Returns:
        dict: PD predictions and uncertainty estimates
    """
```

#### Example

```python
from minglib.credit import DefaultProbabilityModel
import pandas as pd
import numpy as np

# Generate sample credit data
np.random.seed(42)
n_companies = 5000

# Financial ratios
credit_data = pd.DataFrame({
    'debt_to_equity': np.random.lognormal(0.5, 0.8, n_companies),
    'current_ratio': np.random.lognormal(0.3, 0.5, n_companies),
    'roa': np.random.normal(0.05, 0.15, n_companies),
    'interest_coverage': np.random.lognormal(1.5, 1.0, n_companies),
    'asset_turnover': np.random.lognormal(-0.2, 0.6, n_companies),
    'market_cap_log': np.random.normal(15, 2, n_companies),
    'gdp_growth': np.random.normal(0.025, 0.02, n_companies),
    'credit_spread': np.random.lognormal(-3, 0.5, n_companies)
})

# Generate default indicators (higher probability for worse ratios)
default_score = (
    -2.0 + 
    0.5 * credit_data['debt_to_equity'] + 
    -1.0 * credit_data['current_ratio'] + 
    -3.0 * credit_data['roa'] + 
    -0.3 * np.log(credit_data['interest_coverage']) +
    np.random.normal(0, 1, n_companies)
)
default_prob = 1 / (1 + np.exp(-default_score))
defaults = np.random.binomial(1, default_prob)

# Initialize PD model
pd_model = DefaultProbabilityModel(
    model_type="logistic",
    time_horizon=1.0,
    calibration_method="maximum_likelihood"
)

# Train the model
training_results = pd_model.fit_model(
    features=credit_data,
    default_indicators=pd.Series(defaults),
    validation_split=0.25
)

print("PD Model Training Results:")
print(f"Model Type: {training_results['model_type']}")
print(f"Training AUC: {training_results['train_auc']:.4f}")
print(f"Validation AUC: {training_results['validation_auc']:.4f}")
print(f"Gini Coefficient: {training_results['gini_coefficient']:.4f}")
print(f"Kolmogorov-Smirnov Statistic: {training_results['ks_statistic']:.4f}")

# Feature importance
print("\nTop 5 Most Important Features:")
for feature, importance in training_results['feature_importance'].items()[:5]:
    print(f"  {feature}: {importance:.4f}")

# Predict on new data
new_companies = pd.DataFrame({
    'debt_to_equity': [0.8, 2.5, 1.2],
    'current_ratio': [1.5, 0.8, 2.1],
    'roa': [0.08, -0.02, 0.12],
    'interest_coverage': [5.2, 1.1, 8.7],
    'asset_turnover': [0.9, 0.6, 1.3],
    'market_cap_log': [16.5, 14.2, 18.1],
    'gdp_growth': [0.03, 0.02, 0.025],
    'credit_spread': [0.02, 0.08, 0.015]
})

predictions = pd_model.predict_pd(
    features=new_companies,
    confidence_intervals=True
)

print("\nPD Predictions for New Companies:")
print("Company\tPD\t95% CI Lower\t95% CI Upper\tRating Equivalent")
print("-" * 70)
for i, (pd_est, ci_lower, ci_upper) in enumerate(zip(
    predictions['pd_estimates'], 
    predictions['ci_lower'], 
    predictions['ci_upper']
)):
    rating = pd_model.pd_to_rating(pd_est)
    print(f"{i+1}\t{pd_est:.4f}\t{ci_lower:.4f}\t\t{ci_upper:.4f}\t\t{rating}")
```

### LossGivenDefaultEstimator

Estimate loss given default rates using historical recovery data and downturn scenarios.

#### Syntax

```python
class LossGivenDefaultEstimator:
    def __init__(
        self,
        estimation_method: str = "beta_regression",
        downturn_adjustment: bool = True,
        industry_effects: bool = True
    )
```

#### Methods

##### estimate_lgd()

Estimate LGD for different exposure types and industries.

```python
def estimate_lgd(
    self,
    exposure_data: pd.DataFrame,
    recovery_rates: pd.Series,
    downturn_scenario: bool = False
) -> dict:
    """
    Estimate Loss Given Default rates.
    
    Parameters:
        exposure_data (pd.DataFrame): Exposure characteristics
        recovery_rates (pd.Series): Historical recovery rates
        downturn_scenario (bool): Whether to apply downturn adjustments
    
    Returns:
        dict: LGD estimates and confidence intervals
    """
```

#### Example

```python
from minglib.credit import LossGivenDefaultEstimator

# Sample exposure data
exposure_data = pd.DataFrame({
    'facility_type': ['term_loan', 'revolver', 'bond', 'mortgage', 'equipment'],
    'seniority': ['senior_secured', 'senior_unsecured', 'subordinated', 'senior_secured', 'senior_secured'],
    'industry': ['technology', 'retail', 'manufacturing', 'real_estate', 'transportation'],
    'collateral_value': [1000000, 500000, 0, 2000000, 800000],
    'exposure_amount': [800000, 300000, 1000000, 1500000, 600000],
    'geographic_region': ['north_america', 'europe', 'north_america', 'asia', 'north_america']
})

# Historical recovery rates (as a fraction)
recovery_rates = pd.Series([0.75, 0.45, 0.35, 0.80, 0.65])

# Initialize LGD estimator
lgd_estimator = LossGivenDefaultEstimator(
    estimation_method="beta_regression",
    downturn_adjustment=True,
    industry_effects=True
)

# Estimate LGD
lgd_results = lgd_estimator.estimate_lgd(
    exposure_data=exposure_data,
    recovery_rates=recovery_rates,
    downturn_scenario=False
)

print("LGD Estimation Results:")
print("Facility\tIndustry\tBase LGD\tDownturn LGD\tConfidence Interval")
print("-" * 75)

for i, facility in enumerate(exposure_data['facility_type']):
    base_lgd = lgd_results['base_lgd_estimates'][i]
    downturn_lgd = lgd_results['downturn_lgd_estimates'][i]
    ci_lower = lgd_results['confidence_intervals'][i][0]
    ci_upper = lgd_results['confidence_intervals'][i][1]
    industry = exposure_data['industry'].iloc[i]
    
    print(f"{facility}\t{industry}\t{base_lgd:.3f}\t\t{downturn_lgd:.3f}\t\t[{ci_lower:.3f}, {ci_upper:.3f}]")

# Industry-level LGD analysis
industry_lgd = lgd_estimator.industry_lgd_analysis(
    exposure_data=exposure_data,
    recovery_rates=recovery_rates
)

print(f"\nIndustry LGD Analysis:")
for industry, lgd_stats in industry_lgd.items():
    print(f"{industry}: Mean LGD = {lgd_stats['mean_lgd']:.3f}, Std = {lgd_stats['std_lgd']:.3f}")
```

### CreditPortfolioAnalyzer

Analyze credit risk at the portfolio level including concentration risk and correlation effects.

#### Syntax

```python
class CreditPortfolioAnalyzer:
    def __init__(
        self,
        correlation_model: str = "factor_model",
        monte_carlo_simulations: int = 100000
    )
```

#### Methods

##### portfolio_risk_metrics()

Calculate comprehensive portfolio credit risk metrics.

```python
def portfolio_risk_metrics(
    self,
    portfolio_data: pd.DataFrame,
    correlation_matrix: np.ndarray = None,
    confidence_levels: list = [0.95, 0.99, 0.999]
) -> dict:
    """
    Calculate portfolio credit risk metrics.
    
    Parameters:
        portfolio_data (pd.DataFrame): Portfolio exposures with PD and LGD
        correlation_matrix (np.ndarray): Asset correlation matrix
        confidence_levels (list): Confidence levels for VaR calculation
    
    Returns:
        dict: Portfolio risk metrics including expected loss and VaR
    """
```

#### Example

```python
from minglib.credit import CreditPortfolioAnalyzer
import numpy as np

# Define credit portfolio
portfolio = pd.DataFrame({
    'obligor_id': [f'COMP_{i:03d}' for i in range(1, 51)],
    'exposure_amount': np.random.lognormal(15, 1, 50) * 1000,  # $1K to $100M
    'pd_1year': np.random.beta(1, 50, 50),  # Low default probabilities
    'lgd': np.random.beta(2, 3, 50),  # LGD around 40%
    'industry': np.random.choice(['tech', 'finance', 'manufacturing', 'retail', 'energy'], 50),
    'rating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B'], 50, 
                              p=[0.05, 0.1, 0.2, 0.35, 0.2, 0.1]),
    'geographic_region': np.random.choice(['north_america', 'europe', 'asia'], 50, 
                                        p=[0.5, 0.3, 0.2])
})

# Generate correlation matrix (simplified factor model)
np.random.seed(42)
correlation_matrix = np.random.uniform(0.1, 0.3, (50, 50))
np.fill_diagonal(correlation_matrix, 1.0)
# Make symmetric
correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
np.fill_diagonal(correlation_matrix, 1.0)

# Initialize portfolio analyzer
portfolio_analyzer = CreditPortfolioAnalyzer(
    correlation_model="gaussian_copula",
    monte_carlo_simulations=250000
)

# Calculate portfolio risk metrics
risk_metrics = portfolio_analyzer.portfolio_risk_metrics(
    portfolio_data=portfolio,
    correlation_matrix=correlation_matrix,
    confidence_levels=[0.95, 0.99, 0.999]
)

print("Portfolio Credit Risk Analysis:")
print(f"Total Exposure: ${risk_metrics['total_exposure']:,.0f}")
print(f"Expected Loss: ${risk_metrics['expected_loss']:,.0f}")
print(f"Expected Loss Rate: {risk_metrics['expected_loss_rate']:.4f}")
print(f"Loss Volatility: ${risk_metrics['loss_volatility']:,.0f}")

print(f"\nCredit VaR (Monte Carlo):")
for conf_level, var_value in risk_metrics['credit_var'].items():
    print(f"  {conf_level:.1%} VaR: ${var_value:,.0f}")

print(f"\nExpected Shortfall:")
for conf_level, es_value in risk_metrics['expected_shortfall'].items():
    print(f"  {conf_level:.1%} ES: ${es_value:,.0f}")

# Concentration analysis
concentration = portfolio_analyzer.concentration_analysis(portfolio)
print(f"\nConcentration Risk:")
print(f"HHI by Exposure: {concentration['hhi_exposure']:.4f}")
print(f"HHI by Industry: {concentration['hhi_industry']:.4f}")
print(f"Top 10 Exposure Concentration: {concentration['top10_concentration']:.2%}")
print(f"Single Name Concentration Limit Breaches: {concentration['concentration_breaches']}")
```

### StressTestingEngine

Comprehensive stress testing framework for credit portfolios under adverse scenarios.

#### Syntax

```python
class StressTestingEngine:
    def __init__(
        self,
        scenario_types: list = ["recession", "financial_crisis", "sector_specific"],
        correlation_shock: bool = True
    )
```

#### Methods

##### run_stress_tests()

Execute stress tests under multiple scenarios.

```python
def run_stress_tests(
    self,
    portfolio_data: pd.DataFrame,
    stress_scenarios: dict,
    base_correlations: np.ndarray
) -> dict:
    """
    Run comprehensive stress testing.
    
    Parameters:
        portfolio_data (pd.DataFrame): Portfolio data
        stress_scenarios (dict): Stress scenario definitions
        base_correlations (np.ndarray): Base case correlations
    
    Returns:
        dict: Stress testing results
    """
```

#### Example

```python
from minglib.credit import StressTestingEngine

# Define stress scenarios
stress_scenarios = {
    'severe_recession': {
        'pd_multiplier': 3.0,
        'lgd_adjustment': 0.1,  # +10pp
        'correlation_increase': 0.2,
        'gdp_decline': -0.06,
        'unemployment_increase': 0.05
    },
    'financial_crisis': {
        'pd_multiplier': 4.5,
        'lgd_adjustment': 0.15,
        'correlation_increase': 0.35,
        'credit_spread_shock': 0.05,
        'liquidity_crisis': True
    },
    'tech_sector_crash': {
        'sector_specific': 'technology',
        'pd_multiplier': 6.0,
        'lgd_adjustment': 0.20,
        'sector_correlation': 0.8
    }
}

# Initialize stress testing engine
stress_engine = StressTestingEngine(
    scenario_types=["macroeconomic", "market", "sector_specific"],
    correlation_shock=True
)

# Run stress tests
stress_results = stress_engine.run_stress_tests(
    portfolio_data=portfolio,
    stress_scenarios=stress_scenarios,
    base_correlations=correlation_matrix
)

print("Credit Portfolio Stress Testing Results:")
print("=" * 60)

for scenario_name, results in stress_results['scenario_results'].items():
    print(f"\n{scenario_name.upper()}:")
    print(f"  Expected Loss: ${results['expected_loss']:,.0f} ({results['el_increase']:.1f}x base)")
    print(f"  99% VaR: ${results['var_99']:,.0f} ({results['var_increase']:.1f}x base)")
    print(f"  Default Rate: {results['default_rate']:.2%}")
    print(f"  Capital Impact: ${results['capital_impact']:,.0f}")

# Reverse stress testing
reverse_stress = stress_engine.reverse_stress_test(
    portfolio_data=portfolio,
    target_loss_level=risk_metrics['total_exposure'] * 0.05  # 5% portfolio loss
)

print(f"\nReverse Stress Test (5% Portfolio Loss):")
print(f"Required PD Multiplier: {reverse_stress['pd_multiplier']:.2f}")
print(f"Probability of Scenario: {reverse_stress['scenario_probability']:.4f}")
```

## Regulatory Capital Functions

### RegulatorCapitalCalculator

Calculate regulatory capital requirements under Basel III/IV frameworks.

#### Example

```python
from minglib.credit import RegulatoryCapitalCalculator

# Initialize capital calculator
capital_calc = RegulatoryCapitalCalculator(
    framework="basel_iii",
    internal_model=True
)

# Calculate capital requirements
capital_requirements = capital_calc.calculate_capital(
    portfolio_data=portfolio,
    pd_estimates=portfolio['pd_1year'],
    lgd_estimates=portfolio['lgd'],
    correlation_matrix=correlation_matrix
)

print("Regulatory Capital Requirements:")
print(f"Risk-Weighted Assets: ${capital_requirements['rwa']:,.0f}")
print(f"Minimum Capital (8%): ${capital_requirements['minimum_capital']:,.0f}")
print(f"Tier 1 Capital Requirement: ${capital_requirements['tier1_capital']:,.0f}")
print(f"Total Capital Requirement: ${capital_requirements['total_capital']:,.0f}")
print(f"Capital Conservation Buffer: ${capital_requirements['conservation_buffer']:,.0f}")
```

## Performance Considerations

- Use sparse matrices for large correlation matrices
- Implement parallel processing for Monte Carlo simulations
- Cache frequently accessed model parameters
- Use approximation methods for real-time risk monitoring
- Optimize memory usage for large portfolio calculations

## Error Handling

```python
try:
    pd_predictions = pd_model.predict_pd(features)
except ValueError as e:
    print(f"Invalid feature data: {e}")
except ModelNotFittedError as e:
    print(f"Model must be trained first: {e}")
except CorrelationMatrixError as e:
    print(f"Invalid correlation matrix: {e}")
```

## Dependencies

- NumPy >= 1.19.0
- SciPy >= 1.7.0
- Pandas >= 1.3.0
- scikit-learn >= 0.24.0
- statsmodels >= 0.12.0
- matplotlib >= 3.3.0

## See Also

- [Risk Management](risk_management.md)
- [Fixed Income](fixed_income.md)
- [Portfolio Optimization](portfolio_optimization.md)
