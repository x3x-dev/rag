# Fixed Income Calculations Module

## Overview

The `minglib.fixed_income` module provides comprehensive tools for bond pricing, yield curve construction, duration analysis, and interest rate derivative valuation. This module is essential for fixed income trading desks and risk management teams.

## Installation

```python
from minglib.fixed_income import (
    BondPricer,
    YieldCurveBuilder,
    DurationCalculator,
    InterestRateSwap,
    CreditSpreadAnalyzer,
    InflationLinkedBonds
)
```

## Core Classes

### BondPricer

Comprehensive bond pricing engine for various bond types.

#### Syntax

```python
class BondPricer:
    def __init__(
        self,
        face_value: float = 1000.0,
        coupon_rate: float = 0.05,
        coupon_frequency: int = 2,
        maturity_years: float = 10.0,
        day_count_convention: str = "30/360"
    )
```

#### Parameters

- **face_value** (float, optional): Bond face value. Default: 1000.0
- **coupon_rate** (float, optional): Annual coupon rate. Default: 0.05
- **coupon_frequency** (int, optional): Coupons per year. Default: 2
- **maturity_years** (float, optional): Years to maturity. Default: 10.0
- **day_count_convention** (str, optional): Day count convention. Default: "30/360"

#### Methods

##### price_bond()

Calculate bond price given yield to maturity.

```python
def price_bond(
    self,
    yield_to_maturity: float,
    settlement_date: str = None,
    clean_price: bool = True
) -> dict:
    """
    Calculate bond price given yield to maturity.
    
    Parameters:
        yield_to_maturity (float): YTM for pricing
        settlement_date (str): Settlement date in 'YYYY-MM-DD' format
        clean_price (bool): Return clean price (excluding accrued interest)
    
    Returns:
        dict: Bond pricing results
    """
```

##### yield_to_maturity()

Calculate yield to maturity given bond price.

```python
def yield_to_maturity(
    self,
    bond_price: float,
    guess: float = 0.05,
    tolerance: float = 1e-8
) -> dict:
    """
    Calculate yield to maturity given bond price.
    
    Parameters:
        bond_price (float): Current bond price
        guess (float): Initial YTM guess
        tolerance (float): Convergence tolerance
    
    Returns:
        dict: YTM calculation results
    """
```

#### Example

```python
from minglib.fixed_income import BondPricer
from datetime import datetime, timedelta

# Initialize bond pricer for a 10-year Treasury bond
bond = BondPricer(
    face_value=1000.0,
    coupon_rate=0.04,      # 4% annual coupon
    coupon_frequency=2,     # Semi-annual payments
    maturity_years=10.0,
    day_count_convention="actual/actual"
)

# Price bond at 3.5% yield
pricing_result = bond.price_bond(
    yield_to_maturity=0.035,
    clean_price=True
)

print(f"Bond Price at 3.5% YTM: ${pricing_result['clean_price']:.4f}")
print(f"Accrued Interest: ${pricing_result['accrued_interest']:.4f}")
print(f"Dirty Price: ${pricing_result['dirty_price']:.4f}")
print(f"Duration: {pricing_result['modified_duration']:.4f}")
print(f"Convexity: {pricing_result['convexity']:.4f}")

# Calculate YTM for given price
ytm_result = bond.yield_to_maturity(bond_price=1025.50)
print(f"\nYTM for price $1025.50: {ytm_result['ytm']:.4f} or {ytm_result['ytm']*100:.2f}%")
print(f"Convergence achieved: {ytm_result['converged']}")
print(f"Iterations: {ytm_result['iterations']}")

# Sensitivity analysis
yield_range = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
print("\nPrice-Yield Sensitivity:")
print("YTM\tPrice\tDuration\tConvexity")
print("-" * 40)

for ytm in yield_range:
    result = bond.price_bond(ytm)
    print(f"{ytm:.3f}\t${result['clean_price']:.2f}\t{result['modified_duration']:.3f}\t\t{result['convexity']:.3f}")
```

### YieldCurveBuilder

Construct and interpolate yield curves from market data.

#### Syntax

```python
class YieldCurveBuilder:
    def __init__(
        self,
        interpolation_method: str = "cubic_spline",
        currency: str = "USD",
        curve_type: str = "zero_coupon"
    )
```

#### Methods

##### build_curve()

Build yield curve from bond market data.

```python
def build_curve(
    self,
    market_data: pd.DataFrame,
    reference_date: str,
    bootstrap_method: str = "standard"
) -> dict:
    """
    Build yield curve from market bond data.
    
    Parameters:
        market_data (pd.DataFrame): Bond market data with yields and maturities
        reference_date (str): Curve reference date
        bootstrap_method (str): Bootstrapping method for zero curve
    
    Returns:
        dict: Yield curve object and interpolation functions
    """
```

##### forward_rates()

Calculate forward rates from the yield curve.

```python
def forward_rates(
    self,
    start_periods: list,
    end_periods: list,
    compounding: str = "continuous"
) -> pd.DataFrame:
    """
    Calculate forward rates between specified periods.
    
    Parameters:
        start_periods (list): Start periods for forward rates
        end_periods (list): End periods for forward rates
        compounding (str): Compounding convention
    
    Returns:
        pd.DataFrame: Forward rates matrix
    """
```

#### Example

```python
from minglib.fixed_income import YieldCurveBuilder
import pandas as pd
import numpy as np

# Create sample market data
market_data = pd.DataFrame({
    'maturity_years': [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
    'yield': [0.015, 0.018, 0.022, 0.025, 0.028, 0.032, 0.035, 0.038, 0.042, 0.045],
    'bond_type': ['Treasury'] * 10,
    'coupon_rate': [0.02, 0.02, 0.025, 0.03, 0.035, 0.04, 0.042, 0.045, 0.05, 0.052]
})

# Initialize yield curve builder
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

print("Yield Curve Construction:")
print(f"Curve type: {yield_curve['curve_type']}")
print(f"Reference date: {yield_curve['reference_date']}")
print(f"Number of instruments: {yield_curve['num_instruments']}")
print(f"Curve fitting R-squared: {yield_curve['goodness_of_fit']:.4f}")

# Extract zero rates at specific maturities
test_maturities = [0.5, 1.5, 2.5, 4.0, 6.0, 8.0, 12.0, 25.0]
print("\nZero Rates (Bootstrapped):")
print("Maturity\tZero Rate\tPar Rate\tDiscount Factor")
print("-" * 50)

for maturity in test_maturities:
    zero_rate = yield_curve['zero_curve_function'](maturity)
    par_rate = yield_curve['par_curve_function'](maturity)
    discount_factor = np.exp(-zero_rate * maturity)
    print(f"{maturity:.1f}\t\t{zero_rate:.4f}\t\t{par_rate:.4f}\t\t{discount_factor:.6f}")

# Calculate forward rates
forward_periods = [(1, 2), (2, 3), (3, 5), (5, 10), (10, 30)]
print("\nForward Rates:")
print("Period\t\tForward Rate")
print("-" * 25)

for start, end in forward_periods:
    forward_rate = yield_curve['forward_rate_function'](start, end)
    print(f"{start}Y-{end}Y\t\t{forward_rate:.4f}")
```

### DurationCalculator

Calculate various duration and convexity measures for fixed income portfolios.

#### Syntax

```python
class DurationCalculator:
    def __init__(
        self,
        calculation_method: str = "modified",
        yield_shift: float = 0.0001
    )
```

#### Methods

##### portfolio_duration()

Calculate portfolio-level duration and risk metrics.

```python
def portfolio_duration(
    self,
    portfolio: dict,
    yield_curve: dict,
    duration_type: str = "modified"
) -> dict:
    """
    Calculate portfolio duration and convexity.
    
    Parameters:
        portfolio (dict): Portfolio holdings with weights and bond details
        yield_curve (dict): Yield curve for pricing
        duration_type (str): Type of duration calculation
    
    Returns:
        dict: Portfolio duration metrics
    """
```

#### Example

```python
from minglib.fixed_income import DurationCalculator

# Define a bond portfolio
portfolio = {
    'bonds': [
        {
            'cusip': 'BOND001',
            'face_value': 1000000,  # $1M face value
            'coupon_rate': 0.035,
            'maturity_years': 5.0,
            'market_value': 995000,
            'weight': 0.30
        },
        {
            'cusip': 'BOND002', 
            'face_value': 2000000,
            'coupon_rate': 0.045,
            'maturity_years': 10.0,
            'market_value': 2050000,
            'weight': 0.50
        },
        {
            'cusip': 'BOND003',
            'face_value': 500000,
            'coupon_rate': 0.025,
            'maturity_years': 2.0,
            'market_value': 485000,
            'weight': 0.20
        }
    ],
    'total_value': 3530000
}

# Initialize duration calculator
duration_calc = DurationCalculator(
    calculation_method="modified",
    yield_shift=0.0001
)

# Calculate portfolio duration
duration_metrics = duration_calc.portfolio_duration(
    portfolio=portfolio,
    yield_curve=yield_curve,
    duration_type="modified"
)

print("Portfolio Duration Analysis:")
print(f"Portfolio Modified Duration: {duration_metrics['portfolio_duration']:.4f}")
print(f"Portfolio Convexity: {duration_metrics['portfolio_convexity']:.4f}")
print(f"Duration Contribution by Bond:")
print(f"  BOND001 (5Y): {duration_metrics['individual_contributions']['BOND001']:.4f}")
print(f"  BOND002 (10Y): {duration_metrics['individual_contributions']['BOND002']:.4f}")
print(f"  BOND003 (2Y): {duration_metrics['individual_contributions']['BOND003']:.4f}")

# Stress test portfolio
stress_scenarios = [-0.01, -0.005, 0.005, 0.01, 0.02]  # Yield changes
print(f"\nPortfolio Stress Testing:")
print("Yield Change\tP&L\t\tP&L %")
print("-" * 35)

for yield_change in stress_scenarios:
    pnl = duration_calc.estimate_pnl(
        portfolio=portfolio,
        yield_change=yield_change,
        include_convexity=True
    )
    pnl_pct = (pnl / portfolio['total_value']) * 100
    print(f"{yield_change:+.3f}\t\t${pnl:,.0f}\t{pnl_pct:+.2f}%")
```

### InterestRateSwap

Interest rate swap pricing and risk analysis.

#### Syntax

```python
class InterestRateSwap:
    def __init__(
        self,
        notional: float,
        fixed_rate: float,
        floating_spread: float = 0.0,
        tenor_years: float = 5.0,
        payment_frequency: int = 4
    )
```

#### Methods

##### present_value()

Calculate present value of the swap.

```python
def present_value(
    self,
    yield_curve: dict,
    forward_curve: dict = None
) -> dict:
    """
    Calculate present value of interest rate swap.
    
    Parameters:
        yield_curve (dict): Discount curve for present value
        forward_curve (dict): Forward curve for floating leg
    
    Returns:
        dict: Swap valuation results
    """
```

#### Example

```python
from minglib.fixed_income import InterestRateSwap

# Initialize 5-year USD interest rate swap
# Pay fixed 3.5%, receive 3M LIBOR
swap = InterestRateSwap(
    notional=10000000,      # $10M notional
    fixed_rate=0.035,       # Pay 3.5% fixed
    floating_spread=0.0,    # No spread on LIBOR
    tenor_years=5.0,
    payment_frequency=4     # Quarterly payments
)

# Value the swap
swap_value = swap.present_value(
    yield_curve=yield_curve,
    forward_curve=yield_curve  # Using same curve for simplicity
)

print("Interest Rate Swap Valuation:")
print(f"Swap Notional: ${swap_value['notional']:,.0f}")
print(f"Fixed Leg PV: ${swap_value['fixed_leg_pv']:,.2f}")
print(f"Floating Leg PV: ${swap_value['floating_leg_pv']:,.2f}")
print(f"Net Present Value: ${swap_value['net_pv']:,.2f}")
print(f"DV01 (Dollar Duration): ${swap_value['dv01']:,.2f}")

# Calculate swap rate sensitivity
rate_shocks = [-0.005, -0.001, 0.001, 0.005]
print(f"\nSwap Rate Sensitivity:")
print("Rate Shock\tPV Change")
print("-" * 25)

for shock in rate_shocks:
    shocked_pv = swap.sensitivity_analysis(
        yield_curve=yield_curve,
        rate_shock=shock
    )
    pv_change = shocked_pv['net_pv'] - swap_value['net_pv']
    print(f"{shock:+.3f}\t\t${pv_change:,.0f}")
```

### CreditSpreadAnalyzer

Analyze credit spreads and default probabilities for corporate bonds.

#### Syntax

```python
class CreditSpreadAnalyzer:
    def __init__(
        self,
        recovery_rate: float = 0.40,
        default_model: str = "structural"
    )
```

#### Methods

##### calculate_credit_spread()

Calculate credit spread components.

```python
def calculate_credit_spread(
    self,
    corporate_yield: float,
    treasury_yield: float,
    maturity_years: float,
    rating: str = "BBB"
) -> dict:
    """
    Decompose credit spread into components.
    
    Parameters:
        corporate_yield (float): Corporate bond yield
        treasury_yield (float): Risk-free treasury yield
        maturity_years (float): Bond maturity
        rating (str): Credit rating
    
    Returns:
        dict: Credit spread analysis
    """
```

#### Example

```python
from minglib.fixed_income import CreditSpreadAnalyzer

# Initialize credit spread analyzer
credit_analyzer = CreditSpreadAnalyzer(
    recovery_rate=0.40,
    default_model="structural"
)

# Analyze credit spreads for different ratings
bonds_data = [
    {'rating': 'AAA', 'corporate_yield': 0.042, 'treasury_yield': 0.038, 'maturity': 10},
    {'rating': 'BBB', 'corporate_yield': 0.055, 'treasury_yield': 0.038, 'maturity': 10},
    {'rating': 'B', 'corporate_yield': 0.085, 'treasury_yield': 0.038, 'maturity': 10}
]

print("Credit Spread Analysis:")
print("Rating\tSpread\tDefault Prob\tLGD\tExpected Loss")
print("-" * 55)

for bond in bonds_data:
    spread_analysis = credit_analyzer.calculate_credit_spread(
        corporate_yield=bond['corporate_yield'],
        treasury_yield=bond['treasury_yield'],
        maturity_years=bond['maturity'],
        rating=bond['rating']
    )
    
    spread_bps = spread_analysis['credit_spread'] * 10000
    default_prob = spread_analysis['cumulative_default_probability']
    lgd = 1 - credit_analyzer.recovery_rate
    expected_loss = default_prob * lgd
    
    print(f"{bond['rating']}\t{spread_bps:.0f}bp\t{default_prob:.3f}\t\t{lgd:.3f}\t{expected_loss:.3f}")
```

## Utility Functions

### bond_cash_flows()

Generate detailed cash flow schedule for bonds.

```python
from minglib.fixed_income import bond_cash_flows

cash_flows = bond_cash_flows(
    face_value=1000,
    coupon_rate=0.04,
    maturity_years=5,
    frequency=2
)
```

### yield_curve_parallel_shift()

Apply parallel shifts to yield curves for scenario analysis.

```python
from minglib.fixed_income import yield_curve_parallel_shift

shifted_curve = yield_curve_parallel_shift(
    original_curve=yield_curve,
    shift_amount=0.005  # 50 basis points
)
```

## Performance Considerations

- Use vectorized operations for portfolio-level calculations
- Cache intermediate results for repeated bond valuations
- Implement parallel processing for large bond portfolios
- Use approximations for real-time pricing when exact precision isn't required

## Error Handling

```python
try:
    bond_price = bond.price_bond(yield_to_maturity=0.05)
except ValueError as e:
    print(f"Invalid bond parameters: {e}")
except ConvergenceError as e:
    print(f"YTM calculation failed to converge: {e}")
except ZeroDivisionError as e:
    print(f"Mathematical error in calculation: {e}")
```

## Dependencies

- NumPy >= 1.19.0
- SciPy >= 1.7.0
- Pandas >= 1.3.0
- QuantLib >= 1.20 (optional, for advanced models)
- datetime (standard library)

## See Also

- [Options Pricing](options_pricing.md)
- [Credit Risk Assessment](credit_risk.md)
- [Risk Management](risk_management.md)
