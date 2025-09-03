# Options Pricing Models Module

## Overview

The `minglib.options` module provides comprehensive options pricing models including Black-Scholes, Binomial trees, Monte Carlo simulation, and advanced models for exotic options. This module is designed for institutional trading desks and quantitative analysts.

## Installation

```python
from minglib.options import (
    BlackScholesModel,
    BinomialTreeModel,
    MonteCarloEngine,
    ImpliedVolatilitySolver,
    ExoticOptionsCalculator,
    VolatilitySurfaceBuilder
)
```

## Core Models

### BlackScholesModel

Classical Black-Scholes-Merton model for European options pricing.

#### Syntax

```python
class BlackScholesModel:
    def __init__(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0
    )
```

#### Parameters

- **spot_price** (float): Current underlying asset price
- **strike_price** (float): Option strike price
- **time_to_expiry** (float): Time to expiration in years
- **risk_free_rate** (float): Risk-free interest rate (annualized)
- **volatility** (float): Implied volatility (annualized)
- **dividend_yield** (float, optional): Continuous dividend yield. Default: 0.0

#### Methods

##### option_price()

Calculate European option price using Black-Scholes formula.

```python
def option_price(self, option_type: str = "call") -> float:
    """
    Calculate option price using Black-Scholes formula.
    
    Parameters:
        option_type (str): "call" or "put"
    
    Returns:
        float: Option price
    """
```

##### greeks()

Calculate option Greeks (delta, gamma, theta, vega, rho).

```python
def greeks(self, option_type: str = "call") -> dict:
    """
    Calculate all option Greeks.
    
    Parameters:
        option_type (str): "call" or "put"
    
    Returns:
        dict: Dictionary containing all Greeks
    """
```

#### Example

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

# Calculate put option price
put_price = bs_model.option_price(option_type="put")
print(f"Put Option Price: ${put_price:.4f}")

# Calculate Greeks for call option
call_greeks = bs_model.greeks(option_type="call")
print("\nCall Option Greeks:")
print(f"Delta: {call_greeks['delta']:.4f}")
print(f"Gamma: {call_greeks['gamma']:.4f}")
print(f"Theta: {call_greeks['theta']:.4f}")
print(f"Vega: {call_greeks['vega']:.4f}")
print(f"Rho: {call_greeks['rho']:.4f}")

# Verify put-call parity
parity_check = call_price - put_price
theoretical_parity = bs_model.spot_price - bs_model.strike_price * \
    np.exp(-bs_model.risk_free_rate * bs_model.time_to_expiry)
print(f"\nPut-Call Parity Check:")
print(f"Market difference: {parity_check:.6f}")
print(f"Theoretical difference: {theoretical_parity:.6f}")
print(f"Arbitrage opportunity: {abs(parity_check - theoretical_parity) > 0.01}")
```

### BinomialTreeModel

Binomial tree model for American and European options with early exercise features.

#### Syntax

```python
class BinomialTreeModel:
    def __init__(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        num_steps: int = 100,
        dividend_yield: float = 0.0
    )
```

#### Methods

##### american_option_price()

Price American options with early exercise capability.

```python
def american_option_price(
    self,
    option_type: str = "call",
    exercise_style: str = "american"
) -> dict:
    """
    Price American options using binomial tree.
    
    Parameters:
        option_type (str): "call" or "put"
        exercise_style (str): "american" or "european"
    
    Returns:
        dict: Option price and exercise boundary
    """
```

#### Example

```python
from minglib.options import BinomialTreeModel

# Initialize binomial tree model
binomial_model = BinomialTreeModel(
    spot_price=100.0,
    strike_price=110.0,
    time_to_expiry=0.5,    # 6 months
    risk_free_rate=0.06,
    volatility=0.25,
    num_steps=200,         # 200 time steps for accuracy
    dividend_yield=0.03
)

# Price American put option
american_put = binomial_model.american_option_price(
    option_type="put",
    exercise_style="american"
)

print(f"American Put Price: ${american_put['option_price']:.4f}")
print(f"Early Exercise Premium: ${american_put['early_exercise_premium']:.4f}")

# Compare with European put
european_put = binomial_model.american_option_price(
    option_type="put",
    exercise_style="european"
)

print(f"European Put Price: ${european_put['option_price']:.4f}")
print(f"American Premium: ${american_put['option_price'] - european_put['option_price']:.4f}")

# Analyze exercise boundary
exercise_boundary = american_put['exercise_boundary']
print(f"\nExercise Boundary Analysis:")
print(f"Immediate exercise threshold: ${exercise_boundary['immediate_exercise']:.2f}")
print(f"Never exercise above: ${exercise_boundary['never_exercise']:.2f}")
```

### MonteCarloEngine

Monte Carlo simulation engine for complex derivatives and path-dependent options.

#### Syntax

```python
class MonteCarloEngine:
    def __init__(
        self,
        num_simulations: int = 100000,
        num_time_steps: int = 252,
        random_seed: int = None,
        antithetic_variates: bool = True,
        control_variates: bool = False
    )
```

#### Methods

##### price_barrier_option()

Price barrier options using Monte Carlo simulation.

```python
def price_barrier_option(
    self,
    spot_price: float,
    strike_price: float,
    barrier_level: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    barrier_type: str = "knock_out",
    option_type: str = "call"
) -> dict:
    """
    Price barrier options using Monte Carlo simulation.
    
    Parameters:
        spot_price (float): Current underlying price
        strike_price (float): Option strike price
        barrier_level (float): Barrier level
        time_to_expiry (float): Time to expiration
        risk_free_rate (float): Risk-free rate
        volatility (float): Volatility
        barrier_type (str): "knock_out", "knock_in"
        option_type (str): "call" or "put"
    
    Returns:
        dict: Option price and simulation statistics
    """
```

##### price_asian_option()

Price Asian (average price) options.

```python
def price_asian_option(
    self,
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    average_type: str = "arithmetic",
    option_type: str = "call"
) -> dict:
    """
    Price Asian options using Monte Carlo simulation.
    
    Parameters:
        spot_price (float): Current underlying price
        strike_price (float): Option strike price
        time_to_expiry (float): Time to expiration
        risk_free_rate (float): Risk-free rate
        volatility (float): Volatility
        average_type (str): "arithmetic" or "geometric"
        option_type (str): "call" or "put"
    
    Returns:
        dict: Option price and confidence intervals
    """
```

#### Example

```python
from minglib.options import MonteCarloEngine
import numpy as np

# Initialize Monte Carlo engine
mc_engine = MonteCarloEngine(
    num_simulations=500000,
    num_time_steps=252,
    random_seed=42,
    antithetic_variates=True,
    control_variates=True
)

# Price a knock-out barrier call option
barrier_call = mc_engine.price_barrier_option(
    spot_price=100.0,
    strike_price=105.0,
    barrier_level=120.0,     # Knock-out at 120
    time_to_expiry=1.0,      # 1 year
    risk_free_rate=0.05,
    volatility=0.25,
    barrier_type="knock_out",
    option_type="call"
)

print("Barrier Call Option Results:")
print(f"Option Price: ${barrier_call['price']:.4f}")
print(f"95% Confidence Interval: [${barrier_call['ci_lower']:.4f}, ${barrier_call['ci_upper']:.4f}]")
print(f"Probability of barrier hit: {barrier_call['barrier_hit_probability']:.2%}")
print(f"Standard Error: ${barrier_call['standard_error']:.6f}")

# Price an Asian arithmetic call option
asian_call = mc_engine.price_asian_option(
    spot_price=100.0,
    strike_price=100.0,
    time_to_expiry=0.75,     # 9 months
    risk_free_rate=0.04,
    volatility=0.30,
    average_type="arithmetic",
    option_type="call"
)

print("\nAsian Call Option Results:")
print(f"Option Price: ${asian_call['price']:.4f}")
print(f"95% Confidence Interval: [${asian_call['ci_lower']:.4f}, ${asian_call['ci_upper']:.4f}]")
print(f"Effective Volatility: {asian_call['effective_volatility']:.4f}")

# Compare with geometric Asian option
geometric_asian = mc_engine.price_asian_option(
    spot_price=100.0,
    strike_price=100.0,
    time_to_expiry=0.75,
    risk_free_rate=0.04,
    volatility=0.30,
    average_type="geometric",
    option_type="call"
)

print(f"Geometric Asian Price: ${geometric_asian['price']:.4f}")
print(f"Arithmetic vs Geometric Difference: ${asian_call['price'] - geometric_asian['price']:.4f}")
```

### ImpliedVolatilitySolver

Solve for implied volatility using various numerical methods.

#### Syntax

```python
class ImpliedVolatilitySolver:
    def __init__(
        self,
        method: str = "brent",
        tolerance: float = 1e-8,
        max_iterations: int = 100
    )
```

#### Methods

##### solve_iv()

Solve for implied volatility given market price.

```python
def solve_iv(
    self,
    market_price: float,
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: str = "call",
    dividend_yield: float = 0.0
) -> dict:
    """
    Solve for implied volatility.
    
    Parameters:
        market_price (float): Observed market price
        spot_price (float): Current underlying price
        strike_price (float): Option strike price
        time_to_expiry (float): Time to expiration
        risk_free_rate (float): Risk-free rate
        option_type (str): "call" or "put"
        dividend_yield (float): Dividend yield
    
    Returns:
        dict: Implied volatility and convergence info
    """
```

#### Example

```python
from minglib.options import ImpliedVolatilitySolver

# Initialize IV solver
iv_solver = ImpliedVolatilitySolver(
    method="brent",
    tolerance=1e-8,
    max_iterations=50
)

# Market data for options chain
options_data = [
    {'strike': 95, 'market_price': 8.50, 'type': 'call'},
    {'strike': 100, 'market_price': 5.25, 'type': 'call'},
    {'strike': 105, 'market_price': 2.75, 'type': 'call'},
    {'strike': 110, 'market_price': 1.20, 'type': 'call'},
    {'strike': 95, 'market_price': 1.00, 'type': 'put'},
    {'strike': 100, 'market_price': 2.75, 'type': 'put'},
    {'strike': 105, 'market_price': 5.25, 'type': 'put'},
    {'strike': 110, 'market_price': 8.50, 'type': 'put'}
]

# Common parameters
spot_price = 100.0
time_to_expiry = 0.25  # 3 months
risk_free_rate = 0.05

print("Implied Volatility Analysis:")
print("Strike\tType\tMarket Price\tIV\t\tVega")
print("-" * 50)

for option in options_data:
    iv_result = iv_solver.solve_iv(
        market_price=option['market_price'],
        spot_price=spot_price,
        strike_price=option['strike'],
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        option_type=option['type']
    )
    
    print(f"{option['strike']}\t{option['type']}\t${option['market_price']:.2f}\t\t{iv_result['implied_vol']:.4f}\t{iv_result['vega']:.4f}")

# Check for arbitrage opportunities
iv_call_100 = iv_solver.solve_iv(5.25, spot_price, 100, time_to_expiry, risk_free_rate, "call")
iv_put_100 = iv_solver.solve_iv(2.75, spot_price, 100, time_to_expiry, risk_free_rate, "put")

print(f"\nPut-Call IV Parity Check (Strike 100):")
print(f"Call IV: {iv_call_100['implied_vol']:.4f}")
print(f"Put IV: {iv_put_100['implied_vol']:.4f}")
print(f"IV Spread: {abs(iv_call_100['implied_vol'] - iv_put_100['implied_vol']):.4f}")
```

### VolatilitySurfaceBuilder

Build and calibrate volatility surfaces from market data.

#### Syntax

```python
class VolatilitySurfaceBuilder:
    def __init__(
        self,
        interpolation_method: str = "cubic_spline",
        extrapolation_method: str = "linear",
        smoothing_factor: float = 0.1
    )
```

#### Methods

##### build_surface()

Build volatility surface from options data.

```python
def build_surface(
    self,
    options_data: pd.DataFrame,
    spot_price: float,
    risk_free_rate: float
) -> dict:
    """
    Build volatility surface from market data.
    
    Parameters:
        options_data (pd.DataFrame): Options market data
        spot_price (float): Current underlying price
        risk_free_rate (float): Risk-free rate curve
    
    Returns:
        dict: Volatility surface and interpolation functions
    """
```

#### Example

```python
from minglib.options import VolatilitySurfaceBuilder
import pandas as pd
import numpy as np

# Initialize surface builder
surface_builder = VolatilitySurfaceBuilder(
    interpolation_method="cubic_spline",
    extrapolation_method="linear",
    smoothing_factor=0.05
)

# Create sample options market data
strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
expiries = np.array([0.083, 0.25, 0.5, 0.75, 1.0])  # 1m, 3m, 6m, 9m, 1y

# Generate sample implied volatilities (smile/skew pattern)
iv_data = []
for expiry in expiries:
    for strike in strikes:
        moneyness = strike / 100.0
        # Create volatility smile pattern
        iv = 0.20 + 0.1 * (moneyness - 1.0)**2 + 0.02 * np.exp(-10 * expiry)
        iv_data.append({
            'strike': strike,
            'expiry': expiry,
            'implied_vol': iv,
            'market_price': 5.0,  # Placeholder
            'option_type': 'call'
        })

options_df = pd.DataFrame(iv_data)

# Build volatility surface
vol_surface = surface_builder.build_surface(
    options_data=options_df,
    spot_price=100.0,
    risk_free_rate=0.05
)

print("Volatility Surface Analysis:")
print(f"Surface R-squared: {vol_surface['goodness_of_fit']:.4f}")
print(f"Number of data points: {len(options_df)}")
print(f"Interpolation method: {vol_surface['method_used']}")

# Query volatility for specific points
test_points = [
    (102.5, 0.4),  # Strike 102.5, 4.8 months
    (97.5, 0.6),   # Strike 97.5, 7.2 months
    (105.0, 0.2)   # Strike 105, 2.4 months
]

print("\nVolatility Interpolation:")
print("Strike\tExpiry\tImplied Vol")
print("-" * 30)
for strike, expiry in test_points:
    interpolated_vol = vol_surface['interpolation_function'](strike, expiry)
    print(f"{strike}\t{expiry}\t{interpolated_vol:.4f}")
```

## Advanced Features

### ExoticOptionsCalculator

Pricing engine for exotic and structured products.

#### Example

```python
from minglib.options import ExoticOptionsCalculator

exotic_calc = ExoticOptionsCalculator()

# Price a rainbow option on two assets
rainbow_price = exotic_calc.rainbow_option(
    spot_prices=[100.0, 110.0],
    strike_price=105.0,
    correlation=0.6,
    volatilities=[0.25, 0.30],
    time_to_expiry=1.0,
    risk_free_rate=0.05,
    option_type="max_call"  # Call on maximum of two assets
)

print(f"Rainbow Option Price: ${rainbow_price['price']:.4f}")
```

## Performance Considerations

- Use vectorized operations for bulk pricing
- Implement caching for frequently accessed Greeks
- Consider using GPUs for Monte Carlo simulations with large sample sizes
- Use analytical solutions when available (e.g., Black-Scholes vs. Monte Carlo)
- Implement parallel processing for portfolio-level calculations

## Error Handling

```python
try:
    bs_model = BlackScholesModel(100, 105, 0.25, 0.05, 0.20)
    price = bs_model.option_price("call")
except ValueError as e:
    print(f"Invalid parameters: {e}")
except ZeroDivisionError as e:
    print(f"Division by zero in calculation: {e}")
except ConvergenceError as e:
    print(f"Numerical method failed to converge: {e}")
```

## Dependencies

- NumPy >= 1.19.0
- SciPy >= 1.7.0
- Pandas >= 1.3.0
- QuantLib >= 1.20 (optional, for advanced models)
- matplotlib >= 3.3.0 (for visualization)

## See Also

- [Risk Management](risk_management.md)
- [Fixed Income](fixed_income.md)
- [Portfolio Optimization](portfolio_optimization.md)
