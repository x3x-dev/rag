# Backtesting Framework Module

## Overview

The `minglib.backtesting` module provides a comprehensive backtesting framework for quantitative trading strategies, portfolio optimization algorithms, and risk management models. This framework supports vectorized backtesting, event-driven simulation, and advanced performance attribution.

## Installation

```python
from minglib.backtesting import (
    BacktestEngine,
    StrategyBase,
    PortfolioSimulator,
    TransactionCostModel,
    RiskManager,
    PerformanceAnalyzer
)
```

## Core Classes

### BacktestEngine

Main backtesting engine supporting multiple strategy types and execution modes.

#### Syntax

```python
class BacktestEngine:
    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 1000000.0,
        execution_mode: str = "vectorized",
        benchmark: str = "SPY"
    )
```

#### Parameters

- **start_date** (str): Backtest start date in 'YYYY-MM-DD' format
- **end_date** (str): Backtest end date in 'YYYY-MM-DD' format
- **initial_capital** (float, optional): Starting portfolio value. Default: 1000000.0
- **execution_mode** (str, optional): Execution mode. Options: "vectorized", "event_driven". Default: "vectorized"
- **benchmark** (str, optional): Benchmark symbol for comparison. Default: "SPY"

#### Methods

##### run_backtest()

Execute backtest for given strategy and data.

```python
def run_backtest(
    self,
    strategy: StrategyBase,
    data: pd.DataFrame,
    transaction_costs: TransactionCostModel = None,
    risk_manager: RiskManager = None
) -> dict:
    """
    Run complete backtest simulation.
    
    Parameters:
        strategy (StrategyBase): Trading strategy to test
        data (pd.DataFrame): Market data for backtesting
        transaction_costs (TransactionCostModel): Transaction cost model
        risk_manager (RiskManager): Risk management rules
    
    Returns:
        dict: Comprehensive backtest results
    """
```

##### optimize_parameters()

Optimize strategy parameters using grid search or genetic algorithms.

```python
def optimize_parameters(
    self,
    strategy_class: type,
    data: pd.DataFrame,
    parameter_ranges: dict,
    optimization_metric: str = "sharpe_ratio",
    method: str = "grid_search"
) -> dict:
    """
    Optimize strategy parameters.
    
    Parameters:
        strategy_class (type): Strategy class to optimize
        data (pd.DataFrame): Historical data
        parameter_ranges (dict): Parameter ranges for optimization
        optimization_metric (str): Metric to optimize
        method (str): Optimization method
    
    Returns:
        dict: Optimization results with best parameters
    """
```

#### Example

```python
from minglib.backtesting import BacktestEngine, StrategyBase
import pandas as pd
import numpy as np

# Create sample market data
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
n_days = len(dates)

# Generate correlated price series for multiple assets
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
n_assets = len(symbols)

# Create correlation matrix
correlation_matrix = np.full((n_assets, n_assets), 0.3)
np.fill_diagonal(correlation_matrix, 1.0)

# Generate returns
daily_returns = np.random.multivariate_normal(
    mean=[0.0008, 0.0006, 0.0009, 0.0007, 0.0012],  # Different expected returns
    cov=correlation_matrix * 0.02**2,  # 2% daily volatility
    size=n_days
)

# Convert to price data
initial_prices = [150, 300, 2500, 3000, 800]
price_data = pd.DataFrame(index=dates, columns=symbols)

for i, symbol in enumerate(symbols):
    price_data[symbol] = initial_prices[i] * np.cumprod(1 + daily_returns[:, i])

# Add volume data
volume_data = pd.DataFrame(
    np.random.lognormal(15, 0.5, (n_days, n_assets)),
    index=dates,
    columns=[f"{symbol}_volume" for symbol in symbols]
)

# Combine price and volume data
market_data = pd.concat([price_data, volume_data], axis=1)

# Initialize backtest engine
backtest_engine = BacktestEngine(
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=1000000.0,
    execution_mode="vectorized",
    benchmark="SPY"
)

# Define a simple momentum strategy
class MomentumStrategy(StrategyBase):
    def __init__(self, lookback_period=20, top_n=3):
        super().__init__()
        self.lookback_period = lookback_period
        self.top_n = top_n
        self.name = f"Momentum_{lookback_period}_{top_n}"
    
    def generate_signals(self, data):
        """Generate trading signals based on momentum"""
        signals = pd.DataFrame(index=data.index, columns=symbols, data=0.0)
        
        # Calculate rolling returns
        price_cols = [col for col in data.columns if col in symbols]
        returns = data[price_cols].pct_change(self.lookback_period)
        
        # Rank assets by momentum each day
        for date in returns.index[self.lookback_period:]:
            momentum_ranks = returns.loc[date].rank(ascending=False)
            top_assets = momentum_ranks.nsmallest(self.top_n).index
            
            # Equal weight top momentum assets
            for asset in top_assets:
                signals.loc[date, asset] = 1.0 / self.top_n
        
        return signals
    
    def calculate_positions(self, signals, data, current_positions=None):
        """Calculate target positions from signals"""
        return signals

# Create momentum strategy instance
momentum_strategy = MomentumStrategy(lookback_period=20, top_n=3)

# Define transaction cost model
from minglib.backtesting import TransactionCostModel

transaction_costs = TransactionCostModel(
    commission_rate=0.001,  # 10 bps commission
    bid_ask_spread=0.0005,  # 5 bps spread
    market_impact_factor=0.0001  # 1 bp market impact
)

# Define risk manager
from minglib.backtesting import RiskManager

risk_manager = RiskManager(
    max_position_size=0.10,  # Max 10% per position
    max_sector_exposure=0.30,  # Max 30% per sector
    max_leverage=1.0,  # No leverage
    stop_loss_threshold=0.05  # 5% stop loss
)

# Run backtest
backtest_results = backtest_engine.run_backtest(
    strategy=momentum_strategy,
    data=market_data,
    transaction_costs=transaction_costs,
    risk_manager=risk_manager
)

print("Backtest Results Summary:")
print("=" * 50)
print(f"Strategy: {momentum_strategy.name}")
print(f"Period: {backtest_results['start_date']} to {backtest_results['end_date']}")
print(f"Initial Capital: ${backtest_results['initial_capital']:,.0f}")
print(f"Final Portfolio Value: ${backtest_results['final_value']:,.0f}")
print(f"Total Return: {backtest_results['total_return']:.2%}")
print(f"Annualized Return: {backtest_results['annualized_return']:.2%}")
print(f"Annualized Volatility: {backtest_results['annualized_volatility']:.2%}")
print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}")
print(f"Maximum Drawdown: {backtest_results['max_drawdown']:.2%}")
print(f"Calmar Ratio: {backtest_results['calmar_ratio']:.4f}")

print(f"\nTrading Statistics:")
print(f"Total Trades: {backtest_results['total_trades']}")
print(f"Winning Trades: {backtest_results['winning_trades']} ({backtest_results['win_rate']:.1%})")
print(f"Average Trade: {backtest_results['average_trade']:.2%}")
print(f"Best Trade: {backtest_results['best_trade']:.2%}")
print(f"Worst Trade: {backtest_results['worst_trade']:.2%}")
print(f"Profit Factor: {backtest_results['profit_factor']:.2f}")

print(f"\nCost Analysis:")
print(f"Total Transaction Costs: ${backtest_results['total_transaction_costs']:,.0f}")
print(f"Cost as % of PnL: {backtest_results['cost_ratio']:.2%}")
print(f"Average Daily Turnover: {backtest_results['average_turnover']:.1%}")

# Benchmark comparison
if 'benchmark_comparison' in backtest_results:
    bench = backtest_results['benchmark_comparison']
    print(f"\nBenchmark Comparison:")
    print(f"Benchmark Return: {bench['benchmark_return']:.2%}")
    print(f"Active Return: {bench['active_return']:.2%}")
    print(f"Information Ratio: {bench['information_ratio']:.4f}")
    print(f"Beta: {bench['beta']:.4f}")
    print(f"Alpha: {bench['alpha']:.4f}")
```

### StrategyBase

Base class for implementing trading strategies.

#### Syntax

```python
class StrategyBase:
    def __init__(self):
        self.name = "BaseStrategy"
        self.parameters = {}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from market data"""
        raise NotImplementedError
    
    def calculate_positions(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate position sizes from signals"""
        raise NotImplementedError
```

### PortfolioSimulator

Simulate portfolio evolution with realistic trading mechanics.

#### Syntax

```python
class PortfolioSimulator:
    def __init__(
        self,
        initial_capital: float,
        rebalancing_frequency: str = "daily",
        slippage_model: str = "linear"
    )
```

#### Methods

##### simulate_portfolio()

Simulate portfolio performance over time.

```python
def simulate_portfolio(
    self,
    target_weights: pd.DataFrame,
    price_data: pd.DataFrame,
    transaction_costs: TransactionCostModel = None
) -> dict:
    """
    Simulate portfolio evolution.
    
    Parameters:
        target_weights (pd.DataFrame): Target portfolio weights over time
        price_data (pd.DataFrame): Asset price data
        transaction_costs (TransactionCostModel): Cost model
    
    Returns:
        dict: Portfolio simulation results
    """
```

#### Example

```python
from minglib.backtesting import PortfolioSimulator

# Extract target weights from momentum strategy
target_weights = momentum_strategy.generate_signals(market_data)

# Initialize portfolio simulator
portfolio_sim = PortfolioSimulator(
    initial_capital=1000000.0,
    rebalancing_frequency="daily",
    slippage_model="square_root"
)

# Run simulation
simulation_results = portfolio_sim.simulate_portfolio(
    target_weights=target_weights,
    price_data=price_data,
    transaction_costs=transaction_costs
)

print("Portfolio Simulation Results:")
print(f"Final Portfolio Value: ${simulation_results['final_value']:,.0f}")
print(f"Total Transaction Costs: ${simulation_results['total_costs']:,.0f}")
print(f"Portfolio Turnover: {simulation_results['average_turnover']:.1%}")

# Analyze holdings over time
holdings_analysis = simulation_results['holdings_analysis']
print(f"\nHoldings Analysis:")
print(f"Average Number of Positions: {holdings_analysis['avg_positions']:.1f}")
print(f"Max Concentration: {holdings_analysis['max_concentration']:.1%}")
print(f"Average Position Size: {holdings_analysis['avg_position_size']:.1%}")
```

### TransactionCostModel

Model various transaction costs including commissions, spreads, and market impact.

#### Example

```python
from minglib.backtesting import TransactionCostModel

# Define comprehensive cost model
cost_model = TransactionCostModel(
    commission_rate=0.0005,  # 5 bps commission
    bid_ask_spread=0.0003,   # 3 bps average spread
    market_impact_factor=0.0002,  # Market impact scaling
    price_impact_function="square_root",  # Impact function
    minimum_commission=1.0   # Minimum commission per trade
)

# Calculate costs for a specific trade
trade_cost = cost_model.calculate_trade_cost(
    trade_value=100000,  # $100K trade
    daily_volume=5000000,  # $5M average daily volume
    trade_direction="buy"
)

print(f"Trade Cost Analysis:")
print(f"Commission: ${trade_cost['commission']:.2f}")
print(f"Bid-Ask Spread: ${trade_cost['spread_cost']:.2f}")
print(f"Market Impact: ${trade_cost['market_impact']:.2f}")
print(f"Total Cost: ${trade_cost['total_cost']:.2f}")
print(f"Cost as % of Trade: {trade_cost['cost_percentage']:.3%}")
```

### Strategy Optimization

#### Parameter Optimization Example

```python
# Define parameter ranges for momentum strategy
parameter_ranges = {
    'lookback_period': [10, 15, 20, 25, 30, 40, 50],
    'top_n': [1, 2, 3, 4, 5]
}

# Run parameter optimization
optimization_results = backtest_engine.optimize_parameters(
    strategy_class=MomentumStrategy,
    data=market_data,
    parameter_ranges=parameter_ranges,
    optimization_metric="sharpe_ratio",
    method="grid_search"
)

print("Parameter Optimization Results:")
print("=" * 45)
print(f"Best Parameters: {optimization_results['best_parameters']}")
print(f"Best Sharpe Ratio: {optimization_results['best_score']:.4f}")
print(f"Total Combinations Tested: {optimization_results['total_combinations']}")

print(f"\nTop 5 Parameter Combinations:")
for i, result in enumerate(optimization_results['top_results'][:5]):
    params = result['parameters']
    score = result['score']
    print(f"{i+1}. Lookback: {params['lookback_period']}, Top N: {params['top_n']}, "
          f"Sharpe: {score:.4f}")

# Optimization surface analysis
optimization_surface = optimization_results['parameter_surface']
print(f"\nParameter Sensitivity Analysis:")
print(f"Most Sensitive Parameter: {optimization_surface['most_sensitive_param']}")
print(f"Sensitivity Score: {optimization_surface['sensitivity_score']:.4f}")
```

### Walk-Forward Analysis

#### Example

```python
from minglib.backtesting import WalkForwardAnalyzer

# Initialize walk-forward analyzer
wf_analyzer = WalkForwardAnalyzer(
    training_period=252,  # 1 year training
    testing_period=63,   # 3 months testing
    reoptimization_frequency=63  # Reoptimize every quarter
)

# Run walk-forward analysis
wf_results = wf_analyzer.run_analysis(
    strategy_class=MomentumStrategy,
    data=market_data,
    parameter_ranges=parameter_ranges,
    optimization_metric="sharpe_ratio"
)

print("Walk-Forward Analysis Results:")
print("=" * 50)
print(f"Number of Walk-Forward Periods: {wf_results['num_periods']}")
print(f"Average In-Sample Sharpe: {wf_results['avg_in_sample_sharpe']:.4f}")
print(f"Average Out-of-Sample Sharpe: {wf_results['avg_out_of_sample_sharpe']:.4f}")
print(f"Sharpe Decay: {wf_results['sharpe_decay']:.4f}")
print(f"Consistency Score: {wf_results['consistency_score']:.4f}")

# Parameter stability analysis
param_stability = wf_results['parameter_stability']
print(f"\nParameter Stability:")
for param, stability in param_stability.items():
    print(f"  {param}: {stability:.4f}")
```

## Advanced Features

### Multi-Strategy Backtesting

```python
from minglib.backtesting import MultiStrategyEngine

# Define multiple strategies
strategies = [
    MomentumStrategy(lookback_period=20, top_n=3),
    MeanReversionStrategy(lookback_period=5, threshold=2.0),
    TrendFollowingStrategy(fast_ma=10, slow_ma=30)
]

# Multi-strategy backtest
multi_engine = MultiStrategyEngine(allocation_method="equal_weight")
multi_results = multi_engine.run_backtest(
    strategies=strategies,
    data=market_data,
    rebalancing_frequency="monthly"
)

print("Multi-Strategy Results:")
for strategy_name, results in multi_results['individual_results'].items():
    print(f"{strategy_name}: Sharpe = {results['sharpe_ratio']:.4f}")

print(f"Combined Portfolio Sharpe: {multi_results['combined_sharpe']:.4f}")
```

### Regime-Aware Backtesting

```python
from minglib.backtesting import RegimeAwareEngine

regime_engine = RegimeAwareEngine(
    regime_indicators=['vix', 'yield_curve_slope', 'credit_spreads'],
    regime_detection_method="hidden_markov"
)

regime_results = regime_engine.run_backtest(
    strategy=momentum_strategy,
    data=market_data,
    regime_data=regime_indicators
)

print("Regime-Aware Backtest Results:")
for regime, performance in regime_results['regime_performance'].items():
    print(f"Regime {regime}: Return = {performance['return']:.2%}, "
          f"Sharpe = {performance['sharpe']:.4f}")
```

## Performance Considerations

- Use vectorized operations for faster backtesting
- Implement parallel processing for parameter optimization
- Cache intermediate calculations for repeated use
- Use memory-efficient data structures for large datasets
- Consider using compiled libraries (Numba/Cython) for critical paths

## Error Handling

```python
try:
    backtest_results = backtest_engine.run_backtest(strategy, data)
except InsufficientDataError as e:
    print(f"Insufficient data for backtest: {e}")
except StrategyError as e:
    print(f"Strategy implementation error: {e}")
except ParameterError as e:
    print(f"Invalid parameter values: {e}")
```

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.19.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0 (for optimization)

## See Also

- [Performance Analytics](performance_analytics.md)
- [Portfolio Optimization](portfolio_optimization.md)
- [Risk Management](risk_management.md)
