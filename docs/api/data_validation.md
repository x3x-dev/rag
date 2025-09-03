# Data Validation Module

## Overview

The `minglib.validation` module provides comprehensive data validation, quality assessment, and reconciliation tools for financial data. This module ensures data integrity across trading systems, risk management platforms, and regulatory reporting pipelines.

## Installation

```python
from minglib.validation import (
    DataValidator,
    QualityAssessment,
    ReconciliationEngine,
    SchemaValidator,
    TimeSeriesValidator,
    RegulatoryReportValidator
)
```

## Core Classes

### DataValidator

Primary data validation engine with configurable rules and automated quality checks.

#### Syntax

```python
class DataValidator:
    def __init__(
        self,
        validation_rules: dict = None,
        error_threshold: float = 0.05,
        auto_correction: bool = False
    )
```

#### Parameters

- **validation_rules** (dict, optional): Custom validation rules configuration. Default: None
- **error_threshold** (float, optional): Maximum allowed error rate. Default: 0.05
- **auto_correction** (bool, optional): Enable automatic error correction. Default: False

#### Methods

##### validate_dataset()

Perform comprehensive validation on financial datasets.

```python
def validate_dataset(
    self,
    data: pd.DataFrame,
    schema: dict = None,
    business_rules: list = None,
    cross_validation: bool = True
) -> dict:
    """
    Validate complete dataset against schema and business rules.
    
    Parameters:
        data (pd.DataFrame): Dataset to validate
        schema (dict): Expected data schema with types and constraints
        business_rules (list): List of business rule functions
        cross_validation (bool): Enable cross-field validation
    
    Returns:
        dict: Validation results with errors, warnings, and quality scores
    """
```

##### validate_prices()

Specialized validation for price and market data.

```python
def validate_prices(
    self,
    price_data: pd.DataFrame,
    previous_close: pd.Series = None,
    market_hours: dict = None,
    corporate_actions: pd.DataFrame = None
) -> dict:
    """
    Validate price data for accuracy and consistency.
    
    Parameters:
        price_data (pd.DataFrame): OHLCV price data
        previous_close (pd.Series): Previous trading session close prices
        market_hours (dict): Trading hours for validation
        corporate_actions (pd.DataFrame): Corporate actions for adjustment
    
    Returns:
        dict: Price validation results
    """
```

#### Example

```python
from minglib.validation import DataValidator
import pandas as pd
import numpy as np

# Create sample market data with intentional errors
np.random.seed(42)
dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Generate sample data with errors
sample_data = []
for date in dates:
    for symbol in symbols:
        base_price = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500, 'AMZN': 3000, 'TSLA': 800}[symbol]
        
        # Add some realistic price movement
        price_change = np.random.normal(0, 0.02) * base_price
        close_price = base_price + price_change
        
        # Introduce intentional errors
        if np.random.random() < 0.02:  # 2% error rate
            if np.random.random() < 0.5:
                close_price = 0  # Invalid zero price
            else:
                close_price *= 10  # Price spike error
        
        open_price = close_price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.003)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.003)))
        
        # Introduce OHLC relationship errors
        if np.random.random() < 0.01:  # 1% OHLC error rate
            high_price = low_price * 0.9  # High < Low error
        
        volume = np.random.lognormal(15, 1)
        if np.random.random() < 0.005:  # Occasional missing volume
            volume = None
        
        sample_data.append({
            'date': date,
            'symbol': symbol,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

price_data = pd.DataFrame(sample_data)

# Define validation schema
price_schema = {
    'date': {'type': 'datetime', 'required': True},
    'symbol': {'type': 'string', 'required': True, 'max_length': 10},
    'open': {'type': 'float', 'required': True, 'min_value': 0.01, 'max_value': 10000},
    'high': {'type': 'float', 'required': True, 'min_value': 0.01, 'max_value': 10000},
    'low': {'type': 'float', 'required': True, 'min_value': 0.01, 'max_value': 10000},
    'close': {'type': 'float', 'required': True, 'min_value': 0.01, 'max_value': 10000},
    'volume': {'type': 'float', 'required': False, 'min_value': 0}
}

# Define business rules
def validate_ohlc_relationship(row):
    """Validate OHLC price relationships"""
    errors = []
    if row['high'] < row['low']:
        errors.append(f"High price ({row['high']}) < Low price ({row['low']})")
    if row['high'] < max(row['open'], row['close']):
        errors.append(f"High price < max(Open, Close)")
    if row['low'] > min(row['open'], row['close']):
        errors.append(f"Low price > min(Open, Close)")
    return errors

def validate_price_movement(row, previous_close=None):
    """Validate reasonable price movements"""
    errors = []
    if previous_close is not None:
        price_change = abs(row['close'] - previous_close) / previous_close
        if price_change > 0.20:  # 20% daily movement threshold
            errors.append(f"Excessive price movement: {price_change:.2%}")
    return errors

business_rules = [validate_ohlc_relationship, validate_price_movement]

# Initialize data validator
validator = DataValidator(
    error_threshold=0.10,  # Allow up to 10% errors
    auto_correction=True
)

# Perform validation
validation_results = validator.validate_dataset(
    data=price_data,
    schema=price_schema,
    business_rules=business_rules,
    cross_validation=True
)

print("Data Validation Results:")
print("=" * 50)
print(f"Total Records: {validation_results['total_records']}")
print(f"Valid Records: {validation_results['valid_records']}")
print(f"Error Records: {validation_results['error_records']}")
print(f"Warning Records: {validation_results['warning_records']}")
print(f"Data Quality Score: {validation_results['quality_score']:.2%}")

print(f"\nValidation Summary by Type:")
for error_type, count in validation_results['error_summary'].items():
    print(f"  {error_type}: {count} errors")

# Detailed error analysis
if validation_results['detailed_errors']:
    print(f"\nSample Errors (First 10):")
    for i, error in enumerate(validation_results['detailed_errors'][:10]):
        print(f"  {i+1}. Row {error['row_index']}: {error['error_description']}")

# Auto-correction results
if validation_results['auto_corrections']:
    print(f"\nAuto-Corrections Applied: {len(validation_results['auto_corrections'])}")
    for correction in validation_results['auto_corrections'][:5]:
        print(f"  Row {correction['row_index']}: {correction['correction_description']}")
```

### QualityAssessment

Comprehensive data quality assessment with scoring and trend analysis.

#### Syntax

```python
class QualityAssessment:
    def __init__(
        self,
        quality_dimensions: list = ["completeness", "accuracy", "consistency", "timeliness"],
        scoring_method: str = "weighted"
    )
```

#### Methods

##### assess_quality()

Perform multi-dimensional quality assessment.

```python
def assess_quality(
    self,
    data: pd.DataFrame,
    reference_data: pd.DataFrame = None,
    time_column: str = 'date'
) -> dict:
    """
    Assess data quality across multiple dimensions.
    
    Parameters:
        data (pd.DataFrame): Data to assess
        reference_data (pd.DataFrame): Reference data for comparison
        time_column (str): Column name for temporal analysis
    
    Returns:
        dict: Quality assessment scores and detailed analysis
    """
```

#### Example

```python
from minglib.validation import QualityAssessment

# Initialize quality assessment
quality_assessor = QualityAssessment(
    quality_dimensions=["completeness", "accuracy", "consistency", "timeliness", "uniqueness"],
    scoring_method="weighted"
)

# Assess data quality
quality_results = quality_assessor.assess_quality(
    data=price_data,
    time_column='date'
)

print("Data Quality Assessment:")
print("=" * 40)
print(f"Overall Quality Score: {quality_results['overall_score']:.2%}")

print(f"\nQuality Dimensions:")
for dimension, score in quality_results['dimension_scores'].items():
    print(f"  {dimension.capitalize()}: {score:.2%}")

print(f"\nQuality Trends (by Date):")
quality_trends = quality_results['temporal_quality']
print(f"  Best Quality Day: {quality_trends['best_day']} ({quality_trends['best_score']:.2%})")
print(f"  Worst Quality Day: {quality_trends['worst_day']} ({quality_trends['worst_score']:.2%})")
print(f"  Quality Volatility: {quality_trends['quality_volatility']:.2%}")

# Quality by symbol
symbol_quality = quality_results['group_quality']['symbol']
print(f"\nQuality by Symbol:")
for symbol, qual_score in sorted(symbol_quality.items(), key=lambda x: x[1], reverse=True):
    print(f"  {symbol}: {qual_score:.2%}")
```

### ReconciliationEngine

Reconcile data between different sources and identify discrepancies.

#### Syntax

```python
class ReconciliationEngine:
    def __init__(
        self,
        tolerance_levels: dict = None,
        reconciliation_rules: list = None,
        match_criteria: list = ['date', 'symbol']
    )
```

#### Methods

##### reconcile_sources()

Reconcile data between multiple sources.

```python
def reconcile_sources(
    self,
    source1_data: pd.DataFrame,
    source2_data: pd.DataFrame,
    source1_name: str = "Source1",
    source2_name: str = "Source2",
    tolerance: float = 0.01
) -> dict:
    """
    Reconcile data between two sources.
    
    Parameters:
        source1_data (pd.DataFrame): First data source
        source2_data (pd.DataFrame): Second data source
        source1_name (str): Name of first source
        source2_name (str): Name of second source
        tolerance (float): Tolerance for numerical comparisons
    
    Returns:
        dict: Reconciliation results with breaks and statistics
    """
```

#### Example

```python
from minglib.validation import ReconciliationEngine

# Create second data source with slight differences
price_data_source2 = price_data.copy()

# Introduce some reconciliation breaks
np.random.seed(123)
break_indices = np.random.choice(len(price_data_source2), size=50, replace=False)

for idx in break_indices:
    if np.random.random() < 0.3:
        # Price difference
        price_data_source2.loc[idx, 'close'] *= 1.001  # 0.1% difference
    elif np.random.random() < 0.3:
        # Volume difference  
        price_data_source2.loc[idx, 'volume'] *= 1.05  # 5% difference
    else:
        # Missing record
        price_data_source2.loc[idx, 'close'] = np.nan

# Initialize reconciliation engine
recon_engine = ReconciliationEngine(
    tolerance_levels={'close': 0.001, 'volume': 0.02},  # 0.1% for prices, 2% for volume
    match_criteria=['date', 'symbol']
)

# Perform reconciliation
recon_results = recon_engine.reconcile_sources(
    source1_data=price_data,
    source2_data=price_data_source2,
    source1_name="Primary_Feed",
    source2_name="Backup_Feed",
    tolerance=0.001
)

print("Data Reconciliation Results:")
print("=" * 50)
print(f"Total Records Compared: {recon_results['total_comparisons']}")
print(f"Perfect Matches: {recon_results['perfect_matches']}")
print(f"Within Tolerance: {recon_results['within_tolerance']}")
print(f"Breaks Identified: {recon_results['breaks_identified']}")
print(f"Missing in Source 1: {recon_results['missing_source1']}")
print(f"Missing in Source 2: {recon_results['missing_source2']}")
print(f"Match Rate: {recon_results['match_rate']:.2%}")

print(f"\nBreaks by Field:")
for field, count in recon_results['breaks_by_field'].items():
    print(f"  {field}: {count} breaks")

# Sample breaks
if recon_results['break_details']:
    print(f"\nSample Breaks (First 5):")
    for i, break_detail in enumerate(recon_results['break_details'][:5]):
        print(f"  {i+1}. {break_detail['date']} {break_detail['symbol']}: "
              f"{break_detail['field']} = {break_detail['source1_value']:.6f} vs "
              f"{break_detail['source2_value']:.6f} (Diff: {break_detail['difference']:.6f})")
```

### TimeSeriesValidator

Specialized validation for time series financial data.

#### Syntax

```python
class TimeSeriesValidator:
    def __init__(
        self,
        frequency_detection: bool = True,
        gap_tolerance: str = "1D",
        seasonality_check: bool = True
    )
```

#### Methods

##### validate_time_series()

Validate time series data for gaps, frequency, and patterns.

```python
def validate_time_series(
    self,
    time_series_data: pd.DataFrame,
    datetime_column: str = 'date',
    value_columns: list = None,
    expected_frequency: str = None
) -> dict:
    """
    Validate time series data structure and patterns.
    
    Parameters:
        time_series_data (pd.DataFrame): Time series data
        datetime_column (str): Name of datetime column
        value_columns (list): Columns to validate for patterns
        expected_frequency (str): Expected data frequency
    
    Returns:
        dict: Time series validation results
    """
```

#### Example

```python
from minglib.validation import TimeSeriesValidator

# Initialize time series validator
ts_validator = TimeSeriesValidator(
    frequency_detection=True,
    gap_tolerance="1D",
    seasonality_check=True
)

# Prepare time series data
ts_data = price_data.pivot(index='date', columns='symbol', values='close')

# Validate time series
ts_results = ts_validator.validate_time_series(
    time_series_data=ts_data,
    datetime_column='date',
    value_columns=['AAPL', 'MSFT', 'GOOGL'],
    expected_frequency='D'
)

print("Time Series Validation Results:")
print("=" * 45)
print(f"Detected Frequency: {ts_results['detected_frequency']}")
print(f"Expected Frequency: {ts_results['expected_frequency']}")
print(f"Frequency Match: {ts_results['frequency_match']}")
print(f"Date Range: {ts_results['start_date']} to {ts_results['end_date']}")
print(f"Total Time Points: {ts_results['total_periods']}")
print(f"Missing Time Points: {ts_results['missing_periods']}")
print(f"Data Completeness: {ts_results['completeness_rate']:.2%}")

if ts_results['gaps_detected']:
    print(f"\nDate Gaps Found:")
    for gap in ts_results['gap_details'][:5]:
        print(f"  Gap from {gap['start_date']} to {gap['end_date']} ({gap['gap_size']} periods)")

print(f"\nStatistical Validation:")
for column, stats in ts_results['column_statistics'].items():
    print(f"  {column}:")
    print(f"    Outliers Detected: {stats['outliers_count']}")
    print(f"    Stationarity (p-value): {stats['stationarity_pvalue']:.4f}")
    print(f"    Autocorrelation (lag-1): {stats['autocorrelation_lag1']:.4f}")
```

### SchemaValidator

Validate data against predefined schemas for regulatory compliance.

#### Example

```python
from minglib.validation import SchemaValidator

# Define regulatory schema (e.g., for trade reporting)
trade_schema = {
    'trade_id': {'type': 'string', 'required': True, 'pattern': r'^TRD\d{8}$'},
    'trade_date': {'type': 'datetime', 'required': True},
    'symbol': {'type': 'string', 'required': True, 'max_length': 12},
    'quantity': {'type': 'integer', 'required': True, 'min_value': 1},
    'price': {'type': 'float', 'required': True, 'min_value': 0.01},
    'counterparty': {'type': 'string', 'required': True},
    'trader_id': {'type': 'string', 'required': True, 'pattern': r'^T\d{4}$'}
}

schema_validator = SchemaValidator(schema=trade_schema)

# Sample trade data
trade_data = pd.DataFrame({
    'trade_id': ['TRD20240101', 'TRD20240102', 'INVALID_ID', 'TRD20240104'],
    'trade_date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
    'symbol': ['AAPL', 'MSFT', 'GOOGL', 'A' * 15],  # Last one too long
    'quantity': [100, 200, -50, 300],  # Negative quantity error
    'price': [150.25, 300.50, 0.00, 250.75],  # Zero price error
    'counterparty': ['BANK_A', 'HEDGE_B', 'BANK_C', ''],  # Empty counterparty
    'trader_id': ['T1001', 'T1002', 'INVALID', 'T1004']  # Invalid format
})

schema_results = schema_validator.validate(trade_data)

print("Schema Validation Results:")
print(f"Schema Compliance Rate: {schema_results['compliance_rate']:.2%}")
print(f"Total Violations: {schema_results['total_violations']}")

for field, violations in schema_results['violations_by_field'].items():
    if violations:
        print(f"\n{field} violations:")
        for violation in violations[:3]:
            print(f"  Row {violation['row']}: {violation['description']}")
```

## Utility Functions

### data_profiling()

Generate comprehensive data profiling reports.

```python
from minglib.validation import data_profiling

profile_report = data_profiling(
    data=price_data,
    include_correlations=True,
    generate_plots=False
)
```

### anomaly_detection()

Detect anomalies in financial time series.

```python
from minglib.validation import anomaly_detection

anomalies = anomaly_detection(
    data=ts_data,
    method="isolation_forest",
    contamination=0.01
)
```

### data_lineage_tracking()

Track data lineage and transformations.

```python
from minglib.validation import data_lineage_tracking

lineage = data_lineage_tracking(
    source_data=price_data,
    transformations=['clean_nulls', 'calculate_returns', 'outlier_removal'],
    output_data=cleaned_data
)
```

## Performance Considerations

- Use vectorized operations for large-scale validation
- Implement parallel processing for multi-dataset validation
- Cache validation results for repeated checks
- Use sampling techniques for very large datasets
- Optimize memory usage for streaming data validation

## Error Handling

```python
try:
    validation_results = validator.validate_dataset(data, schema)
except SchemaValidationError as e:
    print(f"Schema validation failed: {e}")
except DataFormatError as e:
    print(f"Data format error: {e}")
except ValidationTimeoutError as e:
    print(f"Validation timeout: {e}")
```

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.19.0
- scipy >= 1.7.0
- scikit-learn >= 0.24.0 (for anomaly detection)
- jsonschema >= 3.2.0 (for schema validation)

## See Also

- [Market Data](market_data.md)
- [Risk Management](risk_management.md)
- [Reporting](reporting.md)
