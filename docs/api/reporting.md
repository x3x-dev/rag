# Reporting Generators Module

## Overview

The `minglib.reporting` module provides comprehensive report generation capabilities for investment management, risk reporting, regulatory compliance, and client communications. This module supports automated report generation, customizable templates, and multi-format output.

## Installation

```python
from minglib.reporting import (
    ReportGenerator,
    TemplateManager,
    RiskReportBuilder,
    PerformanceReportBuilder,
    RegulatoryReportBuilder,
    ClientReportBuilder
)
```

## Core Classes

### ReportGenerator

Main report generation engine with support for multiple output formats and automated scheduling.

#### Syntax

```python
class ReportGenerator:
    def __init__(
        self,
        output_formats: list = ["pdf", "html", "excel"],
        template_directory: str = "./templates",
        output_directory: str = "./reports",
        branding: dict = None
    )
```

#### Parameters

- **output_formats** (list, optional): Supported output formats. Options: "pdf", "html", "excel", "powerpoint". Default: ["pdf", "html", "excel"]
- **template_directory** (str, optional): Directory containing report templates. Default: "./templates"
- **output_directory** (str, optional): Directory for generated reports. Default: "./reports"
- **branding** (dict, optional): Company branding configuration. Default: None

#### Methods

##### generate_report()

Generate a report from data and template.

```python
def generate_report(
    self,
    report_type: str,
    data: dict,
    template_name: str = None,
    output_filename: str = None,
    parameters: dict = None
) -> dict:
    """
    Generate a report from template and data.
    
    Parameters:
        report_type (str): Type of report to generate
        data (dict): Data dictionary for report population
        template_name (str): Template file name
        output_filename (str): Output file name
        parameters (dict): Additional report parameters
    
    Returns:
        dict: Report generation results and file paths
    """
```

##### schedule_report()

Schedule automatic report generation.

```python
def schedule_report(
    self,
    report_config: dict,
    schedule: str,
    distribution_list: list = None
) -> dict:
    """
    Schedule automated report generation.
    
    Parameters:
        report_config (dict): Report configuration
        schedule (str): Cron-style schedule string
        distribution_list (list): Email distribution list
    
    Returns:
        dict: Scheduling confirmation and job ID
    """
```

#### Example

```python
from minglib.reporting import ReportGenerator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize report generator
report_gen = ReportGenerator(
    output_formats=["pdf", "html", "excel"],
    template_directory="./report_templates",
    output_directory="./generated_reports",
    branding={
        'company_name': 'ABC Asset Management',
        'logo_path': './assets/logo.png',
        'color_scheme': 'blue',
        'footer_text': 'Confidential - For Authorized Use Only'
    }
)

# Prepare sample portfolio data
portfolio_data = {
    'portfolio_name': 'Growth Equity Fund',
    'as_of_date': '2024-01-31',
    'fund_nav': 125.67,
    'total_assets': 2.5e9,  # $2.5B
    'inception_date': '2020-01-01',
    'benchmark': 'S&P 500',
    'performance_data': {
        'mtd_return': 0.0234,
        'qtd_return': 0.0234,
        'ytd_return': 0.0234,
        'one_year': 0.1567,
        'three_year': 0.1234,
        'since_inception': 0.1123,
        'benchmark_one_year': 0.1345,
        'alpha': 0.0234,
        'beta': 1.05,
        'sharpe_ratio': 1.67,
        'information_ratio': 0.45,
        'tracking_error': 0.0567,
        'max_drawdown': -0.0789
    },
    'holdings_summary': {
        'number_of_holdings': 47,
        'top_10_concentration': 0.456,
        'cash_percentage': 0.023,
        'avg_market_cap': 125.6e9,
        'median_market_cap': 67.3e9
    },
    'sector_allocation': {
        'Technology': 0.34,
        'Healthcare': 0.18,
        'Financials': 0.16,
        'Consumer Discretionary': 0.12,
        'Industrials': 0.08,
        'Communication Services': 0.07,
        'Materials': 0.03,
        'Cash': 0.02
    },
    'top_holdings': [
        {'symbol': 'AAPL', 'name': 'Apple Inc.', 'weight': 0.087, 'sector': 'Technology'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corp.', 'weight': 0.074, 'sector': 'Technology'},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'weight': 0.063, 'sector': 'Consumer Discretionary'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'weight': 0.059, 'sector': 'Communication Services'},
        {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'weight': 0.045, 'sector': 'Consumer Discretionary'},
        {'symbol': 'NVDA', 'name': 'NVIDIA Corp.', 'weight': 0.042, 'sector': 'Technology'},
        {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'weight': 0.038, 'sector': 'Communication Services'},
        {'symbol': 'UNH', 'name': 'UnitedHealth Group', 'weight': 0.035, 'sector': 'Healthcare'},
        {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'weight': 0.032, 'sector': 'Healthcare'},
        {'symbol': 'V', 'name': 'Visa Inc.', 'weight': 0.029, 'sector': 'Financials'}
    ]
}

# Generate monthly performance report
performance_report = report_gen.generate_report(
    report_type="monthly_performance",
    data=portfolio_data,
    template_name="monthly_performance_template.html",
    output_filename="growth_equity_fund_jan_2024",
    parameters={
        'include_benchmarks': True,
        'include_attribution': True,
        'include_risk_metrics': True,
        'chart_style': 'professional'
    }
)

print("Performance Report Generated:")
print("=" * 40)
print(f"Report Type: {performance_report['report_type']}")
print(f"Generated Files:")
for format_type, file_path in performance_report['output_files'].items():
    print(f"  {format_type.upper()}: {file_path}")

print(f"Generation Time: {performance_report['generation_time']:.2f} seconds")
print(f"File Sizes:")
for format_type, size in performance_report['file_sizes'].items():
    print(f"  {format_type.upper()}: {size:.1f} KB")

# Schedule monthly report generation
monthly_schedule = {
    'report_type': 'monthly_performance',
    'template_name': 'monthly_performance_template.html',
    'data_source': 'portfolio_database',
    'output_formats': ['pdf', 'excel'],
    'parameters': {
        'include_benchmarks': True,
        'include_attribution': True
    }
}

schedule_result = report_gen.schedule_report(
    report_config=monthly_schedule,
    schedule="0 9 1 * *",  # 9 AM on the 1st of every month
    distribution_list=['portfolio_managers@abc.com', 'clients@abc.com']
)

print(f"\nScheduled Report:")
print(f"Job ID: {schedule_result['job_id']}")
print(f"Next Run: {schedule_result['next_run_date']}")
print(f"Distribution List: {len(schedule_result['recipients'])} recipients")
```

### RiskReportBuilder

Specialized builder for risk management reports.

#### Syntax

```python
class RiskReportBuilder:
    def __init__(
        self,
        risk_metrics: list = ["var", "expected_shortfall", "stress_tests"],
        confidence_levels: list = [0.95, 0.99],
        time_horizons: list = [1, 10, 250]
    )
```

#### Methods

##### build_var_report()

Generate Value at Risk report.

```python
def build_var_report(
    self,
    portfolio_data: pd.DataFrame,
    risk_factors: pd.DataFrame,
    methodology: str = "historical_simulation"
) -> dict:
    """
    Build comprehensive VaR report.
    
    Parameters:
        portfolio_data (pd.DataFrame): Portfolio positions and exposures
        risk_factors (pd.DataFrame): Risk factor returns
        methodology (str): VaR calculation methodology
    
    Returns:
        dict: VaR report data and visualizations
    """
```

##### build_stress_test_report()

Generate stress testing report.

```python
def build_stress_test_report(
    self,
    portfolio_data: pd.DataFrame,
    stress_scenarios: dict,
    historical_scenarios: list = None
) -> dict:
    """
    Build stress testing report.
    
    Parameters:
        portfolio_data (pd.DataFrame): Portfolio data
        stress_scenarios (dict): Stress scenario definitions
        historical_scenarios (list): Historical stress events
    
    Returns:
        dict: Stress testing report data
    """
```

#### Example

```python
from minglib.reporting import RiskReportBuilder
import numpy as np

# Initialize risk report builder
risk_builder = RiskReportBuilder(
    risk_metrics=["var", "expected_shortfall", "stress_tests", "concentration"],
    confidence_levels=[0.95, 0.99, 0.999],
    time_horizons=[1, 5, 10, 22]  # 1day, 1week, 2weeks, 1month
)

# Create sample portfolio positions
portfolio_positions = pd.DataFrame({
    'asset_id': ['EQUITY_001', 'EQUITY_002', 'BOND_001', 'BOND_002', 'OPTION_001'],
    'asset_name': ['Tech Growth Fund', 'Value Equity ETF', '10Y Treasury', 'Corp Bond Fund', 'SPY Call Option'],
    'asset_class': ['Equity', 'Equity', 'Fixed Income', 'Fixed Income', 'Derivatives'],
    'market_value': [150e6, 120e6, 80e6, 60e6, 10e6],  # $150M, $120M, etc.
    'weight': [0.35, 0.28, 0.19, 0.14, 0.02],
    'beta': [1.2, 0.8, 0.1, 0.3, 2.5],
    'duration': [None, None, 7.2, 4.8, None],
    'delta': [None, None, None, None, 0.6],
    'gamma': [None, None, None, None, 0.15]
})

# Generate risk factor returns (sample data)
np.random.seed(42)
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
risk_factors = pd.DataFrame({
    'equity_market': np.random.normal(0.0008, 0.015, len(dates)),
    'credit_spread': np.random.normal(0.0001, 0.003, len(dates)),
    'interest_rate': np.random.normal(0.0000, 0.002, len(dates)),
    'volatility': np.random.normal(0.0000, 0.05, len(dates)),
    'fx_usd': np.random.normal(0.0000, 0.008, len(dates))
}, index=dates)

# Build VaR report
var_report = risk_builder.build_var_report(
    portfolio_data=portfolio_positions,
    risk_factors=risk_factors,
    methodology="monte_carlo"
)

print("Value at Risk Report:")
print("=" * 30)
print(f"Portfolio Value: ${var_report['portfolio_value']:,.0f}")
print(f"VaR Methodology: {var_report['methodology']}")

print(f"\nVaR by Time Horizon:")
for horizon, var_data in var_report['var_by_horizon'].items():
    print(f"  {horizon}-day VaR:")
    for conf_level, var_value in var_data.items():
        var_pct = var_value / var_report['portfolio_value'] * 100
        print(f"    {conf_level:.0%}: ${var_value:,.0f} ({var_pct:.2f}%)")

print(f"\nComponent VaR:")
for asset, component_var in var_report['component_var'].items():
    print(f"  {asset}: ${component_var:,.0f}")

# Define stress scenarios
stress_scenarios = {
    'equity_crash': {
        'equity_market': -0.30,
        'credit_spread': 0.02,
        'volatility': 2.0,
        'description': '2008-style equity market crash'
    },
    'interest_rate_shock': {
        'interest_rate': 0.02,
        'credit_spread': 0.01,
        'description': 'Sudden 200bp rate increase'
    },
    'credit_crisis': {
        'credit_spread': 0.05,
        'equity_market': -0.20,
        'description': 'Credit market freeze'
    }
}

# Build stress test report
stress_report = risk_builder.build_stress_test_report(
    portfolio_data=portfolio_positions,
    stress_scenarios=stress_scenarios,
    historical_scenarios=['covid_2020', 'gfc_2008', 'dotcom_2000']
)

print(f"\nStress Testing Results:")
print("=" * 30)
for scenario_name, results in stress_report['scenario_results'].items():
    pnl = results['portfolio_pnl']
    pnl_pct = pnl / var_report['portfolio_value'] * 100
    print(f"{scenario_name}: ${pnl:,.0f} ({pnl_pct:+.2f}%)")

print(f"\nWorst Case Scenario: {stress_report['worst_scenario']['name']}")
print(f"Worst Case Loss: ${stress_report['worst_scenario']['loss']:,.0f}")
```

### PerformanceReportBuilder

Specialized builder for performance and attribution reports.

#### Example

```python
from minglib.reporting import PerformanceReportBuilder

perf_builder = PerformanceReportBuilder(
    attribution_levels=['sector', 'security', 'currency'],
    benchmark_comparisons=['primary', 'peer_group', 'custom'],
    time_periods=['1M', '3M', '6M', '1Y', '3Y', '5Y', 'ITD']
)

# Build comprehensive performance report
performance_report = perf_builder.build_performance_report(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    attribution_data=attribution_data,
    holdings_data=holdings_data
)

print("Performance Attribution Report:")
print(f"Active Return (1Y): {performance_report['active_return_1y']:.2%}")
print(f"Information Ratio: {performance_report['information_ratio']:.3f}")

# Attribution breakdown
attribution = performance_report['attribution_analysis']
print(f"\nAttribution Analysis:")
print(f"Allocation Effect: {attribution['allocation_effect']:.2%}")
print(f"Selection Effect: {attribution['selection_effect']:.2%}")
print(f"Interaction Effect: {attribution['interaction_effect']:.2%}")
```

### RegulatoryReportBuilder

Generate regulatory compliance reports.

#### Example

```python
from minglib.reporting import RegulatoryReportBuilder

reg_builder = RegulatoryReportBuilder(
    regulations=['UCITS', 'AIFMD', 'MiFID', 'SEC_N-PORT'],
    frequency='monthly',
    validation_rules=True
)

# Build regulatory report
reg_report = reg_builder.build_regulatory_report(
    fund_data=fund_data,
    regulation='SEC_N-PORT',
    reporting_period='2024-01'
)

print("Regulatory Report Generated:")
print(f"Regulation: {reg_report['regulation']}")
print(f"Validation Status: {reg_report['validation_passed']}")
print(f"Filing Ready: {reg_report['filing_ready']}")

if reg_report['validation_errors']:
    print("Validation Errors:")
    for error in reg_report['validation_errors']:
        print(f"  - {error}")
```

### ClientReportBuilder

Generate client-facing reports and factsheets.

#### Example

```python
from minglib.reporting import ClientReportBuilder

client_builder = ClientReportBuilder(
    client_segments=['institutional', 'retail', 'private_wealth'],
    customization_level='high',
    language_support=['en', 'de', 'fr']
)

# Build client quarterly report
client_report = client_builder.build_quarterly_report(
    client_data=client_data,
    portfolio_performance=performance_data,
    market_commentary=market_commentary,
    client_segment='institutional'
)

print("Client Report Generated:")
print(f"Report Type: {client_report['report_type']}")
print(f"Client Segment: {client_report['client_segment']}")
print(f"Customization Applied: {client_report['customization_level']}")
```

## Template Management

### TemplateManager

Manage report templates and customizations.

#### Example

```python
from minglib.reporting import TemplateManager

template_mgr = TemplateManager(
    template_directory="./templates",
    version_control=True,
    approval_workflow=True
)

# Create new template
template_mgr.create_template(
    template_name="risk_dashboard",
    template_type="html",
    sections=['summary', 'var_analysis', 'stress_tests', 'exposures'],
    styling={'theme': 'corporate', 'color_scheme': 'blue'}
)

# Version control
template_mgr.version_template(
    template_name="risk_dashboard",
    changes="Added new stress test scenarios",
    author="risk_team@abc.com"
)
```

## Automated Distribution

### ReportDistributor

Automate report distribution via email, file shares, and portals.

#### Example

```python
from minglib.reporting import ReportDistributor

distributor = ReportDistributor(
    email_config={'smtp_server': 'smtp.abc.com', 'port': 587},
    file_share_config={'server': 'reports.abc.com', 'path': '/reports'},
    portal_config={'api_endpoint': 'https://portal.abc.com/api'}
)

# Distribute report
distribution_result = distributor.distribute_report(
    report_files=['risk_report.pdf', 'risk_report.xlsx'],
    distribution_list=[
        {'email': 'cio@abc.com', 'format': 'pdf'},
        {'email': 'risk_committee@abc.com', 'format': 'excel'},
        {'portal': 'client_portal', 'client_ids': ['CLIENT_001', 'CLIENT_002']}
    ]
)

print("Distribution Results:")
print(f"Emails Sent: {distribution_result['emails_sent']}")
print(f"Portal Uploads: {distribution_result['portal_uploads']}")
print(f"File Share Updates: {distribution_result['file_share_updates']}")
```

## Performance Considerations

- Use efficient templating engines (Jinja2) for large reports
- Implement caching for frequently generated reports
- Use asynchronous processing for batch report generation
- Optimize image and chart generation for faster rendering
- Consider using CDN for report asset distribution

## Error Handling

```python
try:
    report = report_gen.generate_report(report_type, data)
except TemplateNotFoundError as e:
    print(f"Template not found: {e}")
except DataValidationError as e:
    print(f"Data validation failed: {e}")
except ReportGenerationError as e:
    print(f"Report generation failed: {e}")
```

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.19.0
- jinja2 >= 2.11.0 (for templating)
- matplotlib >= 3.3.0 (for charts)
- reportlab >= 3.5.0 (for PDF generation)
- openpyxl >= 3.0.0 (for Excel reports)
- weasyprint >= 52.0 (for HTML to PDF conversion)

## See Also

- [Performance Analytics](performance_analytics.md)
- [Risk Management](risk_management.md)
- [Data Validation](data_validation.md)
