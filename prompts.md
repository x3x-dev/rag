
# Prompts

A bunch of prompts used to generate stuff on this project...


## minglib docs generation

This prompt was used to generate the fictional minglib documentation.

```markdown

Please create comprehensive documentation for a fictional Python library called "minglib" that I can use for testing RAG (Retrieval-Augmented Generation) systems. This should be documentation for a library that doesn't exist in your training data.

**Library Overview:**
- **Name:** minglib
- **Purpose:** A Python library implementing quantitative models and utility functions for investment banking and trading activities
- **Target:** Internal use at a fictional investment bank

**Documentation Requirements:**

1. **Structure:** Create 10 separate markdown files, each focusing on a different module/functionality
2. **Organization:** Arrange files in appropriate folders (e.g., `/docs/api/`, `/docs/tutorials/`, `/docs/examples/`)
3. **Content for each document:**
   - Detailed function descriptions
   - Parameter specifications with types
   - Return value documentation
   - Practical code examples showing real-world usage
   - Error handling examples
   - Performance considerations where relevant

**Suggested Modules to Document:**
- Risk management functions
- Portfolio optimization algorithms
- Market data processing utilities
- Options pricing models
- Fixed income calculations
- Credit risk assessment tools
- Performance analytics
- Data validation utilities
- Backtesting framework
- Reporting generators

**Format Requirements:**
- Use proper markdown formatting
- Include syntax-highlighted code blocks
- Add realistic parameter examples
- Include both basic and advanced usage examples
- Make the documentation comprehensive enough to seem like real enterprise software documentation

**Goal:** Generate realistic, detailed technical documentation that an LLM wouldn't have prior knowledge of, suitable for testing document retrieval and question-answering systems.

---

This refined query is much clearer about:
- The specific purpose (RAG testing)
- The fictional nature of the library
- Detailed structural requirements
- Content expectations
- The end goal
@docs/ 
```

# RAG Test Questions

You are an expert in quantitative finance and RAG (Retrieval-Augmented Generation) system evaluation. I need you to create a comprehensive test question set for evaluating a RAG system built on fictional financial library documentation.

**Context:**
I have created detailed documentation for a fictional Python library called "minglib" - a comprehensive quantitative finance toolkit for investment banking. The library includes 10 major modules covering portfolio optimization, risk management, derivatives pricing, fixed income, credit risk, performance analytics, data validation, backtesting, and reporting.

**Task:**
Generate exactly 10 test questions (one per module) that will effectively test a RAG system's ability to retrieve and synthesize information from the minglib documentation. For each question, provide the complete expected output that a well-functioning RAG system should generate.

**Requirements:**

### Question Selection Criteria:
1. **Practical Usage Focus**: Questions should ask "how to" perform specific tasks with minglib
2. **Module Coverage**: Cover all major modules (risk, portfolio, options, fixed_income, credit, performance, validation, backtesting, reporting, market_data)
3. **Varying Complexity**: Mix of simple function usage and more complex conceptual questions
4. **Real-world Relevance**: Questions should reflect actual quantitative finance workflows
5. **Specific Function Names**: Questions should target specific classes/functions from the documentation

### Expected Output Requirements:
For each question, provide a detailed expected output that includes:

1. **Clear Explanation**: Concise but comprehensive explanation of the concept/process
2. **Specific Code Examples**: Working Python code with:
   - Correct import statements
   - Realistic parameter values
   - Proper function calls with accurate syntax
   - Expected output format
3. **Key Details**: 
   - Function parameters and their purposes
   - Return value structures
   - Available options/methods
   - Important notes about functionality
4. **Reference Citation**: End each answer with "**Reference:** [filename].md" indicating the source documentation file

### Question Categories:
Organize questions into logical groups:
- **Portfolio and Risk Management** (5 questions)
- **Options and Fixed Income** (2 questions) 
- **Backtesting and Analytics** (1 question)
- **Data Management** (1 question)
- **Reporting** (1 question)

### Specific Module Targets:
1. **minglib.risk**: VaR calculation, stress testing functions
2. **minglib.portfolio**: Optimization methods, comparison between algorithms
3. **minglib.options**: Black-Scholes pricing
4. **minglib.fixed_income**: Yield curve construction
5. **minglib.backtesting**: Strategy backtesting workflow
6. **minglib.validation**: Data validation processes
7. **minglib.reporting**: Report generation capabilities

### Code Example Standards:
- Use realistic financial data and parameters
- Include proper imports from minglib modules
- Show both basic and slightly advanced usage patterns
- Use meaningful variable names
- Include print statements showing expected outputs
- Follow Python best practices

### Documentation Reference Files:
The expected answers should reference these documentation files:
- risk_management.md
- portfolio_optimization.md  
- options_pricing.md
- fixed_income.md
- backtesting.md
- data_validation.md
- reporting.md

### Format Requirements:
```markdown
# Test Questions

## MingLib RAG Test Questions

[Introduction paragraph explaining purpose]

### [Category Name]

**[Question Number]. [Question Text]**

*Expected Output:*
[Detailed explanation with code examples]

**Reference:** [filename].md

---
```

