<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Customer Churn Prediction Project - Copilot Instructions

## Project Context
This is a machine learning project focused on predicting customer churn for a telecommunications company. The project follows a complete data science workflow from EDA to model deployment.

## Key Guidelines for Code Generation

### Data Science Best Practices
- Always include proper data validation and error handling
- Use descriptive variable names and add comments for complex logic
- Follow pandas and scikit-learn best practices
- Include visualization code with proper labels and titles
- Ensure reproducibility with random seeds

### Machine Learning Workflow
- Start with baseline models before complex ones
- Always split data before any preprocessing to avoid data leakage
- Use cross-validation for model evaluation
- Include proper metrics evaluation (accuracy, precision, recall, F1)
- Generate confusion matrices and feature importance plots

### Code Structure
- Use modular code with functions for reusable operations
- Include docstrings for functions
- Use type hints where appropriate
- Follow PEP 8 style guidelines
- Add markdown cells to explain analysis steps

### Visualization Guidelines
- Use consistent color schemes across plots
- Include proper titles, axis labels, and legends
- Save important plots to outputs/plots/ directory
- Use both matplotlib and seaborn for different visualization needs

### Model Management
- Save trained models to outputs/models/ directory using joblib
- Include model versioning and metadata
- Generate model interpretation visualizations
- Document model performance metrics

### Business Focus
- Always relate technical findings to business impact
- Include actionable insights and recommendations
- Focus on interpretability for business stakeholders
- Highlight cost-benefit analysis where relevant
