"""
Experiment templates for the Prometheus AI Automation Platform.

This module provides pre-defined experiment templates for common scientific
research workflows, making it easier to get started with new experiments.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Callable
import datetime

from .workflow import (
    Experiment, 
    ExperimentParameter, 
    ExperimentMetric, 
    ExperimentStep,
    ExperimentArtifact,
    ParameterType,
    ExperimentStatus
)

# Configure logging
logger = logging.getLogger(__name__)

def create_data_analysis_experiment(
    name: str,
    description: str = "",
    author: str = "",
    data_source: Optional[str] = None
) -> Experiment:
    """
    Create a template experiment for data analysis.
    
    Args:
        name: Experiment name
        description: Experiment description
        author: Experiment author
        data_source: Optional data source path or identifier
        
    Returns:
        Experiment object with data analysis template
    """
    # Create experiment
    experiment = Experiment(
        name=name,
        description=description,
        author=author,
        tags=["data-analysis", "template"]
    )
    
    # Add parameter definitions
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="data_source",
            type=ParameterType.STRING,
            description="Path or identifier for the data source",
            default_value=data_source,
            required=True
        )
    )
    
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="preprocessing_steps",
            type=ParameterType.ARRAY,
            description="List of preprocessing steps to apply",
            default_value=["remove_missing", "normalize"],
            required=False
        )
    )
    
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="analysis_method",
            type=ParameterType.STRING,
            description="Statistical analysis method to use",
            default_value="regression",
            allowed_values=["regression", "classification", "clustering", "dimensionality_reduction"],
            required=True
        )
    )
    
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="visualization_type",
            type=ParameterType.STRING,
            description="Type of visualization to generate",
            default_value="scatter",
            allowed_values=["scatter", "histogram", "heatmap", "box", "line", "bar"],
            required=False
        )
    )
    
    # Add metric definitions
    experiment.add_metric_definition(
        ExperimentMetric(
            name="accuracy",
            description="Model accuracy",
            unit="%",
            higher_is_better=True
        )
    )
    
    experiment.add_metric_definition(
        ExperimentMetric(
            name="processing_time",
            description="Total processing time",
            unit="seconds",
            higher_is_better=False
        )
    )
    
    # Define step functions
    def load_data_step(params, context):
        """Load data from source."""
        context['log']("Loading data from source...")
        # This is a placeholder for actual implementation
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="raw_data",
                    type="dataset",
                    description="Raw data loaded from source",
                    metadata={
                        'source': params['data_source'],
                        'rows': 1000,  # Placeholder
                        'columns': 10  # Placeholder
                    }
                )
            ]
        }
    
    def preprocess_data_step(params, context):
        """Preprocess the data."""
        context['log']("Preprocessing data...")
        # This is a placeholder for actual implementation
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="processed_data",
                    type="dataset",
                    description="Preprocessed data",
                    metadata={
                        'preprocessing_steps': params['preprocessing_steps'],
                        'rows': 950,  # Placeholder
                        'columns': 8  # Placeholder
                    }
                )
            ]
        }
    
    def analyze_data_step(params, context):
        """Analyze the data."""
        context['log']("Analyzing data...")
        # This is a placeholder for actual implementation
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="analysis_results",
                    type="dataset",
                    description="Analysis results",
                    metadata={
                        'analysis_method': params['analysis_method']
                    }
                )
            ],
            'metrics': {
                'accuracy': 85.5  # Placeholder
            }
        }
    
    def visualize_data_step(params, context):
        """Visualize the data and results."""
        context['log']("Visualizing data...")
        # This is a placeholder for actual implementation
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="visualization",
                    type="image",
                    description="Data visualization",
                    metadata={
                        'visualization_type': params['visualization_type']
                    }
                )
            ]
        }
    
    # Add steps to experiment
    experiment.add_step(
        ExperimentStep(
            name="load_data",
            description="Load data from source",
            function=load_data_step,
            parameters={}
        )
    )
    
    experiment.add_step(
        ExperimentStep(
            name="preprocess_data",
            description="Preprocess the data",
            function=preprocess_data_step,
            parameters={},
            depends_on=["load_data"]
        )
    )
    
    experiment.add_step(
        ExperimentStep(
            name="analyze_data",
            description="Analyze the data",
            function=analyze_data_step,
            parameters={},
            depends_on=["preprocess_data"]
        )
    )
    
    experiment.add_step(
        ExperimentStep(
            name="visualize_data",
            description="Visualize the data and results",
            function=visualize_data_step,
            parameters={},
            depends_on=["analyze_data"]
        )
    )
    
    # Set default parameters
    if data_source:
        experiment.set_parameters({
            "data_source": data_source,
            "preprocessing_steps": ["remove_missing", "normalize"],
            "analysis_method": "regression",
            "visualization_type": "scatter"
        })
    
    logger.info(f"Created data analysis experiment template: {name}")
    return experiment


def create_machine_learning_experiment(
    name: str,
    description: str = "",
    author: str = "",
    data_source: Optional[str] = None,
    model_type: str = "classification"
) -> Experiment:
    """
    Create a template experiment for machine learning.
    
    Args:
        name: Experiment name
        description: Experiment description
        author: Experiment author
        data_source: Optional data source path or identifier
        model_type: Type of machine learning model ('classification', 'regression', 'clustering')
        
    Returns:
        Experiment object with machine learning template
    """
    # Create experiment
    experiment = Experiment(
        name=name,
        description=description,
        author=author,
        tags=["machine-learning", model_type, "template"]
    )
    
    # Add parameter definitions
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="data_source",
            type=ParameterType.STRING,
            description="Path or identifier for the data source",
            default_value=data_source,
            required=True
        )
    )
    
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="model_type",
            type=ParameterType.STRING,
            description="Type of machine learning model",
            default_value=model_type,
            allowed_values=["classification", "regression", "clustering"],
            required=True
        )
    )
    
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="features",
            type=ParameterType.ARRAY,
            description="List of features to use",
            default_value=[],
            required=False
        )
    )
    
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="target",
            type=ParameterType.STRING,
            description="Target variable for supervised learning",
            default_value="",
            required=model_type in ["classification", "regression"]
        )
    )
    
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="test_size",
            type=ParameterType.FLOAT,
            description="Proportion of data to use for testing",
            default_value=0.2,
            min_value=0.1,
            max_value=0.5,
            required=False
        )
    )
    
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="hyperparameters",
            type=ParameterType.OBJECT,
            description="Model hyperparameters",
            default_value={},
            required=False
        )
    )
    
    # Add metric definitions
    if model_type == "classification":
        experiment.add_metric_definition(
            ExperimentMetric(
                name="accuracy",
                description="Model accuracy",
                unit="%",
                higher_is_better=True
            )
        )
        
        experiment.add_metric_definition(
            ExperimentMetric(
                name="precision",
                description="Model precision",
                unit="",
                higher_is_better=True
            )
        )
        
        experiment.add_metric_definition(
            ExperimentMetric(
                name="recall",
                description="Model recall",
                unit="",
                higher_is_better=True
            )
        )
        
        experiment.add_metric_definition(
            ExperimentMetric(
                name="f1_score",
                description="Model F1 score",
                unit="",
                higher_is_better=True
            )
        )
    
    elif model_type == "regression":
        experiment.add_metric_definition(
            ExperimentMetric(
                name="mse",
                description="Mean squared error",
                unit="",
                higher_is_better=False
            )
        )
        
        experiment.add_metric_definition(
            ExperimentMetric(
                name="rmse",
                description="Root mean squared error",
                unit="",
                higher_is_better=False
            )
        )
        
        experiment.add_metric_definition(
            ExperimentMetric(
                name="r2",
                description="R-squared",
                unit="",
                higher_is_better=True
            )
        )
    
    elif model_type == "clustering":
        experiment.add_metric_definition(
            ExperimentMetric(
                name="silhouette_score",
                description="Silhouette score",
                unit="",
                higher_is_better=True
            )
        )
        
        experiment.add_metric_definition(
            ExperimentMetric(
                name="inertia",
                description="Inertia (sum of squared distances)",
                unit="",
                higher_is_better=False
            )
        )
    
    experiment.add_metric_definition(
        ExperimentMetric(
            name="training_time",
            description="Model training time",
            unit="seconds",
            higher_is_better=False
        )
    )
    
    # Define step functions
    def load_data_step(params, context):
        """Load data from source."""
        context['log']("Loading data from source...")
        # This is a placeholder for actual implementation
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="raw_data",
                    type="dataset",
                    description="Raw data loaded from source",
                    metadata={
                        'source': params['data_source'],
                        'rows': 1000,  # Placeholder
                        'columns': 10  # Placeholder
                    }
                )
            ]
        }
    
    def preprocess_data_step(params, context):
        """Preprocess the data."""
        context['log']("Preprocessing data...")
        # This is a placeholder for actual implementation
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="processed_data",
                    type="dataset",
                    description="Preprocessed data",
                    metadata={
                        'features': params.get('features', []),
                        'target': params.get('target', ''),
                        'rows': 950,  # Placeholder
                        'columns': 8  # Placeholder
                    }
                )
            ]
        }
    
    def split_data_step(params, context):
        """Split data into training and testing sets."""
        context['log']("Splitting data...")
        # This is a placeholder for actual implementation
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="train_data",
                    type="dataset",
                    description="Training data",
                    metadata={
                        'rows': 760,  # Placeholder
                        'columns': 8  # Placeholder
                    }
                ),
                ExperimentArtifact(
                    name="test_data",
                    type="dataset",
                    description="Testing data",
                    metadata={
                        'rows': 190,  # Placeholder
                        'columns': 8  # Placeholder
                    }
                )
            ]
        }
    
    def train_model_step(params, context):
        """Train the machine learning model."""
        context['log']("Training model...")
        # This is a placeholder for actual implementation
        
        metrics = {}
        if params['model_type'] == "classification":
            metrics = {
                'accuracy': 85.5,  # Placeholder
                'precision': 0.83,  # Placeholder
                'recall': 0.81,  # Placeholder
                'f1_score': 0.82,  # Placeholder
                'training_time': 10.5  # Placeholder
            }
        elif params['model_type'] == "regression":
            metrics = {
                'mse': 25.3,  # Placeholder
                'rmse': 5.03,  # Placeholder
                'r2': 0.78,  # Placeholder
                'training_time': 8.7  # Placeholder
            }
        elif params['model_type'] == "clustering":
            metrics = {
                'silhouette_score': 0.65,  # Placeholder
                'inertia': 120.5,  # Placeholder
                'training_time': 6.2  # Placeholder
            }
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="model",
                    type="model",
                    description="Trained machine learning model",
                    metadata={
                        'model_type': params['model_type'],
                        'hyperparameters': params.get('hyperparameters', {})
                    }
                )
            ],
            'metrics': metrics
        }
    
    def evaluate_model_step(params, context):
        """Evaluate the trained model."""
        context['log']("Evaluating model...")
        # This is a placeholder for actual implementation
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="evaluation_results",
                    type="dataset",
                    description="Model evaluation results",
                    metadata={
                        'model_type': params['model_type']
                    }
                )
            ]
        }
    
    def visualize_results_step(params, context):
        """Visualize the model results."""
        context['log']("Visualizing results...")
        # This is a placeholder for actual implementation
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="performance_visualization",
                    type="image",
                    description="Model performance visualization",
                    metadata={
                        'model_type': params['model_type']
                    }
                )
            ]
        }
    
    # Add steps to experiment
    experiment.add_step(
        ExperimentStep(
            name="load_data",
            description="Load data from source",
            function=load_data_step,
            parameters={}
        )
    )
    
    experiment.add_step(
        ExperimentStep(
            name="preprocess_data",
            description="Preprocess the data",
            function=preprocess_data_step,
            parameters={},
            depends_on=["load_data"]
        )
    )
    
    experiment.add_step(
        ExperimentStep(
            name="split_data",
            description="Split data into training and testing sets",
            function=split_data_step,
            parameters={},
            depends_on=["preprocess_data"]
        )
    )
    
    experiment.add_step(
        ExperimentStep(
            name="train_model",
            description="Train the machine learning model",
            function=train_model_step,
            parameters={},
            depends_on=["split_data"]
        )
    )
    
    experiment.add_step(
        ExperimentStep(
            name="evaluate_model",
            description="Evaluate the trained model",
            function=evaluate_model_step,
            parameters={},
            depends_on=["train_model"]
        )
    )
    
    experiment.add_step(
        ExperimentStep(
            name="visualize_results",
            description="Visualize the model results",
            function=visualize_results_step,
            parameters={},
            depends_on=["evaluate_model"]
        )
    )
    
    # Set default parameters
    default_params = {
        "model_type": model_type,
        "test_size": 0.2
    }
    
    if data_source:
        default_params["data_source"] = data_source
    
    if model_type == "classification":
        default_params["hyperparameters"] = {
            "random_state": 42,
            "max_depth": 5
        }
    elif model_type == "regression":
        default_params["hyperparameters"] = {
            "random_state": 42,
            "alpha": 0.01
        }
    elif model_type == "clustering":
        default_params["hyperparameters"] = {
            "random_state": 42,
            "n_clusters": 3
        }
    
    experiment.set_parameters(default_params)
    
    logger.info(f"Created machine learning experiment template: {name}")
    return experiment


def create_hypothesis_testing_experiment(
    name: str,
    description: str = "",
    author: str = "",
    data_source: Optional[str] = None,
    hypothesis_type: str = "t_test"
) -> Experiment:
    """
    Create a template experiment for hypothesis testing.
    
    Args:
        name: Experiment name
        description: Experiment description
        author: Experiment author
        data_source: Optional data source path or identifier
        hypothesis_type: Type of hypothesis test ('t_test', 'chi_square', 'anova', 'correlation')
        
    Returns:
        Experiment object with hypothesis testing template
    """
    # Create experiment
    experiment = Experiment(
        name=name,
        description=description,
        author=author,
        tags=["hypothesis-testing", hypothesis_type, "template"]
    )
    
    # Add parameter definitions
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="data_source",
            type=ParameterType.STRING,
            description="Path or identifier for the data source",
            default_value=data_source,
            required=True
        )
    )
    
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="hypothesis_type",
            type=ParameterType.STRING,
            description="Type of hypothesis test",
            default_value=hypothesis_type,
            allowed_values=["t_test", "chi_square", "anova", "correlation"],
            required=True
        )
    )
    
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="variables",
            type=ParameterType.ARRAY,
            description="Variables to test",
            default_value=[],
            required=True
        )
    )
    
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="alpha",
            type=ParameterType.FLOAT,
            description="Significance level",
            default_value=0.05,
            min_value=0.001,
            max_value=0.1,
            required=False
        )
    )
    
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="alternative",
            type=ParameterType.STRING,
            description="Alternative hypothesis",
            default_value="two-sided",
            allowed_values=["two-sided", "less", "greater"],
            required=False
        )
    )
    
    # Add metric definitions
    experiment.add_metric_definition(
        ExperimentMetric(
            name="p_value",
            description="P-value of the test",
            unit="",
            higher_is_better=False
        )
    )
    
    experiment.add_metric_definition(
        ExperimentMetric(
            name="test_statistic",
            description="Test statistic",
            unit="",
            higher_is_better=True
        )
    )
    
    experiment.add_metric_definition(
        ExperimentMetric(
            name="effect_size",
            description="Effect size",
            unit="",
            higher_is_better=True
        )
    )
    
    # Define step functions
    def load_data_step(params, context):
        """Load data from source."""
        context['log']("Loading data from source...")
        # This is a placeholder for actual implementation
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="raw_data",
                    type="dataset",
                    description="Raw data loaded from source",
                    metadata={
                        'source': params['data_source'],
                        'rows': 1000,  # Placeholder
                        'columns': 10  # Placeholder
                    }
                )
            ]
        }
    
    def preprocess_data_step(params, context):
        """Preprocess the data."""
        context['log']("Preprocessing data...")
        # This is a placeholder for actual implementation
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="processed_data",
                    type="dataset",
                    description="Preprocessed data",
                    metadata={
                        'variables': params['variables'],
                        'rows': 950,  # Placeholder
                        'columns': len(params['variables'])  # Placeholder
                    }
                )
            ]
        }
    
    def exploratory_analysis_step(params, context):
        """Perform exploratory data analysis."""
        context['log']("Performing exploratory analysis...")
        # This is a placeholder for actual implementation
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="exploratory_analysis",
                    type="dataset",
                    description="Exploratory data analysis results",
                    metadata={
                        'variables': params['variables']
                    }
                ),
                ExperimentArtifact(
                    name="distribution_plots",
                    type="image",
                    description="Distribution plots",
                    metadata={
                        'variables': params['variables']
                    }
                )
            ]
        }
    
    def hypothesis_test_step(params, context):
        """Perform the hypothesis test."""
        context['log']("Performing hypothesis test...")
        # This is a placeholder for actual implementation
        
        # Placeholder metrics based on hypothesis type
        metrics = {}
        if params['hypothesis_type'] == "t_test":
            metrics = {
                'p_value': 0.032,  # Placeholder
                'test_statistic': 2.15,  # Placeholder
                'effect_size': 0.68  # Placeholder (Cohen's d)
            }
        elif params['hypothesis_type'] == "chi_square":
            metrics = {
                'p_value': 0.018,  # Placeholder
                'test_statistic': 11.42,  # Placeholder
                'effect_size': 0.22  # Placeholder (Cramer's V)
            }
        elif params['hypothesis_type'] == "anova":
            metrics = {
                'p_value': 0.008,  # Placeholder
                'test_statistic': 5.63,  # Placeholder
                'effect_size': 0.31  # Placeholder (Eta-squared)
            }
        elif params['hypothesis_type'] == "correlation":
            metrics = {
                'p_value': 0.003,  # Placeholder
                'test_statistic': 0.72,  # Placeholder
                'effect_size': 0.72  # Placeholder (r value is the effect size)
            }
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="test_results",
                    type="dataset",
                    description="Hypothesis test results",
                    metadata={
                        'hypothesis_type': params['hypothesis_type'],
                        'variables': params['variables'],
                        'alpha': params['alpha'],
                        'alternative': params.get('alternative', 'two-sided')
                    }
                )
            ],
            'metrics': metrics
        }
    
    def visualize_results_step(params, context):
        """Visualize the test results."""
        context['log']("Visualizing results...")
        # This is a placeholder for actual implementation
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="results_visualization",
                    type="image",
                    description="Test results visualization",
                    metadata={
                        'hypothesis_type': params['hypothesis_type']
                    }
                )
            ]
        }
    
    def interpret_results_step(params, context):
        """Interpret the test results."""
        context['log']("Interpreting results...")
        # This is a placeholder for actual implementation
        
        # Get p-value from metrics
        p_value = context['metrics'].get('p_value', 1.0)
        alpha = params['alpha']
        
        # Determine if null hypothesis is rejected
        reject_null = p_value < alpha
        
        interpretation = f"The p-value ({p_value:.4f}) is {'less' if reject_null else 'greater'} than the significance level ({alpha}). "
        interpretation += f"Therefore, we {'reject' if reject_null else 'fail to reject'} the null hypothesis."
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="interpretation",
                    type="text",
                    description="Interpretation of test results",
                    content=interpretation,
                    metadata={
                        'hypothesis_type': params['hypothesis_type'],
                        'p_value': p_value,
                        'alpha': alpha,
                        'reject_null': reject_null
                    }
                )
            ]
        }
    
    # Add steps to experiment
    experiment.add_step(
        ExperimentStep(
            name="load_data",
            description="Load data from source",
            function=load_data_step,
            parameters={}
        )
    )
    
    experiment.add_step(
        ExperimentStep(
            name="preprocess_data",
            description="Preprocess the data",
            function=preprocess_data_step,
            parameters={},
            depends_on=["load_data"]
        )
    )
    
    experiment.add_step(
        ExperimentStep(
            name="exploratory_analysis",
            description="Perform exploratory data analysis",
            function=exploratory_analysis_step,
            parameters={},
            depends_on=["preprocess_data"]
        )
    )
    
    experiment.add_step(
        ExperimentStep(
            name="hypothesis_test",
            description="Perform the hypothesis test",
            function=hypothesis_test_step,
            parameters={},
            depends_on=["exploratory_analysis"]
        )
    )
    
    experiment.add_step(
        ExperimentStep(
            name="visualize_results",
            description="Visualize the test results",
            function=visualize_results_step,
            parameters={},
            depends_on=["hypothesis_test"]
        )
    )
    
    experiment.add_step(
        ExperimentStep(
            name="interpret_results",
            description="Interpret the test results",
            function=interpret_results_step,
            parameters={},
            depends_on=["hypothesis_test"]
        )
    )
    
    # Set default parameters
    default_params = {
        "hypothesis_type": hypothesis_type,
        "alpha": 0.05,
        "alternative": "two-sided"
    }
    
    if data_source:
        default_params["data_source"] = data_source
    
    if hypothesis_type == "t_test":
        default_params["variables"] = ["group_a", "group_b"]
    elif hypothesis_type == "chi_square":
        default_params["variables"] = ["category_a", "category_b"]
    elif hypothesis_type == "anova":
        default_params["variables"] = ["group_a", "group_b", "group_c"]
    elif hypothesis_type == "correlation":
        default_params["variables"] = ["variable_a", "variable_b"]
    
    experiment.set_parameters(default_params)
    
    logger.info(f"Created hypothesis testing experiment template: {name}")
    return experiment


def create_simulation_experiment(
    name: str,
    description: str = "",
    author: str = "",
    simulation_type: str = "monte_carlo"
) -> Experiment:
    """
    Create a template experiment for simulation.
    
    Args:
        name: Experiment name
        description: Experiment description
        author: Experiment author
        simulation_type: Type of simulation ('monte_carlo', 'agent_based', 'system_dynamics')
        
    Returns:
        Experiment object with simulation template
    """
    # Create experiment
    experiment = Experiment(
        name=name,
        description=description,
        author=author,
        tags=["simulation", simulation_type, "template"]
    )
    
    # Add parameter definitions
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="simulation_type",
            type=ParameterType.STRING,
            description="Type of simulation",
            default_value=simulation_type,
            allowed_values=["monte_carlo", "agent_based", "system_dynamics"],
            required=True
        )
    )
    
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="num_iterations",
            type=ParameterType.INTEGER,
            description="Number of simulation iterations",
            default_value=1000,
            min_value=10,
            max_value=100000,
            required=True
        )
    )
    
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="random_seed",
            type=ParameterType.INTEGER,
            description="Random seed for reproducibility",
            default_value=42,
            required=False
        )
    )
    
    experiment.add_parameter_definition(
        ExperimentParameter(
            name="model_parameters",
            type=ParameterType.OBJECT,
            description="Simulation model parameters",
            default_value={},
            required=True
        )
    )
    
    # Add metric definitions
    experiment.add_metric_definition(
        ExperimentMetric(
            name="simulation_time",
            description="Total simulation time",
            unit="seconds",
            higher_is_better=False
        )
    )
    
    experiment.add_metric_definition(
        ExperimentMetric(
            name="convergence_rate",
            description="Rate of convergence",
            unit="%",
            higher_is_better=True
        )
    )
    
    # Define step functions
    def setup_simulation_step(params, context):
        """Set up the simulation environment."""
        context['log']("Setting up simulation environment...")
        # This is a placeholder for actual implementation
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="simulation_config",
                    type="config",
                    description="Simulation configuration",
                    metadata={
                        'simulation_type': params['simulation_type'],
                        'num_iterations': params['num_iterations'],
                        'random_seed': params['random_seed'],
                        'model_parameters': params['model_parameters']
                    }
                )
            ]
        }
    
    def run_simulation_step(params, context):
        """Run the simulation."""
        context['log']("Running simulation...")
        # This is a placeholder for actual implementation
        
        # Placeholder metrics
        metrics = {
            'simulation_time': 45.3,  # Placeholder
            'convergence_rate': 98.5  # Placeholder
        }
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="simulation_results",
                    type="dataset",
                    description="Simulation results",
                    metadata={
                        'simulation_type': params['simulation_type'],
                        'num_iterations': params['num_iterations']
                    }
                )
            ],
            'metrics': metrics
        }
    
    def analyze_results_step(params, context):
        """Analyze the simulation results."""
        context['log']("Analyzing simulation results...")
        # This is a placeholder for actual implementation
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="analysis_results",
                    type="dataset",
                    description="Analysis of simulation results",
                    metadata={
                        'simulation_type': params['simulation_type']
                    }
                )
            ]
        }
    
    def visualize_results_step(params, context):
        """Visualize the simulation results."""
        context['log']("Visualizing simulation results...")
        # This is a placeholder for actual implementation
        
        return {
            'artifacts': [
                ExperimentArtifact(
                    name="results_visualization",
                    type="image",
                    description="Visualization of simulation results",
                    metadata={
                        'simulation_type': params['simulation_type']
                    }
                )
            ]
        }
    
    # Add steps to experiment
    experiment.add_step(
        ExperimentStep(
            name="setup_simulation",
            description="Set up the simulation environment",
            function=setup_simulation_step,
            parameters={}
        )
    )
    
    experiment.add_step(
        ExperimentStep(
            name="run_simulation",
            description="Run the simulation",
            function=run_simulation_step,
            parameters={},
            depends_on=["setup_simulation"]
        )
    )
    
    experiment.add_step(
        ExperimentStep(
            name="analyze_results",
            description="Analyze the simulation results",
            function=analyze_results_step,
            parameters={},
            depends_on=["run_simulation"]
        )
    )
    
    experiment.add_step(
        ExperimentStep(
            name="visualize_results",
            description="Visualize the simulation results",
            function=visualize_results_step,
            parameters={},
            depends_on=["analyze_results"]
        )
    )
    
    # Set default parameters based on simulation type
    model_parameters = {}
    
    if simulation_type == "monte_carlo":
        model_parameters = {
            "distribution": "normal",
            "mean": 0.0,
            "std_dev": 1.0
        }
    elif simulation_type == "agent_based":
        model_parameters = {
            "num_agents": 100,
            "interaction_radius": 5.0,
            "behavior_rules": {
                "rule1": 0.5,
                "rule2": 0.3,
                "rule3": 0.2
            }
        }
    elif simulation_type == "system_dynamics":
        model_parameters = {
            "initial_state": {
                "variable1": 10.0,
                "variable2": 5.0
            },
            "rate_constants": {
                "k1": 0.1,
                "k2": 0.05
            },
            "time_step": 0.1,
            "total_time": 100.0
        }
    
    experiment.set_parameters({
        "simulation_type": simulation_type,
        "num_iterations": 1000,
        "random_seed": 42,
        "model_parameters": model_parameters
    })
    
    logger.info(f"Created simulation experiment template: {name}")
    return experiment


def get_available_templates() -> Dict[str, str]:
    """
    Get a dictionary of available experiment templates.
    
    Returns:
        Dictionary mapping template names to descriptions
    """
    return {
        "data_analysis": "Template for data analysis experiments",
        "machine_learning": "Template for machine learning experiments",
        "hypothesis_testing": "Template for hypothesis testing experiments",
        "simulation": "Template for simulation experiments"
    }


def create_experiment_from_template(
    template_name: str,
    experiment_name: str,
    description: str = "",
    author: str = "",
    **kwargs
) -> Optional[Experiment]:
    """
    Create an experiment from a template.
    
    Args:
        template_name: Name of the template to use
        experiment_name: Name for the new experiment
        description: Experiment description
        author: Experiment author
        **kwargs: Additional parameters for the template
        
    Returns:
        Created experiment, or None if template not found
    """
    if template_name == "data_analysis":
        return create_data_analysis_experiment(
            name=experiment_name,
            description=description,
            author=author,
            data_source=kwargs.get("data_source")
        )
    
    elif template_name == "machine_learning":
        return create_machine_learning_experiment(
            name=experiment_name,
            description=description,
            author=author,
            data_source=kwargs.get("data_source"),
            model_type=kwargs.get("model_type", "classification")
        )
    
    elif template_name == "hypothesis_testing":
        return create_hypothesis_testing_experiment(
            name=experiment_name,
            description=description,
            author=author,
            data_source=kwargs.get("data_source"),
            hypothesis_type=kwargs.get("hypothesis_type", "t_test")
        )
    
    elif template_name == "simulation":
        return create_simulation_experiment(
            name=experiment_name,
            description=description,
            author=author,
            simulation_type=kwargs.get("simulation_type", "monte_carlo")
        )
    
    else:
        logger.error(f"Unknown template: {template_name}")
        return None
