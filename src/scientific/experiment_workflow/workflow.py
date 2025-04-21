"""
Experiment workflow system for the Prometheus AI Automation Platform.

This module provides classes and functions for defining, executing, and managing
scientific experiments with reproducibility, traceability, and collaboration features.
"""

import os
import logging
import uuid
import json
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
import copy
import hashlib
import time
import traceback
import threading
import queue

# Configure logging
logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    """Status of an experiment."""
    DRAFT = "draft"
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    ARCHIVED = "archived"

class ParameterType(Enum):
    """Type of experiment parameter."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    FILE = "file"
    DATASET = "dataset"

@dataclass
class ExperimentParameter:
    """Definition of an experiment parameter."""
    name: str
    type: ParameterType
    description: str = ""
    default_value: Any = None
    required: bool = True
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a parameter value against its definition.
        
        Args:
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if value is required but not provided
        if self.required and value is None:
            return False, f"Parameter '{self.name}' is required"
        
        # If value is not provided but not required, it's valid
        if value is None and not self.required:
            return True, None
        
        # Validate type
        if self.type == ParameterType.STRING:
            if not isinstance(value, str):
                return False, f"Parameter '{self.name}' must be a string"
        elif self.type == ParameterType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"Parameter '{self.name}' must be an integer"
        elif self.type == ParameterType.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, f"Parameter '{self.name}' must be a number"
        elif self.type == ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Parameter '{self.name}' must be a boolean"
        elif self.type == ParameterType.ARRAY:
            if not isinstance(value, list):
                return False, f"Parameter '{self.name}' must be an array"
        elif self.type == ParameterType.OBJECT:
            if not isinstance(value, dict):
                return False, f"Parameter '{self.name}' must be an object"
        elif self.type == ParameterType.FILE:
            if not isinstance(value, str) or not os.path.exists(value):
                return False, f"Parameter '{self.name}' must be a valid file path"
        elif self.type == ParameterType.DATASET:
            if not isinstance(value, (str, pd.DataFrame)):
                return False, f"Parameter '{self.name}' must be a dataset identifier or DataFrame"
        
        # Validate numeric constraints
        if self.type in [ParameterType.INTEGER, ParameterType.FLOAT]:
            if self.min_value is not None and value < self.min_value:
                return False, f"Parameter '{self.name}' must be at least {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Parameter '{self.name}' must be at most {self.max_value}"
        
        # Validate allowed values
        if self.allowed_values is not None and value not in self.allowed_values:
            return False, f"Parameter '{self.name}' must be one of {self.allowed_values}"
        
        return True, None

@dataclass
class ExperimentMetric:
    """Definition of an experiment metric."""
    name: str
    description: str = ""
    unit: str = ""
    higher_is_better: bool = True

@dataclass
class ExperimentArtifact:
    """Definition of an experiment artifact."""
    name: str
    type: str
    description: str = ""
    path: Optional[str] = None
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)

@dataclass
class ExperimentStep:
    """Definition of an experiment step."""
    name: str
    description: str = ""
    function: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"
    started_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    artifacts: List[ExperimentArtifact] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None

@dataclass
class Experiment:
    """
    Definition of a scientific experiment.
    
    An experiment consists of metadata, parameters, steps, metrics, and artifacts.
    It can be executed, saved, loaded, and shared with other researchers.
    """
    name: str
    description: str = ""
    author: str = ""
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "1.0.0"
    status: ExperimentStatus = ExperimentStatus.DRAFT
    tags: List[str] = field(default_factory=list)
    
    # Experiment definition
    parameter_definitions: List[ExperimentParameter] = field(default_factory=list)
    metric_definitions: List[ExperimentMetric] = field(default_factory=list)
    
    # Experiment execution
    parameters: Dict[str, Any] = field(default_factory=dict)
    steps: List[ExperimentStep] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[ExperimentArtifact] = field(default_factory=list)
    
    # Execution metadata
    started_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    
    def add_parameter_definition(self, param: ExperimentParameter) -> None:
        """
        Add a parameter definition to the experiment.
        
        Args:
            param: Parameter definition
        """
        # Check if parameter with same name already exists
        for existing_param in self.parameter_definitions:
            if existing_param.name == param.name:
                raise ValueError(f"Parameter '{param.name}' already exists")
        
        self.parameter_definitions.append(param)
        logger.info(f"Added parameter definition '{param.name}' to experiment '{self.name}'")
    
    def add_metric_definition(self, metric: ExperimentMetric) -> None:
        """
        Add a metric definition to the experiment.
        
        Args:
            metric: Metric definition
        """
        # Check if metric with same name already exists
        for existing_metric in self.metric_definitions:
            if existing_metric.name == metric.name:
                raise ValueError(f"Metric '{metric.name}' already exists")
        
        self.metric_definitions.append(metric)
        logger.info(f"Added metric definition '{metric.name}' to experiment '{self.name}'")
    
    def add_step(self, step: ExperimentStep) -> None:
        """
        Add a step to the experiment.
        
        Args:
            step: Experiment step
        """
        # Check if step with same name already exists
        for existing_step in self.steps:
            if existing_step.name == step.name:
                raise ValueError(f"Step '{step.name}' already exists")
        
        # Validate step dependencies
        for dep in step.depends_on:
            if not any(s.name == dep for s in self.steps):
                raise ValueError(f"Step '{step.name}' depends on non-existent step '{dep}'")
        
        self.steps.append(step)
        logger.info(f"Added step '{step.name}' to experiment '{self.name}'")
    
    def set_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """
        Set parameter values for the experiment.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            List of validation error messages (empty if all parameters are valid)
        """
        errors = []
        
        # Validate parameters against definitions
        for param_def in self.parameter_definitions:
            if param_def.name in parameters:
                value = parameters[param_def.name]
                is_valid, error = param_def.validate(value)
                if not is_valid:
                    errors.append(error)
            elif param_def.required:
                errors.append(f"Required parameter '{param_def.name}' is missing")
            elif param_def.default_value is not None:
                # Use default value if not provided
                parameters[param_def.name] = param_def.default_value
        
        # Check for extra parameters not in definitions
        for param_name in parameters:
            if not any(param_def.name == param_name for param_def in self.parameter_definitions):
                errors.append(f"Unknown parameter '{param_name}'")
        
        if not errors:
            self.parameters = parameters
            self.updated_at = datetime.datetime.now()
            logger.info(f"Set parameters for experiment '{self.name}'")
        
        return errors
    
    def validate(self) -> List[str]:
        """
        Validate the experiment definition.
        
        Returns:
            List of validation error messages (empty if experiment is valid)
        """
        errors = []
        
        # Validate parameters
        for param_def in self.parameter_definitions:
            if param_def.name in self.parameters:
                is_valid, error = param_def.validate(self.parameters[param_def.name])
                if not is_valid:
                    errors.append(error)
            elif param_def.required:
                errors.append(f"Required parameter '{param_def.name}' is missing")
        
        # Validate steps
        step_names = set()
        for step in self.steps:
            # Check for duplicate step names
            if step.name in step_names:
                errors.append(f"Duplicate step name '{step.name}'")
            step_names.add(step.name)
            
            # Check for missing step functions
            if step.function is None:
                errors.append(f"Step '{step.name}' has no function defined")
            
            # Check for circular dependencies
            visited = set()
            to_visit = [step.name]
            
            while to_visit:
                current = to_visit.pop()
                if current in visited:
                    errors.append(f"Circular dependency detected involving step '{current}'")
                    break
                
                visited.add(current)
                
                # Add dependencies to visit
                for s in self.steps:
                    if s.name == current:
                        for dep in s.depends_on:
                            to_visit.append(dep)
        
        return errors
    
    def plan(self) -> bool:
        """
        Plan the experiment execution.
        
        Returns:
            True if planning was successful, False otherwise
        """
        # Validate experiment
        errors = self.validate()
        if errors:
            logger.error(f"Experiment '{self.name}' validation failed: {errors}")
            return False
        
        # Set status to planned
        self.status = ExperimentStatus.PLANNED
        self.updated_at = datetime.datetime.now()
        
        logger.info(f"Planned experiment '{self.name}'")
        return True
    
    def execute(self, async_mode: bool = False) -> Union[bool, threading.Thread]:
        """
        Execute the experiment.
        
        Args:
            async_mode: If True, execute asynchronously and return a Thread object
            
        Returns:
            True if execution was successful, False if it failed, or Thread object if async_mode is True
        """
        # Plan experiment if not already planned
        if self.status != ExperimentStatus.PLANNED:
            if not self.plan():
                return False
        
        if async_mode:
            # Execute asynchronously
            thread = threading.Thread(target=self._execute_internal)
            thread.daemon = True
            thread.start()
            return thread
        else:
            # Execute synchronously
            return self._execute_internal()
    
    def _execute_internal(self) -> bool:
        """
        Internal method to execute the experiment.
        
        Returns:
            True if execution was successful, False otherwise
        """
        # Set status to running
        self.status = ExperimentStatus.RUNNING
        self.started_at = datetime.datetime.now()
        self.updated_at = datetime.datetime.now()
        
        logger.info(f"Started execution of experiment '{self.name}'")
        self.logs.append(f"Started execution at {self.started_at}")
        
        try:
            # Build execution graph
            execution_graph = self._build_execution_graph()
            
            # Execute steps in topological order
            for step_name in execution_graph:
                step = next(s for s in self.steps if s.name == step_name)
                
                # Check if dependencies are completed
                deps_completed = True
                for dep in step.depends_on:
                    dep_step = next(s for s in self.steps if s.name == dep)
                    if dep_step.status != "completed":
                        deps_completed = False
                        break
                
                if not deps_completed:
                    step.status = "skipped"
                    self.logs.append(f"Skipped step '{step.name}' due to failed dependencies")
                    continue
                
                # Execute step
                self._execute_step(step)
            
            # Check if all steps completed successfully
            all_completed = all(step.status == "completed" for step in self.steps)
            
            if all_completed:
                self.status = ExperimentStatus.COMPLETED
                self.logs.append(f"Experiment completed successfully")
            else:
                self.status = ExperimentStatus.FAILED
                self.logs.append(f"Experiment failed: some steps did not complete successfully")
            
            self.completed_at = datetime.datetime.now()
            self.updated_at = datetime.datetime.now()
            
            logger.info(f"Completed execution of experiment '{self.name}' with status {self.status}")
            
            return all_completed
        
        except Exception as e:
            self.status = ExperimentStatus.FAILED
            self.error = str(e)
            self.completed_at = datetime.datetime.now()
            self.updated_at = datetime.datetime.now()
            
            error_traceback = traceback.format_exc()
            self.logs.append(f"Experiment failed with error: {str(e)}")
            self.logs.append(error_traceback)
            
            logger.error(f"Experiment '{self.name}' failed with error: {str(e)}")
            logger.error(error_traceback)
            
            return False
    
    def _build_execution_graph(self) -> List[str]:
        """
        Build a topological ordering of steps for execution.
        
        Returns:
            List of step names in execution order
        """
        # Build dependency graph
        graph = {step.name: set(step.depends_on) for step in self.steps}
        
        # Perform topological sort
        result = []
        visited = set()
        temp_visited = set()
        
        def visit(node):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected involving step '{node}'")
            
            if node not in visited:
                temp_visited.add(node)
                
                for dep in graph[node]:
                    visit(dep)
                
                temp_visited.remove(node)
                visited.add(node)
                result.append(node)
        
        # Visit all nodes
        for node in graph:
            if node not in visited:
                visit(node)
        
        # Reverse the result to get correct execution order
        return list(reversed(result))
    
    def _execute_step(self, step: ExperimentStep) -> None:
        """
        Execute a single experiment step.
        
        Args:
            step: Experiment step to execute
        """
        logger.info(f"Executing step '{step.name}' of experiment '{self.name}'")
        self.logs.append(f"Executing step '{step.name}'")
        
        step.status = "running"
        step.started_at = datetime.datetime.now()
        
        try:
            # Prepare step parameters
            step_params = copy.deepcopy(step.parameters)
            
            # Add experiment parameters
            for param_name, param_value in self.parameters.items():
                if param_name not in step_params:
                    step_params[param_name] = param_value
            
            # Add experiment context
            context = {
                'experiment_id': self.id,
                'experiment_name': self.name,
                'step_name': step.name,
                'artifacts': {a.name: a for a in self.artifacts},
                'metrics': self.metrics,
                'log': lambda msg: step.logs.append(msg)
            }
            
            # Execute step function
            if step.function is not None:
                result = step.function(step_params, context)
                
                # Process step results
                if isinstance(result, dict):
                    # Extract artifacts
                    if 'artifacts' in result:
                        for artifact in result['artifacts']:
                            if isinstance(artifact, ExperimentArtifact):
                                step.artifacts.append(artifact)
                                self.artifacts.append(artifact)
                            else:
                                logger.warning(f"Invalid artifact returned by step '{step.name}'")
                    
                    # Extract metrics
                    if 'metrics' in result:
                        for metric_name, metric_value in result['metrics'].items():
                            step.metrics[metric_name] = metric_value
                            self.metrics[metric_name] = metric_value
            
            step.status = "completed"
            step.completed_at = datetime.datetime.now()
            
            logger.info(f"Step '{step.name}' completed successfully")
            self.logs.append(f"Step '{step.name}' completed successfully")
        
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.datetime.now()
            
            error_traceback = traceback.format_exc()
            step.logs.append(f"Step failed with error: {str(e)}")
            step.logs.append(error_traceback)
            
            self.logs.append(f"Step '{step.name}' failed with error: {str(e)}")
            
            logger.error(f"Step '{step.name}' of experiment '{self.name}' failed with error: {str(e)}")
            logger.error(error_traceback)
    
    def abort(self) -> None:
        """Abort the experiment execution."""
        if self.status == ExperimentStatus.RUNNING:
            self.status = ExperimentStatus.ABORTED
            self.completed_at = datetime.datetime.now()
            self.updated_at = datetime.datetime.now()
            
            self.logs.append(f"Experiment aborted at {self.completed_at}")
            logger.info(f"Aborted experiment '{self.name}'")
    
    def archive(self) -> None:
        """Archive the experiment."""
        self.status = ExperimentStatus.ARCHIVED
        self.updated_at = datetime.datetime.now()
        
        self.logs.append(f"Experiment archived at {self.updated_at}")
        logger.info(f"Archived experiment '{self.name}'")
    
    def clone(self, new_name: Optional[str] = None) -> 'Experiment':
        """
        Create a clone of the experiment.
        
        Args:
            new_name: Name for the cloned experiment (default: original name + " (clone)")
            
        Returns:
            Cloned experiment
        """
        # Create a deep copy
        clone = copy.deepcopy(self)
        
        # Update metadata
        clone.id = str(uuid.uuid4())
        clone.name = new_name if new_name else f"{self.name} (clone)"
        clone.created_at = datetime.datetime.now()
        clone.updated_at = datetime.datetime.now()
        clone.status = ExperimentStatus.DRAFT
        
        # Reset execution data
        clone.started_at = None
        clone.completed_at = None
        clone.logs = []
        clone.error = None
        
        # Reset step execution data
        for step in clone.steps:
            step.status = "pending"
            step.started_at = None
            step.completed_at = None
            step.logs = []
            step.error = None
        
        logger.info(f"Cloned experiment '{self.name}' to '{clone.name}'")
        return clone
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the experiment to a dictionary.
        
        Returns:
            Dictionary representation of the experiment
        """
        # Convert to dictionary using dataclasses.asdict
        result = asdict(self)
        
        # Convert enums to strings
        result['status'] = self.status.value
        
        # Convert parameter definitions
        result['parameter_definitions'] = [
            {**asdict(param), 'type': param.type.value}
            for param in self.parameter_definitions
        ]
        
        # Convert datetime objects to ISO format strings
        for key in ['created_at', 'updated_at', 'started_at', 'completed_at']:
            if result[key] is not None:
                result[key] = result[key].isoformat()
        
        # Convert step datetime objects
        for step in result['steps']:
            for key in ['started_at', 'completed_at']:
                if step[key] is not None:
                    step[key] = step[key].isoformat()
        
        # Convert artifact datetime objects
        for artifact in result['artifacts']:
            if artifact['created_at'] is not None:
                artifact['created_at'] = artifact['created_at'].isoformat()
        
        # Remove function references
        for step in result['steps']:
            step.pop('function', None)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experiment':
        """
        Create an experiment from a dictionary.
        
        Args:
            data: Dictionary representation of the experiment
            
        Returns:
            Experiment object
        """
        # Create a copy of the data to avoid modifying the original
        data = copy.deepcopy(data)
        
        # Convert status string to enum
        data['status'] = ExperimentStatus(data['status'])
        
        # Convert parameter definition dictionaries to objects
        param_defs = []
        for param_dict in data.get('parameter_definitions', []):
            param_type = param_dict.pop('type')
            param_defs.append(ExperimentParameter(
                type=ParameterType(param_type),
                **param_dict
            ))
        data['parameter_definitions'] = param_defs
        
        # Convert metric definition dictionaries to objects
        metric_defs = []
        for metric_dict in data.get('metric_definitions', []):
            metric_defs.append(ExperimentMetric(**metric_dict))
        data['metric_definitions'] = metric_defs
        
        # Convert step dictionaries to objects
        steps = []
        for step_dict in data.get('steps', []):
            # Convert artifact dictionaries to objects
            artifacts = []
            for artifact_dict in step_dict.get('artifacts', []):
                created_at = artifact_dict.get('created_at')
                if created_at and isinstance(created_at, str):
                    artifact_dict['created_at'] = datetime.datetime.fromisoformat(created_at)
                artifacts.append(ExperimentArtifact(**artifact_dict))
            step_dict['artifacts'] = artifacts
            
            # Convert datetime strings to objects
            for key in ['started_at', 'completed_at']:
                if step_dict.get(key) and isinstance(step_dict[key], str):
                    step_dict[key] = datetime.datetime.fromisoformat(step_dict[key])
            
            steps.append(ExperimentStep(**step_dict))
        data['steps'] = steps
        
        # Convert artifact dictionaries to objects
        artifacts = []
        for artifact_dict in data.get('artifacts', []):
            created_at = artifact_dict.get('created_at')
            if created_at and isinstance(created_at, str):
                artifact_dict['created_at'] = datetime.datetime.fromisoformat(created_at)
            artifacts.append(ExperimentArtifact(**artifact_dict))
        data['artifacts'] = artifacts
        
        # Convert datetime strings to objects
        for key in ['created_at', 'updated_at', 'started_at', 'completed_at']:
            if data.get(key) and isinstance(data[key], str):
                data[key] = datetime.datetime.fromisoformat(data[key])
        
        # Create experiment object
        return cls(**data)
    
    def save(self, directory: str) -> str:
        """
        Save the experiment to a file.
        
        Args:
            directory: Directory to save the experiment
            
        Returns:
            Path to the saved experiment file
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Convert to dictionary
        data = self.to_dict()
        
        # Generate filename
        filename = f"{self.id}.json"
        filepath = os.path.join(directory, filename)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved experiment '{self.name}' to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'Experiment':
        """
        Load an experiment from a file.
        
        Args:
            filepath: Path to the experiment file
            
        Returns:
            Loaded experiment
        """
        # Load from file
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create experiment object
        experiment = cls.from_dict(data)
        
        logger.info(f"Loaded experiment '{experiment.name}' from {filepath}")
        return experiment


class ExperimentWorkflowManager:
    """
    Manager for experiment workflows.
    
    This class provides methods for creating, executing, and managing
    scientific experiments with reproducibility and traceability.
    """
    
    def __init__(self, storage_dir: str = "experiments"):
        """
        Initialize the ExperimentWorkflowManager.
        
        Args:
            storage_dir: Directory for storing experiment data
        """
        self.storage_dir = storage_dir
        self.experiments = {}
        self.running_experiments = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        logger.info(f"Initialized ExperimentWorkflowManager with storage directory: {storage_dir}")
    
    def create_experiment(self, name: str, description: str = "", author: str = "") -> Experiment:
        """
        Create a new experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            author: Experiment author
            
        Returns:
            Created experiment
        """
        experiment = Experiment(
            name=name,
            description=description,
            author=author
        )
        
        self.experiments[experiment.id] = experiment
        
        logger.info(f"Created experiment '{name}' with ID {experiment.id}")
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get an experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment object or None if not found
        """
        return self.experiments.get(experiment_id)
    
    def list_experiments(self, status: Optional[ExperimentStatus] = None, 
                        author: Optional[str] = None,
                        tag: Optional[str] = None) -> List[Experiment]:
        """
        List experiments with optional filtering.
        
        Args:
            status: Filter by experiment status
            author: Filter by experiment author
            tag: Filter by experiment tag
            
        Returns:
            List of matching experiments
        """
        results = list(self.experiments.values())
        
        # Apply filters
        if status is not None:
            results = [exp for exp in results if exp.status == status]
        
        if author is not None:
            results = [exp for exp in results if exp.author == author]
        
        if tag is not None:
            results = [exp for exp in results if tag in exp.tags]
        
        return results
    
    def execute_experiment(self, experiment_id: str, async_mode: bool = True) -> bool:
        """
        Execute an experiment.
        
        Args:
            experiment_id: Experiment ID
            async_mode: If True, execute asynchronously
            
        Returns:
            True if execution started successfully, False otherwise
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            logger.error(f"Experiment with ID {experiment_id} not found")
            return False
        
        result = experiment.execute(async_mode=async_mode)
        
        if async_mode:
            # Store thread reference
            self.running_experiments[experiment_id] = result
            return True
        else:
            return result
    
    def abort_experiment(self, experiment_id: str) -> bool:
        """
        Abort a running experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if abort was successful, False otherwise
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            logger.error(f"Experiment with ID {experiment_id} not found")
            return False
        
        if experiment.status != ExperimentStatus.RUNNING:
            logger.error(f"Experiment '{experiment.name}' is not running")
            return False
        
        experiment.abort()
        
        # Remove from running experiments
        if experiment_id in self.running_experiments:
            del self.running_experiments[experiment_id]
        
        return True
    
    def save_experiment(self, experiment_id: str) -> Optional[str]:
        """
        Save an experiment to disk.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Path to the saved experiment file, or None if experiment not found
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            logger.error(f"Experiment with ID {experiment_id} not found")
            return None
        
        return experiment.save(self.storage_dir)
    
    def load_experiment(self, filepath: str) -> Optional[Experiment]:
        """
        Load an experiment from disk.
        
        Args:
            filepath: Path to the experiment file
            
        Returns:
            Loaded experiment, or None if loading failed
        """
        try:
            experiment = Experiment.load(filepath)
            self.experiments[experiment.id] = experiment
            return experiment
        except Exception as e:
            logger.error(f"Error loading experiment from {filepath}: {str(e)}")
            return None
    
    def load_all_experiments(self) -> int:
        """
        Load all experiments from the storage directory.
        
        Returns:
            Number of experiments loaded
        """
        count = 0
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_dir, filename)
                experiment = self.load_experiment(filepath)
                if experiment is not None:
                    count += 1
        
        logger.info(f"Loaded {count} experiments from {self.storage_dir}")
        return count
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if deletion was successful, False otherwise
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            logger.error(f"Experiment with ID {experiment_id} not found")
            return False
        
        # Check if experiment is running
        if experiment.status == ExperimentStatus.RUNNING:
            logger.error(f"Cannot delete running experiment '{experiment.name}'")
            return False
        
        # Remove from experiments dictionary
        del self.experiments[experiment_id]
        
        # Delete experiment file if it exists
        filepath = os.path.join(self.storage_dir, f"{experiment_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
        
        logger.info(f"Deleted experiment '{experiment.name}' with ID {experiment_id}")
        return True
    
    def clone_experiment(self, experiment_id: str, new_name: Optional[str] = None) -> Optional[Experiment]:
        """
        Clone an experiment.
        
        Args:
            experiment_id: Experiment ID
            new_name: Name for the cloned experiment
            
        Returns:
            Cloned experiment, or None if original experiment not found
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            logger.error(f"Experiment with ID {experiment_id} not found")
            return None
        
        clone = experiment.clone(new_name)
        self.experiments[clone.id] = clone
        
        logger.info(f"Cloned experiment '{experiment.name}' to '{clone.name}' with ID {clone.id}")
        return clone
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            
        Returns:
            Dictionary with comparison results
        """
        experiments = []
        for exp_id in experiment_ids:
            experiment = self.get_experiment(exp_id)
            if experiment is not None:
                experiments.append(experiment)
        
        if not experiments:
            logger.error("No valid experiments to compare")
            return {}
        
        # Collect metrics from all experiments
        metrics = {}
        for experiment in experiments:
            for metric_name, metric_value in experiment.metrics.items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append((experiment.id, experiment.name, metric_value))
        
        # Collect parameters from all experiments
        parameters = {}
        for experiment in experiments:
            for param_name, param_value in experiment.parameters.items():
                if param_name not in parameters:
                    parameters[param_name] = []
                parameters[param_name].append((experiment.id, experiment.name, param_value))
        
        # Prepare comparison result
        result = {
            'experiments': [
                {
                    'id': exp.id,
                    'name': exp.name,
                    'status': exp.status.value,
                    'created_at': exp.created_at.isoformat() if exp.created_at else None,
                    'completed_at': exp.completed_at.isoformat() if exp.completed_at else None
                }
                for exp in experiments
            ],
            'metrics': metrics,
            'parameters': parameters
        }
        
        logger.info(f"Compared {len(experiments)} experiments")
        return result
    
    def export_experiment_results(self, experiment_id: str, format: str = 'json') -> Optional[str]:
        """
        Export experiment results to a file.
        
        Args:
            experiment_id: Experiment ID
            format: Export format ('json', 'csv', 'html')
            
        Returns:
            Path to the exported file, or None if export failed
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            logger.error(f"Experiment with ID {experiment_id} not found")
            return None
        
        # Create export directory if it doesn't exist
        export_dir = os.path.join(self.storage_dir, 'exports')
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate export filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{experiment.name.replace(' ', '_')}_{timestamp}"
        
        if format == 'json':
            # Export as JSON
            filepath = os.path.join(export_dir, f"{filename}.json")
            
            # Prepare export data
            export_data = {
                'experiment': experiment.to_dict(),
                'results': {
                    'metrics': experiment.metrics,
                    'artifacts': [asdict(a) for a in experiment.artifacts],
                    'steps': [
                        {
                            'name': step.name,
                            'status': step.status,
                            'metrics': step.metrics,
                            'started_at': step.started_at.isoformat() if step.started_at else None,
                            'completed_at': step.completed_at.isoformat() if step.completed_at else None,
                            'error': step.error
                        }
                        for step in experiment.steps
                    ]
                }
            }
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        elif format == 'csv':
            # Export as CSV
            filepath = os.path.join(export_dir, f"{filename}.csv")
            
            # Prepare metrics data
            metrics_data = [{'metric': k, 'value': v} for k, v in experiment.metrics.items()]
            metrics_df = pd.DataFrame(metrics_data)
            
            # Prepare steps data
            steps_data = [
                {
                    'step': step.name,
                    'status': step.status,
                    'started_at': step.started_at,
                    'completed_at': step.completed_at,
                    'duration': (step.completed_at - step.started_at).total_seconds() if step.completed_at and step.started_at else None
                }
                for step in experiment.steps
            ]
            steps_df = pd.DataFrame(steps_data)
            
            # Write to file
            with open(filepath, 'w') as f:
                f.write(f"# Experiment: {experiment.name}\n")
                f.write(f"# ID: {experiment.id}\n")
                f.write(f"# Status: {experiment.status.value}\n")
                f.write(f"# Created: {experiment.created_at}\n")
                f.write(f"# Completed: {experiment.completed_at}\n\n")
                
                f.write("# Metrics\n")
                f.write(metrics_df.to_csv(index=False))
                
                f.write("\n# Steps\n")
                f.write(steps_df.to_csv(index=False))
        
        elif format == 'html':
            # Export as HTML
            filepath = os.path.join(export_dir, f"{filename}.html")
            
            # Prepare HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Experiment Results: {experiment.name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .success {{ color: green; }}
                    .failure {{ color: red; }}
                    .pending {{ color: orange; }}
                </style>
            </head>
            <body>
                <h1>Experiment Results: {experiment.name}</h1>
                <p><strong>ID:</strong> {experiment.id}</p>
                <p><strong>Description:</strong> {experiment.description}</p>
                <p><strong>Author:</strong> {experiment.author}</p>
                <p><strong>Status:</strong> {experiment.status.value}</p>
                <p><strong>Created:</strong> {experiment.created_at}</p>
                <p><strong>Started:</strong> {experiment.started_at}</p>
                <p><strong>Completed:</strong> {experiment.completed_at}</p>
                
                <h2>Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
            """
            
            # Add metrics
            for metric_name, metric_value in experiment.metrics.items():
                html_content += f"""
                    <tr>
                        <td>{metric_name}</td>
                        <td>{metric_value}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <h2>Steps</h2>
                <table>
                    <tr>
                        <th>Step</th>
                        <th>Status</th>
                        <th>Started</th>
                        <th>Completed</th>
                        <th>Duration</th>
                    </tr>
            """
            
            # Add steps
            for step in experiment.steps:
                duration = ""
                if step.started_at and step.completed_at:
                    duration = f"{(step.completed_at - step.started_at).total_seconds():.2f} seconds"
                
                status_class = ""
                if step.status == "completed":
                    status_class = "success"
                elif step.status == "failed":
                    status_class = "failure"
                elif step.status == "pending":
                    status_class = "pending"
                
                html_content += f"""
                    <tr>
                        <td>{step.name}</td>
                        <td class="{status_class}">{step.status}</td>
                        <td>{step.started_at}</td>
                        <td>{step.completed_at}</td>
                        <td>{duration}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <h2>Parameters</h2>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
            """
            
            # Add parameters
            for param_name, param_value in experiment.parameters.items():
                html_content += f"""
                    <tr>
                        <td>{param_name}</td>
                        <td>{param_value}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <h2>Logs</h2>
                <pre>
            """
            
            # Add logs
            for log in experiment.logs:
                html_content += f"{log}\n"
            
            html_content += """
                </pre>
            </body>
            </html>
            """
            
            # Write to file
            with open(filepath, 'w') as f:
                f.write(html_content)
        
        else:
            logger.error(f"Unsupported export format: {format}")
            return None
        
        logger.info(f"Exported experiment '{experiment.name}' results to {filepath}")
        return filepath


# Create a singleton instance for easy import
experiment_manager = None

def get_experiment_manager(storage_dir: str = "experiments") -> ExperimentWorkflowManager:
    """
    Get the experiment workflow manager singleton instance.
    
    Args:
        storage_dir: Directory for storing experiment data
        
    Returns:
        ExperimentWorkflowManager instance
    """
    global experiment_manager
    
    if experiment_manager is None:
        experiment_manager = ExperimentWorkflowManager(storage_dir=storage_dir)
    
    return experiment_manager
