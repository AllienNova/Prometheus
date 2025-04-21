"""
Machine learning module for scientific data analysis.

This module provides machine learning capabilities for scientific research,
including supervised and unsupervised learning, model evaluation, and interpretation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, explained_variance_score,
    confusion_matrix, classification_report, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib
import os
import json

# Configure logging
logger = logging.getLogger(__name__)

class MachineLearningModel:
    """
    Class for building, training, evaluating, and interpreting machine learning models
    for scientific data analysis.
    """
    
    def __init__(self, model_type: str = 'classification', random_state: int = 42):
        """
        Initialize the MachineLearningModel.
        
        Args:
            model_type: Type of model ('classification' or 'regression')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.pipeline = None
        self.feature_names = None
        self.target_name = None
        self.classes = None
        self.scaler = None
        self.is_fitted = False
        self.model_info = {
            'model_type': model_type,
            'random_state': random_state,
            'model_name': None,
            'feature_importance': None,
            'performance_metrics': None,
            'hyperparameters': None
        }
        
        logger.info(f"Initialized MachineLearningModel for {model_type}")
    
    def create_model(self, algorithm: str, hyperparameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a machine learning model with the specified algorithm and hyperparameters.
        
        Args:
            algorithm: Name of the algorithm to use
            hyperparameters: Dictionary of hyperparameters for the model
        """
        if hyperparameters is None:
            hyperparameters = {}
        
        # Classification models
        if self.model_type == 'classification':
            if algorithm == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression(random_state=self.random_state, **hyperparameters)
            
            elif algorithm == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(random_state=self.random_state, **hyperparameters)
            
            elif algorithm == 'svm':
                from sklearn.svm import SVC
                self.model = SVC(random_state=self.random_state, probability=True, **hyperparameters)
            
            elif algorithm == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingClassifier
                self.model = GradientBoostingClassifier(random_state=self.random_state, **hyperparameters)
            
            elif algorithm == 'xgboost':
                try:
                    import xgboost as xgb
                    self.model = xgb.XGBClassifier(random_state=self.random_state, **hyperparameters)
                except ImportError:
                    logger.error("XGBoost is not installed. Please install it with 'pip install xgboost'")
                    raise
            
            elif algorithm == 'lightgbm':
                try:
                    import lightgbm as lgb
                    self.model = lgb.LGBMClassifier(random_state=self.random_state, **hyperparameters)
                except ImportError:
                    logger.error("LightGBM is not installed. Please install it with 'pip install lightgbm'")
                    raise
            
            elif algorithm == 'knn':
                from sklearn.neighbors import KNeighborsClassifier
                self.model = KNeighborsClassifier(**hyperparameters)
            
            elif algorithm == 'naive_bayes':
                from sklearn.naive_bayes import GaussianNB
                self.model = GaussianNB(**hyperparameters)
            
            elif algorithm == 'neural_network':
                from sklearn.neural_network import MLPClassifier
                self.model = MLPClassifier(random_state=self.random_state, **hyperparameters)
            
            else:
                raise ValueError(f"Unknown classification algorithm: {algorithm}")
        
        # Regression models
        elif self.model_type == 'regression':
            if algorithm == 'linear_regression':
                from sklearn.linear_model import LinearRegression
                self.model = LinearRegression(**hyperparameters)
            
            elif algorithm == 'ridge':
                from sklearn.linear_model import Ridge
                self.model = Ridge(random_state=self.random_state, **hyperparameters)
            
            elif algorithm == 'lasso':
                from sklearn.linear_model import Lasso
                self.model = Lasso(random_state=self.random_state, **hyperparameters)
            
            elif algorithm == 'elastic_net':
                from sklearn.linear_model import ElasticNet
                self.model = ElasticNet(random_state=self.random_state, **hyperparameters)
            
            elif algorithm == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(random_state=self.random_state, **hyperparameters)
            
            elif algorithm == 'svm':
                from sklearn.svm import SVR
                self.model = SVR(**hyperparameters)
            
            elif algorithm == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingRegressor
                self.model = GradientBoostingRegressor(random_state=self.random_state, **hyperparameters)
            
            elif algorithm == 'xgboost':
                try:
                    import xgboost as xgb
                    self.model = xgb.XGBRegressor(random_state=self.random_state, **hyperparameters)
                except ImportError:
                    logger.error("XGBoost is not installed. Please install it with 'pip install xgboost'")
                    raise
            
            elif algorithm == 'lightgbm':
                try:
                    import lightgbm as lgb
                    self.model = lgb.LGBMRegressor(random_state=self.random_state, **hyperparameters)
                except ImportError:
                    logger.error("LightGBM is not installed. Please install it with 'pip install lightgbm'")
                    raise
            
            elif algorithm == 'knn':
                from sklearn.neighbors import KNeighborsRegressor
                self.model = KNeighborsRegressor(**hyperparameters)
            
            elif algorithm == 'neural_network':
                from sklearn.neural_network import MLPRegressor
                self.model = MLPRegressor(random_state=self.random_state, **hyperparameters)
            
            else:
                raise ValueError(f"Unknown regression algorithm: {algorithm}")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Use 'classification' or 'regression'.")
        
        # Create a pipeline with scaling
        self.scaler = StandardScaler()
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('model', self.model)
        ])
        
        # Update model info
        self.model_info['model_name'] = algorithm
        self.model_info['hyperparameters'] = hyperparameters
        
        logger.info(f"Created {algorithm} model with hyperparameters: {hyperparameters}")
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
             y: Union[pd.Series, np.ndarray],
             test_size: float = 0.2,
             feature_names: Optional[List[str]] = None,
             target_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data to use for testing
            feature_names: Names of the features (if X is not a DataFrame)
            target_name: Name of the target variable (if y is not a Series)
            
        Returns:
            Dictionary with training results
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Store feature and target names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        elif feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            self.target_name = y.name
        elif target_name is not None:
            self.target_name = target_name
        else:
            self.target_name = "target"
        
        # Convert to numpy arrays if pandas objects
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array, test_size=test_size, random_state=self.random_state
        )
        
        # Store classes for classification problems
        if self.model_type == 'classification':
            self.classes = np.unique(y_array)
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        self.model_info['performance_metrics'] = metrics
        
        # Extract feature importance if available
        self._extract_feature_importance()
        
        logger.info(f"Trained {self.model_info['model_name']} model with {X_train.shape[0]} samples")
        logger.info(f"Test set performance: {metrics}")
        
        return {
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'metrics': metrics,
            'feature_importance': self.model_info['feature_importance']
        }
    
    def cross_validate(self, X: Union[pd.DataFrame, np.ndarray], 
                      y: Union[pd.Series, np.ndarray],
                      cv: int = 5,
                      scoring: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            scoring: Scoring metric(s) to use
            
        Returns:
            Dictionary with cross-validation results
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Convert to numpy arrays if pandas objects
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Set default scoring based on model type
        if scoring is None:
            if self.model_type == 'classification':
                scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'
        
        # Perform cross-validation
        if isinstance(scoring, list):
            cv_results = {}
            for score in scoring:
                cv_scores = cross_val_score(self.pipeline, X_array, y_array, cv=cv, scoring=score)
                cv_results[score] = {
                    'mean': float(cv_scores.mean()),
                    'std': float(cv_scores.std()),
                    'scores': cv_scores.tolist()
                }
        else:
            cv_scores = cross_val_score(self.pipeline, X_array, y_array, cv=cv, scoring=scoring)
            cv_results = {
                scoring: {
                    'mean': float(cv_scores.mean()),
                    'std': float(cv_scores.std()),
                    'scores': cv_scores.tolist()
                }
            }
        
        logger.info(f"Performed {cv}-fold cross-validation with scoring: {scoring}")
        
        return {
            'cv_folds': cv,
            'scoring': scoring,
            'results': cv_results
        }
    
    def hyperparameter_tuning(self, X: Union[pd.DataFrame, np.ndarray], 
                             y: Union[pd.Series, np.ndarray],
                             param_grid: Dict[str, List[Any]],
                             cv: int = 5,
                             scoring: Optional[str] = None,
                             n_jobs: int = -1) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X: Feature matrix
            y: Target vector
            param_grid: Dictionary with hyperparameter grid
            cv: Number of cross-validation folds
            scoring: Scoring metric to use
            n_jobs: Number of parallel jobs (-1 for all processors)
            
        Returns:
            Dictionary with tuning results
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Convert to numpy arrays if pandas objects
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Set default scoring based on model type
        if scoring is None:
            if self.model_type == 'classification':
                scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'
        
        # Prepare parameter grid for pipeline
        pipeline_param_grid = {}
        for param, values in param_grid.items():
            pipeline_param_grid[f'model__{param}'] = values
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.pipeline, pipeline_param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs
        )
        grid_search.fit(X_array, y_array)
        
        # Update model with best parameters
        self.pipeline = grid_search.best_estimator_
        self.model = self.pipeline.named_steps['model']
        self.is_fitted = True
        
        # Update model info
        best_params = {param.replace('model__', ''): value 
                      for param, value in grid_search.best_params_.items()}
        self.model_info['hyperparameters'] = best_params
        
        # Extract feature importance if available
        self._extract_feature_importance()
        
        logger.info(f"Performed hyperparameter tuning with {cv}-fold cross-validation")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': float(grid_search.best_score_),
            'cv_results': grid_search.cv_results_,
            'feature_importance': self.model_info['feature_importance']
        }
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert to numpy array if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Make predictions
        predictions = self.pipeline.predict(X_array)
        
        logger.info(f"Made predictions for {X_array.shape[0]} samples")
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for classification models.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.model_type != 'classification':
            raise ValueError("predict_proba() is only available for classification models")
        
        # Check if the model has predict_proba method
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"The {self.model_info['model_name']} model does not support probability predictions")
        
        # Convert to numpy array if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Predict probabilities
        probabilities = self.pipeline.predict_proba(X_array)
        
        logger.info(f"Predicted probabilities for {X_array.shape[0]} samples")
        
        return probabilities
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], 
                y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate the model on the provided data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert to numpy arrays if pandas objects
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Make predictions
        y_pred = self.pipeline.predict(X_array)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(y_array, y_pred)
        
        logger.info(f"Evaluated model on {X_array.shape[0]} samples")
        logger.info(f"Performance metrics: {metrics}")
        
        return metrics
    
    def feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores if available.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.model_info['feature_importance'] is None:
            raise ValueError(f"Feature importance not available for {self.model_info['model_name']} model")
        
        return self.model_info['feature_importance']
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the pipeline
        joblib.dump(self.pipeline, filepath)
        
        # Save model info as JSON
        info_filepath = f"{os.path.splitext(filepath)[0]}_info.json"
        with open(info_filepath, 'w') as f:
            json.dump(self.model_info, f, indent=2)
        
        logger.info(f"Saved model to {filepath} and model info to {info_filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        # Load the pipeline
        self.pipeline = joblib.load(filepath)
        self.model = self.pipeline.named_steps['model']
        self.scaler = self.pipeline.named_steps['scaler']
        self.is_fitted = True
        
        # Load model info
        info_filepath = f"{os.path.splitext(filepath)[0]}_info.json"
        if os.path.exists(info_filepath):
            with open(info_filepath, 'r') as f:
                self.model_info = json.load(f)
            
            # Update model type and other attributes
            self.model_type = self.model_info['model_type']
            self.random_state = self.model_info['random_state']
            self.feature_names = self.model_info.get('feature_names')
            self.target_name = self.model_info.get('target_name')
        
        logger.info(f"Loaded model from {filepath}")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics based on model type.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        if self.model_type == 'classification':
            # Classification metrics
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            
            # For binary classification
            if len(np.unique(y_true)) == 2:
                metrics['precision'] = float(precision_score(y_true, y_pred, average='binary'))
                metrics['recall'] = float(recall_score(y_true, y_pred, average='binary'))
                metrics['f1'] = float(f1_score(y_true, y_pred, average='binary'))
            else:
                # For multiclass classification
                metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro'))
                metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro'))
                metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro'))
                metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted'))
                metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted'))
                metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted'))
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
        else:
            # Regression metrics
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['r2'] = float(r2_score(y_true, y_pred))
            metrics['explained_variance'] = float(explained_variance_score(y_true, y_pred))
        
        return metrics
    
    def _extract_feature_importance(self) -> None:
        """
        Extract feature importance from the model if available.
        """
        # Check if the model has feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            self.model_info['feature_importance'] = dict(zip(self.feature_names, importances.tolist()))
        
        elif hasattr(self.model, 'coef_'):
            if self.model_type == 'classification' and len(self.classes) > 2:
                # For multiclass, take the mean absolute coefficient across all classes
                importances = np.mean(np.abs(self.model.coef_), axis=0)
            else:
                # For binary classification or regression
                importances = np.abs(self.model.coef_)
            
            self.model_info['feature_importance'] = dict(zip(self.feature_names, importances.tolist()))
        
        else:
            self.model_info['feature_importance'] = None


class UnsupervisedLearning:
    """
    Class for unsupervised learning methods including clustering, dimensionality
    reduction, and anomaly detection for scientific data analysis.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the UnsupervisedLearning class.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.model_type = None
        self.model_name = None
        self.is_fitted = False
        self.feature_names = None
        self.scaler = None
        
        logger.info("Initialized UnsupervisedLearning")
    
    def create_clustering(self, algorithm: str, 
                         hyperparameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a clustering model with the specified algorithm and hyperparameters.
        
        Args:
            algorithm: Name of the clustering algorithm
            hyperparameters: Dictionary of hyperparameters for the model
        """
        if hyperparameters is None:
            hyperparameters = {}
        
        self.model_type = 'clustering'
        self.model_name = algorithm
        
        if algorithm == 'kmeans':
            from sklearn.cluster import KMeans
            self.model = KMeans(random_state=self.random_state, **hyperparameters)
        
        elif algorithm == 'dbscan':
            from sklearn.cluster import DBSCAN
            self.model = DBSCAN(**hyperparameters)
        
        elif algorithm == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            self.model = AgglomerativeClustering(**hyperparameters)
        
        elif algorithm == 'gaussian_mixture':
            from sklearn.mixture import GaussianMixture
            self.model = GaussianMixture(random_state=self.random_state, **hyperparameters)
        
        elif algorithm == 'spectral':
            from sklearn.cluster import SpectralClustering
            self.model = SpectralClustering(random_state=self.random_state, **hyperparameters)
        
        elif algorithm == 'mean_shift':
            from sklearn.cluster import MeanShift
            self.model = MeanShift(**hyperparameters)
        
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        
        # Create scaler
        self.scaler = StandardScaler()
        
        logger.info(f"Created {algorithm} clustering model with hyperparameters: {hyperparameters}")
    
    def create_dimensionality_reduction(self, algorithm: str, 
                                      hyperparameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a dimensionality reduction model with the specified algorithm and hyperparameters.
        
        Args:
            algorithm: Name of the dimensionality reduction algorithm
            hyperparameters: Dictionary of hyperparameters for the model
        """
        if hyperparameters is None:
            hyperparameters = {}
        
        self.model_type = 'dimensionality_reduction'
        self.model_name = algorithm
        
        if algorithm == 'pca':
            from sklearn.decomposition import PCA
            self.model = PCA(random_state=self.random_state, **hyperparameters)
        
        elif algorithm == 'tsne':
            from sklearn.manifold import TSNE
            self.model = TSNE(random_state=self.random_state, **hyperparameters)
        
        elif algorithm == 'umap':
            try:
                import umap
                self.model = umap.UMAP(random_state=self.random_state, **hyperparameters)
            except ImportError:
                logger.error("UMAP is not installed. Please install it with 'pip install umap-learn'")
                raise
        
        elif algorithm == 'lda':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            self.model = LinearDiscriminantAnalysis(**hyperparameters)
        
        elif algorithm == 'isomap':
            from sklearn.manifold import Isomap
            self.model = Isomap(**hyperparameters)
        
        elif algorithm == 'nmf':
            from sklearn.decomposition import NMF
            self.model = NMF(random_state=self.random_state, **hyperparameters)
        
        else:
            raise ValueError(f"Unknown dimensionality reduction algorithm: {algorithm}")
        
        # Create scaler
        self.scaler = StandardScaler()
        
        logger.info(f"Created {algorithm} dimensionality reduction model with hyperparameters: {hyperparameters}")
    
    def create_anomaly_detection(self, algorithm: str, 
                               hyperparameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Create an anomaly detection model with the specified algorithm and hyperparameters.
        
        Args:
            algorithm: Name of the anomaly detection algorithm
            hyperparameters: Dictionary of hyperparameters for the model
        """
        if hyperparameters is None:
            hyperparameters = {}
        
        self.model_type = 'anomaly_detection'
        self.model_name = algorithm
        
        if algorithm == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            self.model = IsolationForest(random_state=self.random_state, **hyperparameters)
        
        elif algorithm == 'one_class_svm':
            from sklearn.svm import OneClassSVM
            self.model = OneClassSVM(**hyperparameters)
        
        elif algorithm == 'local_outlier_factor':
            from sklearn.neighbors import LocalOutlierFactor
            self.model = LocalOutlierFactor(**hyperparameters)
        
        elif algorithm == 'elliptic_envelope':
            from sklearn.covariance import EllipticEnvelope
            self.model = EllipticEnvelope(random_state=self.random_state, **hyperparameters)
        
        else:
            raise ValueError(f"Unknown anomaly detection algorithm: {algorithm}")
        
        # Create scaler
        self.scaler = StandardScaler()
        
        logger.info(f"Created {algorithm} anomaly detection model with hyperparameters: {hyperparameters}")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
           y: Optional[Union[pd.Series, np.ndarray]] = None,
           feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fit the model to the data.
        
        Args:
            X: Feature matrix
            y: Target vector (for supervised dimensionality reduction like LDA)
            feature_names: Names of the features (if X is not a DataFrame)
            
        Returns:
            Dictionary with fitting results
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_*() first.")
        
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        elif feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Convert to numpy arrays if pandas objects
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if y is not None and isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Fit the model
        if self.model_type == 'dimensionality_reduction' and self.model_name == 'lda':
            # LDA requires target values
            if y_array is None:
                raise ValueError("LDA requires target values (y)")
            self.model.fit(X_scaled, y_array)
        else:
            # Other models don't need target values
            self.model.fit(X_scaled)
        
        self.is_fitted = True
        
        # Prepare results based on model type
        results = {'model_type': self.model_type, 'model_name': self.model_name}
        
        if self.model_type == 'clustering':
            # Get cluster labels if available
            if hasattr(self.model, 'labels_'):
                labels = self.model.labels_
                results['labels'] = labels.tolist()
                results['n_clusters'] = len(np.unique(labels[labels != -1]))  # Exclude noise points (-1)
                
                # Calculate silhouette score if more than one cluster
                if results['n_clusters'] > 1:
                    from sklearn.metrics import silhouette_score
                    try:
                        silhouette = silhouette_score(X_scaled, labels)
                        results['silhouette_score'] = float(silhouette)
                    except:
                        # Some clustering algorithms may have noise points
                        pass
            
            # Get cluster centers if available
            if hasattr(self.model, 'cluster_centers_'):
                centers = self.model.cluster_centers_
                results['cluster_centers'] = centers.tolist()
            
            # Get inertia if available (K-means)
            if hasattr(self.model, 'inertia_'):
                results['inertia'] = float(self.model.inertia_)
        
        elif self.model_type == 'dimensionality_reduction':
            # Get explained variance if available (PCA)
            if hasattr(self.model, 'explained_variance_ratio_'):
                explained_variance = self.model.explained_variance_ratio_
                results['explained_variance_ratio'] = explained_variance.tolist()
                results['cumulative_explained_variance'] = np.cumsum(explained_variance).tolist()
            
            # Get components if available
            if hasattr(self.model, 'components_'):
                components = self.model.components_
                results['components'] = components.tolist()
                
                # Create feature importance for PCA
                if self.model_name == 'pca':
                    feature_importance = np.abs(components).sum(axis=0)
                    feature_importance = feature_importance / feature_importance.sum()
                    results['feature_importance'] = dict(zip(self.feature_names, feature_importance.tolist()))
        
        elif self.model_type == 'anomaly_detection':
            # Some models predict during fit (LocalOutlierFactor)
            if hasattr(self.model, 'fit_predict'):
                # Convert to binary labels (1: normal, -1: anomaly)
                labels = self.model.fit_predict(X_scaled)
                anomaly_mask = labels == -1
                results['anomaly_indices'] = np.where(anomaly_mask)[0].tolist()
                results['n_anomalies'] = int(anomaly_mask.sum())
                results['anomaly_ratio'] = float(anomaly_mask.sum() / len(labels))
        
        logger.info(f"Fitted {self.model_name} model to {X_array.shape[0]} samples")
        
        return results
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform data using the fitted model (for dimensionality reduction).
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.model_type != 'dimensionality_reduction':
            raise ValueError("transform() is only available for dimensionality reduction models")
        
        # Convert to numpy array if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Scale the data
        X_scaled = self.scaler.transform(X_array)
        
        # Transform the data
        X_transformed = self.model.transform(X_scaled)
        
        logger.info(f"Transformed {X_array.shape[0]} samples to {X_transformed.shape[1]} dimensions")
        
        return X_transformed
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions with the fitted model (for clustering or anomaly detection).
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels or anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.model_type not in ['clustering', 'anomaly_detection']:
            raise ValueError("predict() is only available for clustering and anomaly detection models")
        
        # Convert to numpy array if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Scale the data
        X_scaled = self.scaler.transform(X_array)
        
        # Make predictions
        if self.model_type == 'clustering':
            if hasattr(self.model, 'predict'):
                labels = self.model.predict(X_scaled)
                logger.info(f"Predicted cluster labels for {X_array.shape[0]} samples")
                return labels
            else:
                raise ValueError(f"The {self.model_name} model does not support prediction")
        
        elif self.model_type == 'anomaly_detection':
            if hasattr(self.model, 'predict'):
                # Convert to binary labels (1: normal, -1: anomaly)
                labels = self.model.predict(X_scaled)
                logger.info(f"Predicted anomaly labels for {X_array.shape[0]} samples")
                return labels
            else:
                raise ValueError(f"The {self.model_name} model does not support prediction")
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], 
                     y: Optional[Union[pd.Series, np.ndarray]] = None,
                     feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit the model to the data and transform it (for dimensionality reduction).
        
        Args:
            X: Feature matrix
            y: Target vector (for supervised dimensionality reduction like LDA)
            feature_names: Names of the features (if X is not a DataFrame)
            
        Returns:
            Transformed data
        """
        self.fit(X, y, feature_names)
        
        if self.model_type != 'dimensionality_reduction':
            raise ValueError("fit_transform() with return value is only available for dimensionality reduction models")
        
        # Convert to numpy array if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Scale the data (already fitted in self.fit())
        X_scaled = self.scaler.transform(X_array)
        
        # Transform the data
        X_transformed = self.model.transform(X_scaled)
        
        logger.info(f"Fit and transformed {X_array.shape[0]} samples to {X_transformed.shape[1]} dimensions")
        
        return X_transformed
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create a dictionary with all necessary objects
        model_dict = {
            'model': self.model,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        
        # Save the dictionary
        joblib.dump(model_dict, filepath)
        
        logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a fitted model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        # Load the dictionary
        model_dict = joblib.load(filepath)
        
        # Restore all attributes
        self.model = model_dict['model']
        self.model_type = model_dict['model_type']
        self.model_name = model_dict['model_name']
        self.scaler = model_dict['scaler']
        self.feature_names = model_dict['feature_names']
        self.random_state = model_dict['random_state']
        self.is_fitted = True
        
        logger.info(f"Loaded model from {filepath}")
