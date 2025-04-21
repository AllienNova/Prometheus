"""
Core data processing module for scientific analysis.

This module provides the foundation for scientific data processing,
including data loading, cleaning, transformation, and basic analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Core class for scientific data processing operations.
    
    This class provides methods for loading, cleaning, transforming,
    and performing basic analysis on scientific datasets.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DataProcessor with optional configuration.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or {}
        logger.info("Initialized DataProcessor with config: %s", self.config)
    
    def load_data(self, source: Union[str, pd.DataFrame], 
                  file_format: Optional[str] = None, 
                  **kwargs) -> pd.DataFrame:
        """
        Load data from various sources into a pandas DataFrame.
        
        Args:
            source: Data source (file path, URL, or DataFrame)
            file_format: Format of the data file (csv, excel, json, etc.)
            **kwargs: Additional arguments for the pandas read functions
            
        Returns:
            Loaded data as a pandas DataFrame
        
        Raises:
            ValueError: If the source or format is invalid
        """
        if isinstance(source, pd.DataFrame):
            logger.info("Using provided DataFrame as data source")
            return source
        
        if not isinstance(source, str):
            raise ValueError("Source must be a string path/URL or a DataFrame")
        
        # Determine file format if not provided
        if file_format is None:
            if source.endswith('.csv'):
                file_format = 'csv'
            elif source.endswith(('.xls', '.xlsx')):
                file_format = 'excel'
            elif source.endswith('.json'):
                file_format = 'json'
            elif source.endswith(('.h5', '.hdf5')):
                file_format = 'hdf'
            elif source.endswith('.pkl'):
                file_format = 'pickle'
            else:
                raise ValueError(f"Could not determine file format for {source}")
        
        # Load data based on format
        try:
            if file_format == 'csv':
                data = pd.read_csv(source, **kwargs)
            elif file_format == 'excel':
                data = pd.read_excel(source, **kwargs)
            elif file_format == 'json':
                data = pd.read_json(source, **kwargs)
            elif file_format == 'hdf':
                data = pd.read_hdf(source, **kwargs)
            elif file_format == 'pickle':
                data = pd.read_pickle(source, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"Successfully loaded data from {source} with shape {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {source}: {str(e)}")
            raise
    
    def clean_data(self, data: pd.DataFrame, 
                   handle_missing: bool = True,
                   handle_outliers: bool = False,
                   handle_duplicates: bool = True) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values, outliers, and duplicates.
        
        Args:
            data: Input DataFrame to clean
            handle_missing: Whether to handle missing values
            handle_outliers: Whether to detect and handle outliers
            handle_duplicates: Whether to remove duplicate rows
            
        Returns:
            Cleaned DataFrame
        """
        result = data.copy()
        original_shape = result.shape
        
        # Handle duplicates
        if handle_duplicates:
            original_rows = len(result)
            result = result.drop_duplicates()
            removed_rows = original_rows - len(result)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} duplicate rows")
        
        # Handle missing values
        if handle_missing:
            missing_count = result.isna().sum().sum()
            if missing_count > 0:
                # Get missing value strategy from config or use defaults
                strategy = self.config.get('missing_value_strategy', 'drop')
                
                if strategy == 'drop':
                    # Drop rows with any missing values
                    result = result.dropna()
                    logger.info(f"Dropped rows with missing values, {missing_count} values affected")
                    
                elif strategy == 'fill_mean':
                    # Fill missing values with column means (numeric only)
                    numeric_cols = result.select_dtypes(include=[np.number]).columns
                    result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())
                    logger.info(f"Filled missing numeric values with column means")
                    
                elif strategy == 'fill_median':
                    # Fill missing values with column medians (numeric only)
                    numeric_cols = result.select_dtypes(include=[np.number]).columns
                    result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].median())
                    logger.info(f"Filled missing numeric values with column medians")
                    
                elif strategy == 'fill_mode':
                    # Fill missing values with column modes
                    for col in result.columns:
                        result[col] = result[col].fillna(result[col].mode()[0] if not result[col].mode().empty else None)
                    logger.info(f"Filled missing values with column modes")
                    
                elif strategy == 'fill_constant':
                    # Fill with constant values specified in config
                    fill_values = self.config.get('fill_values', {})
                    for col in result.columns:
                        if col in fill_values:
                            result[col] = result[col].fillna(fill_values[col])
                    logger.info(f"Filled missing values with specified constants")
        
        # Handle outliers
        if handle_outliers:
            # Get outlier strategy from config or use defaults
            strategy = self.config.get('outlier_strategy', 'iqr')
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            
            if strategy == 'iqr':
                # Use IQR method to detect and handle outliers
                for col in numeric_cols:
                    Q1 = result[col].quantile(0.25)
                    Q3 = result[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Replace outliers with bounds or remove them based on config
                    outlier_handling = self.config.get('outlier_handling', 'clip')
                    if outlier_handling == 'clip':
                        result[col] = result[col].clip(lower_bound, upper_bound)
                        logger.info(f"Clipped outliers in column {col}")
                    elif outlier_handling == 'remove':
                        mask = (result[col] >= lower_bound) & (result[col] <= upper_bound)
                        result = result[mask]
                        logger.info(f"Removed outliers in column {col}")
            
            elif strategy == 'zscore':
                # Use Z-score method to detect and handle outliers
                z_threshold = self.config.get('z_threshold', 3.0)
                for col in numeric_cols:
                    z_scores = np.abs((result[col] - result[col].mean()) / result[col].std())
                    
                    # Replace outliers with bounds or remove them based on config
                    outlier_handling = self.config.get('outlier_handling', 'clip')
                    if outlier_handling == 'clip':
                        result.loc[z_scores > z_threshold, col] = None
                        result[col] = result[col].fillna(result[col].mean())
                        logger.info(f"Replaced outliers with mean in column {col}")
                    elif outlier_handling == 'remove':
                        result = result[z_scores <= z_threshold]
                        logger.info(f"Removed outliers in column {col}")
        
        new_shape = result.shape
        logger.info(f"Data cleaning complete. Original shape: {original_shape}, New shape: {new_shape}")
        return result
    
    def transform_data(self, data: pd.DataFrame, 
                       normalize: bool = False,
                       standardize: bool = False,
                       encode_categorical: bool = False) -> pd.DataFrame:
        """
        Transform the dataset with normalization, standardization, and encoding.
        
        Args:
            data: Input DataFrame to transform
            normalize: Whether to normalize numeric features to [0,1]
            standardize: Whether to standardize numeric features (mean=0, std=1)
            encode_categorical: Whether to one-hot encode categorical features
            
        Returns:
            Transformed DataFrame
        
        Note:
            If both normalize and standardize are True, standardization takes precedence.
        """
        result = data.copy()
        
        # Handle numeric features
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        if standardize and len(numeric_cols) > 0:
            # Standardize numeric columns (z-score normalization)
            for col in numeric_cols:
                mean = result[col].mean()
                std = result[col].std()
                if std > 0:  # Avoid division by zero
                    result[col] = (result[col] - mean) / std
                    logger.info(f"Standardized column {col}")
                else:
                    logger.warning(f"Column {col} has zero standard deviation, skipping standardization")
        
        elif normalize and len(numeric_cols) > 0:
            # Min-max normalization to [0,1]
            for col in numeric_cols:
                min_val = result[col].min()
                max_val = result[col].max()
                if max_val > min_val:  # Avoid division by zero
                    result[col] = (result[col] - min_val) / (max_val - min_val)
                    logger.info(f"Normalized column {col} to [0,1] range")
                else:
                    logger.warning(f"Column {col} has constant values, skipping normalization")
        
        # Handle categorical features
        if encode_categorical:
            categorical_cols = result.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) > 0:
                # One-hot encode categorical columns
                for col in categorical_cols:
                    # Get dummies and add prefix to avoid column name conflicts
                    dummies = pd.get_dummies(result[col], prefix=col, drop_first=False)
                    
                    # Drop the original column and join the dummies
                    result = result.drop(col, axis=1)
                    result = pd.concat([result, dummies], axis=1)
                    
                    logger.info(f"One-hot encoded column {col} into {dummies.shape[1]} features")
        
        logger.info(f"Data transformation complete. New shape: {result.shape}")
        return result
    
    def analyze_basic_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform basic statistical analysis on the dataset.
        
        Args:
            data: Input DataFrame to analyze
            
        Returns:
            Dictionary containing basic statistical measures
        """
        stats = {}
        
        # Basic dataset information
        stats['shape'] = data.shape
        stats['columns'] = list(data.columns)
        stats['dtypes'] = {col: str(dtype) for col, dtype in data.dtypes.items()}
        
        # Missing values information
        missing = data.isna().sum()
        stats['missing_values'] = {col: int(count) for col, count in missing.items() if count > 0}
        stats['missing_percentage'] = {col: float(count/len(data)*100) for col, count in missing.items() if count > 0}
        
        # Descriptive statistics for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_stats'] = data[numeric_cols].describe().to_dict()
            
            # Calculate additional statistics
            stats['skewness'] = {col: float(data[col].skew()) for col in numeric_cols}
            stats['kurtosis'] = {col: float(data[col].kurtosis()) for col in numeric_cols}
            
            # Calculate correlation matrix
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr().to_dict()
                stats['correlation_matrix'] = corr_matrix
        
        # Categorical column statistics
        cat_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            stats['categorical_stats'] = {}
            for col in cat_cols:
                value_counts = data[col].value_counts()
                unique_count = len(value_counts)
                stats['categorical_stats'][col] = {
                    'unique_values': unique_count,
                    'top_values': {str(val): int(count) for val, count in value_counts.head(10).items()},
                    'top_percentages': {str(val): float(count/len(data)*100) for val, count in value_counts.head(10).items()}
                }
        
        logger.info(f"Completed basic statistical analysis")
        return stats
    
    def detect_data_issues(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect potential issues in the dataset that might affect analysis.
        
        Args:
            data: Input DataFrame to analyze
            
        Returns:
            Dictionary containing detected issues
        """
        issues = {}
        
        # Check for missing values
        missing = data.isna().sum()
        missing_cols = {col: int(count) for col, count in missing.items() if count > 0}
        if missing_cols:
            issues['missing_values'] = missing_cols
        
        # Check for constant columns
        constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
        if constant_cols:
            issues['constant_columns'] = constant_cols
        
        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            issues['duplicate_rows'] = int(duplicate_count)
        
        # Check for high cardinality categorical columns
        cat_cols = data.select_dtypes(include=['object', 'category']).columns
        high_cardinality_threshold = self.config.get('high_cardinality_threshold', 0.5)
        high_cardinality_cols = []
        
        for col in cat_cols:
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio > high_cardinality_threshold:
                high_cardinality_cols.append({
                    'column': col,
                    'unique_ratio': float(unique_ratio),
                    'unique_count': int(data[col].nunique())
                })
        
        if high_cardinality_cols:
            issues['high_cardinality_columns'] = high_cardinality_cols
        
        # Check for highly correlated features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_threshold = self.config.get('high_correlation_threshold', 0.9)
            high_corr_pairs = []
            
            for col in upper_tri.columns:
                for idx, val in upper_tri[col].items():
                    if val > high_corr_threshold:
                        high_corr_pairs.append({
                            'column1': idx,
                            'column2': col,
                            'correlation': float(val)
                        })
            
            if high_corr_pairs:
                issues['highly_correlated_features'] = high_corr_pairs
        
        # Check for potential outliers in numeric columns
        outlier_issues = {}
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_percentage = float(outliers / len(data) * 100)
                outlier_issues[col] = {
                    'count': int(outliers),
                    'percentage': outlier_percentage,
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
        
        if outlier_issues:
            issues['potential_outliers'] = outlier_issues
        
        logger.info(f"Detected {len(issues)} types of data issues")
        return issues
    
    def feature_engineering(self, data: pd.DataFrame, 
                           operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Perform feature engineering operations on the dataset.
        
        Args:
            data: Input DataFrame
            operations: List of feature engineering operations to perform
            
        Returns:
            DataFrame with engineered features
            
        Example operations:
            [
                {'type': 'polynomial', 'columns': ['x1', 'x2'], 'degree': 2},
                {'type': 'interaction', 'columns': ['x1', 'x2']},
                {'type': 'binning', 'column': 'age', 'bins': 5, 'labels': ['very_young', 'young', 'middle', 'senior', 'elderly']}
            ]
        """
        result = data.copy()
        
        for operation in operations:
            op_type = operation.get('type')
            
            if op_type == 'polynomial':
                columns = operation.get('columns', [])
                degree = operation.get('degree', 2)
                
                if not columns:
                    logger.warning("No columns specified for polynomial features")
                    continue
                
                # Ensure all specified columns exist and are numeric
                valid_columns = [col for col in columns if col in result.columns and 
                                np.issubdtype(result[col].dtype, np.number)]
                
                if not valid_columns:
                    logger.warning("No valid numeric columns found for polynomial features")
                    continue
                
                # Generate polynomial features
                for col in valid_columns:
                    for d in range(2, degree + 1):
                        new_col = f"{col}^{d}"
                        result[new_col] = result[col] ** d
                        logger.info(f"Created polynomial feature {new_col}")
            
            elif op_type == 'interaction':
                columns = operation.get('columns', [])
                
                if len(columns) < 2:
                    logger.warning("At least two columns needed for interaction features")
                    continue
                
                # Ensure all specified columns exist and are numeric
                valid_columns = [col for col in columns if col in result.columns and 
                                np.issubdtype(result[col].dtype, np.number)]
                
                if len(valid_columns) < 2:
                    logger.warning("Not enough valid numeric columns for interaction features")
                    continue
                
                # Generate interaction features
                for i in range(len(valid_columns)):
                    for j in range(i+1, len(valid_columns)):
                        col1 = valid_columns[i]
                        col2 = valid_columns[j]
                        new_col = f"{col1}*{col2}"
                        result[new_col] = result[col1] * result[col2]
                        logger.info(f"Created interaction feature {new_col}")
            
            elif op_type == 'binning':
                column = operation.get('column')
                bins = operation.get('bins', 5)
                labels = operation.get('labels')
                
                if column not in result.columns:
                    logger.warning(f"Column {column} not found for binning")
                    continue
                
                if not np.issubdtype(result[column].dtype, np.number):
                    logger.warning(f"Column {column} is not numeric, cannot perform binning")
                    continue
                
                # Generate bin edges if not provided
                if isinstance(bins, int):
                    bins = np.linspace(result[column].min(), result[column].max(), bins + 1)
                
                # Create new binned column
                new_col = f"{column}_binned"
                result[new_col] = pd.cut(result[column], bins=bins, labels=labels, include_lowest=True)
                logger.info(f"Created binned feature {new_col} from {column}")
            
            elif op_type == 'log_transform':
                columns = operation.get('columns', [])
                base = operation.get('base', 'e')  # 'e' for natural log, 10 for log10
                
                if not columns:
                    logger.warning("No columns specified for log transform")
                    continue
                
                # Ensure all specified columns exist and are numeric
                valid_columns = [col for col in columns if col in result.columns and 
                                np.issubdtype(result[col].dtype, np.number)]
                
                if not valid_columns:
                    logger.warning("No valid numeric columns found for log transform")
                    continue
                
                # Apply log transform
                for col in valid_columns:
                    # Ensure all values are positive
                    min_val = result[col].min()
                    if min_val <= 0:
                        shift = abs(min_val) + 1  # Add 1 to ensure all values are positive
                        result[col] = result[col] + shift
                        logger.info(f"Shifted column {col} by {shift} to ensure positive values for log transform")
                    
                    new_col = f"log_{col}"
                    if base == 'e':
                        result[new_col] = np.log(result[col])
                        logger.info(f"Created natural log feature {new_col}")
                    else:
                        result[new_col] = np.log10(result[col])
                        logger.info(f"Created log10 feature {new_col}")
            
            elif op_type == 'date_features':
                column = operation.get('column')
                features = operation.get('features', ['year', 'month', 'day', 'dayofweek'])
                
                if column not in result.columns:
                    logger.warning(f"Column {column} not found for date features")
                    continue
                
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(result[column]):
                    try:
                        result[column] = pd.to_datetime(result[column])
                    except:
                        logger.warning(f"Could not convert column {column} to datetime")
                        continue
                
                # Extract date features
                for feature in features:
                    if feature == 'year':
                        result[f"{column}_year"] = result[column].dt.year
                    elif feature == 'month':
                        result[f"{column}_month"] = result[column].dt.month
                    elif feature == 'day':
                        result[f"{column}_day"] = result[column].dt.day
                    elif feature == 'dayofweek':
                        result[f"{column}_dayofweek"] = result[column].dt.dayofweek
                    elif feature == 'quarter':
                        result[f"{column}_quarter"] = result[column].dt.quarter
                    elif feature == 'is_weekend':
                        result[f"{column}_is_weekend"] = result[column].dt.dayofweek >= 5
                
                logger.info(f"Created date features from {column}")
            
            elif op_type == 'text_features':
                column = operation.get('column')
                features = operation.get('features', ['length', 'word_count'])
                
                if column not in result.columns:
                    logger.warning(f"Column {column} not found for text features")
                    continue
                
                if result[column].dtype != 'object':
                    logger.warning(f"Column {column} is not text type")
                    continue
                
                # Extract text features
                for feature in features:
                    if feature == 'length':
                        result[f"{column}_length"] = result[column].str.len()
                    elif feature == 'word_count':
                        result[f"{column}_word_count"] = result[column].str.split().str.len()
                    elif feature == 'uppercase_count':
                        result[f"{column}_uppercase_count"] = result[column].str.count(r'[A-Z]')
                    elif feature == 'lowercase_count':
                        result[f"{column}_lowercase_count"] = result[column].str.count(r'[a-z]')
                    elif feature == 'digit_count':
                        result[f"{column}_digit_count"] = result[column].str.count(r'[0-9]')
                
                logger.info(f"Created text features from {column}")
            
            else:
                logger.warning(f"Unknown feature engineering operation type: {op_type}")
        
        logger.info(f"Feature engineering complete. New shape: {result.shape}")
        return result
