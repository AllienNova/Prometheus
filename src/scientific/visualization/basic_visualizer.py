"""
Basic visualization module for scientific data.

This module provides functions for creating common scientific visualizations
including line plots, scatter plots, bar charts, histograms, and heatmaps.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
import seaborn as sns
import os
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Set default style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper")

class BasicVisualizer:
    """
    Class for creating basic scientific visualizations.
    
    This class provides methods for creating common visualizations such as
    line plots, scatter plots, bar charts, histograms, and heatmaps.
    """
    
    def __init__(self, theme: str = 'default', 
                figure_size: Tuple[float, float] = (10, 6),
                dpi: int = 100,
                font_family: str = 'sans-serif',
                font_size: int = 12):
        """
        Initialize the BasicVisualizer.
        
        Args:
            theme: Visual theme for plots ('default', 'dark', 'light', 'colorblind')
            figure_size: Default figure size in inches (width, height)
            dpi: Resolution in dots per inch
            font_family: Font family for text elements
            font_size: Base font size for text elements
        """
        self.figure_size = figure_size
        self.dpi = dpi
        self.font_family = font_family
        self.font_size = font_size
        
        # Set theme
        self._set_theme(theme)
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = figure_size
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['font.family'] = font_family
        plt.rcParams['font.size'] = font_size
        
        logger.info(f"Initialized BasicVisualizer with theme '{theme}' and figure size {figure_size}")
    
    def _set_theme(self, theme: str) -> None:
        """
        Set the visual theme for plots.
        
        Args:
            theme: Theme name ('default', 'dark', 'light', 'colorblind')
        """
        self.theme = theme
        
        if theme == 'default':
            # Use seaborn default style
            sns.set_style("whitegrid")
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['text.color'] = 'black'
            plt.rcParams['axes.labelcolor'] = 'black'
            plt.rcParams['xtick.color'] = 'black'
            plt.rcParams['ytick.color'] = 'black'
            
        elif theme == 'dark':
            # Dark theme
            sns.set_style("darkgrid")
            plt.rcParams['axes.facecolor'] = '#2E3440'
            plt.rcParams['figure.facecolor'] = '#2E3440'
            plt.rcParams['text.color'] = 'white'
            plt.rcParams['axes.labelcolor'] = 'white'
            plt.rcParams['xtick.color'] = 'white'
            plt.rcParams['ytick.color'] = 'white'
            plt.rcParams['grid.color'] = '#3B4252'
            
        elif theme == 'light':
            # Light theme
            sns.set_style("whitegrid")
            plt.rcParams['axes.facecolor'] = '#F8F9FA'
            plt.rcParams['figure.facecolor'] = '#FFFFFF'
            plt.rcParams['text.color'] = '#212529'
            plt.rcParams['axes.labelcolor'] = '#212529'
            plt.rcParams['xtick.color'] = '#212529'
            plt.rcParams['ytick.color'] = '#212529'
            plt.rcParams['grid.color'] = '#E9ECEF'
            
        elif theme == 'colorblind':
            # Colorblind-friendly theme
            sns.set_style("whitegrid")
            plt.rcParams['axes.prop_cycle'] = plt.cycler('color', 
                                                        ['#0072B2', '#E69F00', '#009E73', 
                                                         '#CC79A7', '#56B4E9', '#F0E442', 
                                                         '#D55E00'])
        else:
            logger.warning(f"Unknown theme: {theme}. Using default theme.")
            self._set_theme('default')
    
    def line_plot(self, data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray],
                 x: Optional[Union[str, List[float], np.ndarray]] = None,
                 y: Optional[Union[str, List[str]]] = None,
                 title: str = 'Line Plot',
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 legend_title: Optional[str] = None,
                 color_palette: Optional[str] = None,
                 figsize: Optional[Tuple[float, float]] = None,
                 grid: bool = True,
                 save_path: Optional[str] = None,
                 show_plot: bool = True,
                 **kwargs) -> Figure:
        """
        Create a line plot.
        
        Args:
            data: Data to plot (DataFrame, dictionary, or array)
            x: x-axis data or column name
            y: y-axis data or column name(s)
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            legend_title: Title for the legend
            color_palette: Color palette name
            figsize: Figure size (width, height) in inches
            grid: Whether to show grid lines
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for plt.plot()
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        
        # Process data based on type
        if isinstance(data, pd.DataFrame):
            # DataFrame input
            if x is None:
                # Use index as x-axis
                x_data = data.index
            else:
                # Use specified column as x-axis
                x_data = data[x]
            
            if y is None:
                # Use all numeric columns as y-axis
                y_columns = data.select_dtypes(include=[np.number]).columns
            elif isinstance(y, str):
                # Use single column as y-axis
                y_columns = [y]
            else:
                # Use multiple columns as y-axis
                y_columns = y
            
            # Plot each y column
            for col in y_columns:
                ax.plot(x_data, data[col], label=col, **kwargs)
            
        elif isinstance(data, dict):
            # Dictionary input
            if x is None:
                # Use range as x-axis
                max_len = max(len(v) for v in data.values())
                x_data = np.arange(max_len)
            elif isinstance(x, str) and x in data:
                # Use specified key as x-axis
                x_data = data[x]
                # Remove x from keys to plot
                keys_to_plot = [k for k in data.keys() if k != x]
            else:
                # Use provided x data
                x_data = x
                keys_to_plot = list(data.keys())
            
            # Plot each key
            if y is None:
                # Plot all keys
                for key, values in data.items():
                    if key != x:  # Skip x key if it's in the dict
                        ax.plot(x_data, values, label=key, **kwargs)
            elif isinstance(y, str):
                # Plot single key
                if y in data:
                    ax.plot(x_data, data[y], label=y, **kwargs)
                else:
                    logger.warning(f"Key '{y}' not found in data dictionary")
            else:
                # Plot multiple keys
                for key in y:
                    if key in data:
                        ax.plot(x_data, data[key], label=key, **kwargs)
                    else:
                        logger.warning(f"Key '{key}' not found in data dictionary")
        
        elif isinstance(data, np.ndarray):
            # NumPy array input
            if data.ndim == 1:
                # 1D array, use as y-axis
                if x is None:
                    # Use range as x-axis
                    x_data = np.arange(len(data))
                else:
                    # Use provided x data
                    x_data = x
                
                ax.plot(x_data, data, **kwargs)
                
            elif data.ndim == 2:
                # 2D array
                if x is None:
                    # Use first column as x-axis
                    x_data = data[:, 0]
                    # Use remaining columns as y-axis
                    for i in range(1, data.shape[1]):
                        ax.plot(x_data, data[:, i], label=f'Series {i}', **kwargs)
                else:
                    # Use provided x data
                    x_data = x
                    # Use all columns as y-axis
                    for i in range(data.shape[1]):
                        ax.plot(x_data, data[:, i], label=f'Series {i}', **kwargs)
            
            else:
                raise ValueError("NumPy array must be 1D or 2D")
        
        else:
            raise ValueError("Data must be a pandas DataFrame, dictionary, or NumPy array")
        
        # Set plot attributes
        ax.set_title(title, fontsize=self.font_size + 2)
        
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=self.font_size)
        
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=self.font_size)
        
        # Set grid
        ax.grid(grid)
        
        # Set color palette if specified
        if color_palette is not None:
            sns.set_palette(color_palette)
        
        # Add legend if there are multiple lines
        if ax.get_legend_handles_labels()[0]:
            if legend_title is not None:
                ax.legend(title=legend_title)
            else:
                ax.legend()
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def scatter_plot(self, data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray],
                    x: Union[str, List[float], np.ndarray],
                    y: Union[str, List[float], np.ndarray],
                    hue: Optional[Union[str, List[Any]]] = None,
                    size: Optional[Union[str, List[float]]] = None,
                    title: str = 'Scatter Plot',
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    legend_title: Optional[str] = None,
                    color_palette: Optional[str] = None,
                    figsize: Optional[Tuple[float, float]] = None,
                    alpha: float = 0.7,
                    grid: bool = True,
                    save_path: Optional[str] = None,
                    show_plot: bool = True,
                    **kwargs) -> Figure:
        """
        Create a scatter plot.
        
        Args:
            data: Data to plot (DataFrame, dictionary, or array)
            x: x-axis data or column name
            y: y-axis data or column name
            hue: Variable for color mapping
            size: Variable for size mapping
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            legend_title: Title for the legend
            color_palette: Color palette name
            figsize: Figure size (width, height) in inches
            alpha: Transparency of points
            grid: Whether to show grid lines
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for plt.scatter()
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        
        # Process data based on type
        if isinstance(data, pd.DataFrame):
            # DataFrame input
            if isinstance(x, str) and x in data.columns:
                x_data = data[x]
            else:
                raise ValueError(f"Column '{x}' not found in DataFrame")
            
            if isinstance(y, str) and y in data.columns:
                y_data = data[y]
            else:
                raise ValueError(f"Column '{y}' not found in DataFrame")
            
            # Handle hue parameter
            if hue is not None:
                if isinstance(hue, str) and hue in data.columns:
                    hue_data = data[hue]
                    
                    # Set color palette
                    if color_palette is None:
                        if hue_data.dtype.kind in 'ifc':  # Numeric data
                            color_palette = 'viridis'
                        else:  # Categorical data
                            color_palette = 'tab10'
                    
                    # Create scatter plot with hue
                    scatter = sns.scatterplot(x=x_data, y=y_data, hue=hue_data, 
                                             size=data[size] if size in data.columns else None,
                                             palette=color_palette, alpha=alpha, ax=ax, **kwargs)
                    
                    # Set legend title
                    if legend_title is None:
                        legend_title = hue
                    
                    if scatter.get_legend() is not None:
                        scatter.get_legend().set_title(legend_title)
                else:
                    logger.warning(f"Column '{hue}' not found in DataFrame, ignoring hue parameter")
                    scatter = ax.scatter(x_data, y_data, alpha=alpha, **kwargs)
            else:
                # Simple scatter plot
                scatter = ax.scatter(x_data, y_data, alpha=alpha, **kwargs)
        
        elif isinstance(data, dict):
            # Dictionary input
            if isinstance(x, str) and x in data:
                x_data = data[x]
            else:
                x_data = x
            
            if isinstance(y, str) and y in data:
                y_data = data[y]
            else:
                y_data = y
            
            # Handle hue parameter
            if hue is not None:
                if isinstance(hue, str) and hue in data:
                    hue_data = data[hue]
                    
                    # Set color palette
                    if color_palette is None:
                        if isinstance(hue_data[0], (int, float)):  # Numeric data
                            color_palette = 'viridis'
                        else:  # Categorical data
                            color_palette = 'tab10'
                    
                    # Create scatter plot with hue
                    scatter = sns.scatterplot(x=x_data, y=y_data, hue=hue_data, 
                                             size=data[size] if isinstance(size, str) and size in data else None,
                                             palette=color_palette, alpha=alpha, ax=ax, **kwargs)
                    
                    # Set legend title
                    if legend_title is None:
                        legend_title = hue
                    
                    if scatter.get_legend() is not None:
                        scatter.get_legend().set_title(legend_title)
                else:
                    logger.warning(f"Key '{hue}' not found in data dictionary, ignoring hue parameter")
                    scatter = ax.scatter(x_data, y_data, alpha=alpha, **kwargs)
            else:
                # Simple scatter plot
                scatter = ax.scatter(x_data, y_data, alpha=alpha, **kwargs)
        
        elif isinstance(data, np.ndarray):
            # NumPy array input
            if data.ndim == 2 and data.shape[1] >= 2:
                # Use first two columns as x and y
                x_data = data[:, 0] if x is None else x
                y_data = data[:, 1] if y is None else y
                
                # Handle hue parameter
                if hue is not None and data.shape[1] >= 3:
                    hue_data = data[:, 2]
                    
                    # Set color palette
                    if color_palette is None:
                        color_palette = 'viridis'
                    
                    # Create scatter plot with hue
                    scatter = sns.scatterplot(x=x_data, y=y_data, hue=hue_data, 
                                             size=data[:, 3] if size is None and data.shape[1] >= 4 else size,
                                             palette=color_palette, alpha=alpha, ax=ax, **kwargs)
                    
                    # Set legend title
                    if legend_title is not None and scatter.get_legend() is not None:
                        scatter.get_legend().set_title(legend_title)
                else:
                    # Simple scatter plot
                    scatter = ax.scatter(x_data, y_data, alpha=alpha, **kwargs)
            else:
                raise ValueError("NumPy array must have at least 2 columns for scatter plot")
        
        else:
            raise ValueError("Data must be a pandas DataFrame, dictionary, or NumPy array")
        
        # Set plot attributes
        ax.set_title(title, fontsize=self.font_size + 2)
        
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=self.font_size)
        elif isinstance(x, str):
            ax.set_xlabel(x, fontsize=self.font_size)
        
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=self.font_size)
        elif isinstance(y, str):
            ax.set_ylabel(y, fontsize=self.font_size)
        
        # Set grid
        ax.grid(grid)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def bar_plot(self, data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray],
               x: Optional[Union[str, List[Any]]] = None,
               y: Optional[Union[str, List[float]]] = None,
               hue: Optional[Union[str, List[Any]]] = None,
               title: str = 'Bar Plot',
               xlabel: Optional[str] = None,
               ylabel: Optional[str] = None,
               legend_title: Optional[str] = None,
               color_palette: Optional[str] = None,
               figsize: Optional[Tuple[float, float]] = None,
               orientation: str = 'vertical',
               grid: bool = True,
               save_path: Optional[str] = None,
               show_plot: bool = True,
               **kwargs) -> Figure:
        """
        Create a bar plot.
        
        Args:
            data: Data to plot (DataFrame, dictionary, or array)
            x: x-axis data or column name
            y: y-axis data or column name
            hue: Variable for color grouping
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            legend_title: Title for the legend
            color_palette: Color palette name
            figsize: Figure size (width, height) in inches
            orientation: Bar orientation ('vertical' or 'horizontal')
            grid: Whether to show grid lines
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for sns.barplot()
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        
        # Set color palette if specified
        if color_palette is not None:
            sns.set_palette(color_palette)
        
        # Process data based on type
        if isinstance(data, pd.DataFrame):
            # DataFrame input
            if orientation == 'vertical':
                # Vertical bars
                if hue is not None:
                    bar = sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax, **kwargs)
                else:
                    bar = sns.barplot(x=x, y=y, data=data, ax=ax, **kwargs)
            else:
                # Horizontal bars
                if hue is not None:
                    bar = sns.barplot(x=y, y=x, hue=hue, data=data, ax=ax, **kwargs)
                else:
                    bar = sns.barplot(x=y, y=x, data=data, ax=ax, **kwargs)
        
        elif isinstance(data, dict):
            # Dictionary input
            if x is None:
                # Use keys as x-axis
                x_data = list(data.keys())
                y_data = list(data.values())
            else:
                x_data = x
                if isinstance(y, str) and y in data:
                    y_data = data[y]
                else:
                    y_data = y
            
            # Convert to DataFrame for seaborn
            if isinstance(y_data[0], (list, np.ndarray)) and hue is not None:
                # Multiple series with hue
                df = pd.DataFrame()
                for i, series in enumerate(y_data):
                    temp_df = pd.DataFrame({
                        'x': x_data,
                        'y': series,
                        'hue': [hue[i]] * len(series) if isinstance(hue, list) else [hue] * len(series)
                    })
                    df = pd.concat([df, temp_df])
                
                if orientation == 'vertical':
                    bar = sns.barplot(x='x', y='y', hue='hue', data=df, ax=ax, **kwargs)
                else:
                    bar = sns.barplot(x='y', y='x', hue='hue', data=df, ax=ax, **kwargs)
            else:
                # Single series
                if orientation == 'vertical':
                    bar = sns.barplot(x=x_data, y=y_data, ax=ax, **kwargs)
                else:
                    bar = sns.barplot(x=y_data, y=x_data, ax=ax, **kwargs)
        
        elif isinstance(data, np.ndarray):
            # NumPy array input
            if data.ndim == 1:
                # 1D array, use as y-axis
                if x is None:
                    # Use range as x-axis
                    x_data = np.arange(len(data))
                else:
                    # Use provided x data
                    x_data = x
                
                if orientation == 'vertical':
                    bar = sns.barplot(x=x_data, y=data, ax=ax, **kwargs)
                else:
                    bar = sns.barplot(x=data, y=x_data, ax=ax, **kwargs)
                
            elif data.ndim == 2:
                # 2D array
                if hue is not None:
                    # Convert to DataFrame for seaborn
                    df = pd.DataFrame()
                    for i in range(data.shape[1]):
                        temp_df = pd.DataFrame({
                            'x': np.arange(data.shape[0]) if x is None else x,
                            'y': data[:, i],
                            'hue': [f'Series {i}'] * data.shape[0] if isinstance(hue, bool) else [hue[i]] * data.shape[0]
                        })
                        df = pd.concat([df, temp_df])
                    
                    if orientation == 'vertical':
                        bar = sns.barplot(x='x', y='y', hue='hue', data=df, ax=ax, **kwargs)
                    else:
                        bar = sns.barplot(x='y', y='x', hue='hue', data=df, ax=ax, **kwargs)
                else:
                    # Use first column as x-axis
                    x_data = np.arange(data.shape[0]) if x is None else x
                    
                    if orientation == 'vertical':
                        bar = sns.barplot(x=x_data, y=data[:, 0], ax=ax, **kwargs)
                    else:
                        bar = sns.barplot(x=data[:, 0], y=x_data, ax=ax, **kwargs)
            
            else:
                raise ValueError("NumPy array must be 1D or 2D")
        
        else:
            raise ValueError("Data must be a pandas DataFrame, dictionary, or NumPy array")
        
        # Set plot attributes
        ax.set_title(title, fontsize=self.font_size + 2)
        
        if xlabel is not None:
            if orientation == 'vertical':
                ax.set_xlabel(xlabel, fontsize=self.font_size)
            else:
                ax.set_ylabel(xlabel, fontsize=self.font_size)
        
        if ylabel is not None:
            if orientation == 'vertical':
                ax.set_ylabel(ylabel, fontsize=self.font_size)
            else:
                ax.set_xlabel(ylabel, fontsize=self.font_size)
        
        # Set grid
        ax.grid(grid, axis='y' if orientation == 'vertical' else 'x')
        
        # Set legend title if provided
        if legend_title is not None and ax.get_legend() is not None:
            ax.get_legend().set_title(legend_title)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def histogram(self, data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray, List[float]],
                 column: Optional[str] = None,
                 bins: int = 10,
                 title: str = 'Histogram',
                 xlabel: Optional[str] = None,
                 ylabel: str = 'Frequency',
                 kde: bool = False,
                 color: Optional[str] = None,
                 figsize: Optional[Tuple[float, float]] = None,
                 grid: bool = True,
                 save_path: Optional[str] = None,
                 show_plot: bool = True,
                 **kwargs) -> Figure:
        """
        Create a histogram.
        
        Args:
            data: Data to plot (DataFrame, dictionary, array, or list)
            column: Column name for DataFrame input
            bins: Number of bins
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            kde: Whether to show kernel density estimate
            color: Bar color
            figsize: Figure size (width, height) in inches
            grid: Whether to show grid lines
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for sns.histplot()
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        
        # Process data based on type
        if isinstance(data, pd.DataFrame):
            # DataFrame input
            if column is not None:
                if column in data.columns:
                    plot_data = data[column]
                else:
                    raise ValueError(f"Column '{column}' not found in DataFrame")
            else:
                # Use first numeric column
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    plot_data = data[numeric_cols[0]]
                    column = numeric_cols[0]
                    logger.info(f"No column specified, using first numeric column: {column}")
                else:
                    raise ValueError("No numeric columns found in DataFrame")
        
        elif isinstance(data, dict):
            # Dictionary input
            if column is not None:
                if column in data:
                    plot_data = data[column]
                else:
                    raise ValueError(f"Key '{column}' not found in dictionary")
            else:
                # Use first key
                first_key = list(data.keys())[0]
                plot_data = data[first_key]
                column = first_key
                logger.info(f"No key specified, using first key: {column}")
        
        elif isinstance(data, (np.ndarray, list)):
            # Array or list input
            plot_data = data
        
        else:
            raise ValueError("Data must be a pandas DataFrame, dictionary, NumPy array, or list")
        
        # Create histogram
        sns.histplot(plot_data, bins=bins, kde=kde, color=color, ax=ax, **kwargs)
        
        # Set plot attributes
        ax.set_title(title, fontsize=self.font_size + 2)
        
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=self.font_size)
        elif column is not None:
            ax.set_xlabel(column, fontsize=self.font_size)
        
        ax.set_ylabel(ylabel, fontsize=self.font_size)
        
        # Set grid
        ax.grid(grid)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def box_plot(self, data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray],
               x: Optional[Union[str, List[Any]]] = None,
               y: Optional[Union[str, List[float]]] = None,
               hue: Optional[Union[str, List[Any]]] = None,
               title: str = 'Box Plot',
               xlabel: Optional[str] = None,
               ylabel: Optional[str] = None,
               legend_title: Optional[str] = None,
               color_palette: Optional[str] = None,
               figsize: Optional[Tuple[float, float]] = None,
               orientation: str = 'vertical',
               grid: bool = True,
               save_path: Optional[str] = None,
               show_plot: bool = True,
               **kwargs) -> Figure:
        """
        Create a box plot.
        
        Args:
            data: Data to plot (DataFrame, dictionary, or array)
            x: x-axis data or column name
            y: y-axis data or column name
            hue: Variable for color grouping
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            legend_title: Title for the legend
            color_palette: Color palette name
            figsize: Figure size (width, height) in inches
            orientation: Box orientation ('vertical' or 'horizontal')
            grid: Whether to show grid lines
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for sns.boxplot()
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        
        # Set color palette if specified
        if color_palette is not None:
            sns.set_palette(color_palette)
        
        # Process data based on type
        if isinstance(data, pd.DataFrame):
            # DataFrame input
            if orientation == 'vertical':
                # Vertical boxes
                if hue is not None:
                    box = sns.boxplot(x=x, y=y, hue=hue, data=data, ax=ax, **kwargs)
                else:
                    box = sns.boxplot(x=x, y=y, data=data, ax=ax, **kwargs)
            else:
                # Horizontal boxes
                if hue is not None:
                    box = sns.boxplot(x=y, y=x, hue=hue, data=data, ax=ax, **kwargs)
                else:
                    box = sns.boxplot(x=y, y=x, data=data, ax=ax, **kwargs)
        
        elif isinstance(data, dict):
            # Dictionary input
            if x is None:
                # Use keys as x-axis
                x_data = list(data.keys())
                y_data = list(data.values())
            else:
                x_data = x
                if isinstance(y, str) and y in data:
                    y_data = data[y]
                else:
                    y_data = y
            
            # Convert to DataFrame for seaborn
            if isinstance(y_data[0], (list, np.ndarray)) and hue is not None:
                # Multiple series with hue
                df = pd.DataFrame()
                for i, series in enumerate(y_data):
                    temp_df = pd.DataFrame({
                        'x': x_data,
                        'y': series,
                        'hue': [hue[i]] * len(series) if isinstance(hue, list) else [hue] * len(series)
                    })
                    df = pd.concat([df, temp_df])
                
                if orientation == 'vertical':
                    box = sns.boxplot(x='x', y='y', hue='hue', data=df, ax=ax, **kwargs)
                else:
                    box = sns.boxplot(x='y', y='x', hue='hue', data=df, ax=ax, **kwargs)
            else:
                # Convert to long format for boxplot
                df = pd.DataFrame()
                for i, key in enumerate(data.keys()):
                    temp_df = pd.DataFrame({
                        'category': [key] * len(data[key]),
                        'value': data[key]
                    })
                    df = pd.concat([df, temp_df])
                
                if orientation == 'vertical':
                    box = sns.boxplot(x='category', y='value', data=df, ax=ax, **kwargs)
                else:
                    box = sns.boxplot(x='value', y='category', data=df, ax=ax, **kwargs)
        
        elif isinstance(data, np.ndarray):
            # NumPy array input
            if data.ndim == 1:
                # 1D array, create single box
                if orientation == 'vertical':
                    box = sns.boxplot(y=data, ax=ax, **kwargs)
                else:
                    box = sns.boxplot(x=data, ax=ax, **kwargs)
                
            elif data.ndim == 2:
                # 2D array, create box for each column
                # Convert to DataFrame
                df = pd.DataFrame(data, columns=[f'Column {i}' for i in range(data.shape[1])])
                
                # Melt to long format
                df_melt = pd.melt(df, var_name='Column', value_name='Value')
                
                if orientation == 'vertical':
                    box = sns.boxplot(x='Column', y='Value', data=df_melt, ax=ax, **kwargs)
                else:
                    box = sns.boxplot(x='Value', y='Column', data=df_melt, ax=ax, **kwargs)
            
            else:
                raise ValueError("NumPy array must be 1D or 2D")
        
        else:
            raise ValueError("Data must be a pandas DataFrame, dictionary, or NumPy array")
        
        # Set plot attributes
        ax.set_title(title, fontsize=self.font_size + 2)
        
        if xlabel is not None:
            if orientation == 'vertical':
                ax.set_xlabel(xlabel, fontsize=self.font_size)
            else:
                ax.set_ylabel(xlabel, fontsize=self.font_size)
        
        if ylabel is not None:
            if orientation == 'vertical':
                ax.set_ylabel(ylabel, fontsize=self.font_size)
            else:
                ax.set_xlabel(ylabel, fontsize=self.font_size)
        
        # Set grid
        ax.grid(grid, axis='y' if orientation == 'vertical' else 'x')
        
        # Set legend title if provided
        if legend_title is not None and ax.get_legend() is not None:
            ax.get_legend().set_title(legend_title)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def heatmap(self, data: Union[pd.DataFrame, np.ndarray],
              title: str = 'Heatmap',
              xlabel: Optional[str] = None,
              ylabel: Optional[str] = None,
              cmap: str = 'viridis',
              annot: bool = True,
              fmt: str = '.2f',
              linewidths: float = 0.5,
              figsize: Optional[Tuple[float, float]] = None,
              save_path: Optional[str] = None,
              show_plot: bool = True,
              **kwargs) -> Figure:
        """
        Create a heatmap.
        
        Args:
            data: Data to plot (DataFrame or array)
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            cmap: Colormap name
            annot: Whether to annotate cells with values
            fmt: String formatting code for annotations
            linewidths: Width of lines between cells
            figsize: Figure size (width, height) in inches
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for sns.heatmap()
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        
        # Process data based on type
        if isinstance(data, pd.DataFrame):
            # DataFrame input
            plot_data = data
        
        elif isinstance(data, np.ndarray):
            # NumPy array input
            if data.ndim == 2:
                # Create DataFrame with default indices
                plot_data = pd.DataFrame(
                    data,
                    index=[f'Row {i}' for i in range(data.shape[0])],
                    columns=[f'Col {i}' for i in range(data.shape[1])]
                )
            else:
                raise ValueError("NumPy array must be 2D for heatmap")
        
        else:
            raise ValueError("Data must be a pandas DataFrame or 2D NumPy array")
        
        # Create heatmap
        sns.heatmap(plot_data, cmap=cmap, annot=annot, fmt=fmt, 
                   linewidths=linewidths, ax=ax, **kwargs)
        
        # Set plot attributes
        ax.set_title(title, fontsize=self.font_size + 2)
        
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=self.font_size)
        
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=self.font_size)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def correlation_matrix(self, data: Union[pd.DataFrame, np.ndarray],
                         method: str = 'pearson',
                         title: str = 'Correlation Matrix',
                         cmap: str = 'coolwarm',
                         annot: bool = True,
                         fmt: str = '.2f',
                         figsize: Optional[Tuple[float, float]] = None,
                         save_path: Optional[str] = None,
                         show_plot: bool = True,
                         **kwargs) -> Figure:
        """
        Create a correlation matrix heatmap.
        
        Args:
            data: Data to plot (DataFrame or array)
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            title: Plot title
            cmap: Colormap name
            annot: Whether to annotate cells with values
            fmt: String formatting code for annotations
            figsize: Figure size (width, height) in inches
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for sns.heatmap()
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        
        # Process data based on type
        if isinstance(data, pd.DataFrame):
            # DataFrame input
            # Calculate correlation matrix
            corr = data.corr(method=method)
        
        elif isinstance(data, np.ndarray):
            # NumPy array input
            if data.ndim == 2:
                # Convert to DataFrame with default column names
                df = pd.DataFrame(
                    data,
                    columns=[f'Feature {i}' for i in range(data.shape[1])]
                )
                # Calculate correlation matrix
                corr = df.corr(method=method)
            else:
                raise ValueError("NumPy array must be 2D for correlation matrix")
        
        else:
            raise ValueError("Data must be a pandas DataFrame or 2D NumPy array")
        
        # Create heatmap
        sns.heatmap(corr, cmap=cmap, annot=annot, fmt=fmt, 
                   linewidths=0.5, ax=ax, **kwargs)
        
        # Set plot attributes
        ax.set_title(title, fontsize=self.font_size + 2)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def pair_plot(self, data: Union[pd.DataFrame, np.ndarray],
                columns: Optional[List[str]] = None,
                hue: Optional[str] = None,
                title: str = 'Pair Plot',
                diag_kind: str = 'kde',
                color_palette: Optional[str] = None,
                figsize: Optional[Tuple[float, float]] = None,
                save_path: Optional[str] = None,
                show_plot: bool = True,
                **kwargs) -> Figure:
        """
        Create a pair plot (scatter plot matrix).
        
        Args:
            data: Data to plot (DataFrame or array)
            columns: List of columns to include
            hue: Variable for color mapping
            title: Plot title
            diag_kind: Kind of plot for diagonal ('hist' or 'kde')
            color_palette: Color palette name
            figsize: Figure size (width, height) in inches
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for sns.pairplot()
            
        Returns:
            Matplotlib Figure object
        """
        # Process data based on type
        if isinstance(data, pd.DataFrame):
            # DataFrame input
            plot_data = data
            
            # Filter columns if specified
            if columns is not None:
                plot_data = plot_data[columns]
        
        elif isinstance(data, np.ndarray):
            # NumPy array input
            if data.ndim == 2:
                # Convert to DataFrame with default column names
                col_names = [f'Feature {i}' for i in range(data.shape[1])]
                plot_data = pd.DataFrame(data, columns=col_names)
                
                # Filter columns if specified
                if columns is not None:
                    valid_cols = [col for col in columns if col in col_names]
                    if not valid_cols:
                        raise ValueError("No valid columns specified")
                    plot_data = plot_data[valid_cols]
            else:
                raise ValueError("NumPy array must be 2D for pair plot")
        
        else:
            raise ValueError("Data must be a pandas DataFrame or 2D NumPy array")
        
        # Set color palette if specified
        if color_palette is not None:
            sns.set_palette(color_palette)
        
        # Create pair plot
        g = sns.pairplot(plot_data, hue=hue, diag_kind=diag_kind, **kwargs)
        
        # Set title
        g.fig.suptitle(title, fontsize=self.font_size + 4, y=1.02)
        
        # Adjust figure size if specified
        if figsize is not None:
            g.fig.set_size_inches(figsize)
        
        # Adjust layout
        g.fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(g.fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return g.fig
    
    def _save_figure(self, fig: Figure, save_path: str) -> None:
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib Figure object
            save_path: Path to save the figure
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Save figure
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
