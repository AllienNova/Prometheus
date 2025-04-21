"""
Advanced data visualization module for scientific research.

This module provides classes for creating publication-quality scientific visualizations
with advanced customization options for research papers and presentations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from scipy import stats
import io
import base64

# Configure logging
logger = logging.getLogger(__name__)

class PublicationVisualizer:
    """
    Class for creating publication-quality scientific visualizations.
    
    This class provides methods for creating high-quality visualizations suitable
    for research papers, presentations, and publications with advanced customization
    options and statistical annotations.
    """
    
    def __init__(self, style: str = 'whitegrid',
                context: str = 'paper',
                palette: str = 'deep',
                font_family: str = 'sans-serif',
                font_scale: float = 1.0,
                figure_size: Tuple[float, float] = (8, 6),
                dpi: int = 300):
        """
        Initialize the PublicationVisualizer.
        
        Args:
            style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
            context: Seaborn context ('paper', 'notebook', 'talk', 'poster')
            palette: Color palette name
            font_family: Font family for text elements
            font_scale: Scale factor for font sizes
            figure_size: Default figure size in inches (width, height)
            dpi: Resolution in dots per inch
        """
        self.style = style
        self.context = context
        self.palette = palette
        self.font_family = font_family
        self.font_scale = font_scale
        self.figure_size = figure_size
        self.dpi = dpi
        
        # Set up the visualization style
        self._setup_style()
        
        logger.info(f"Initialized PublicationVisualizer with style '{style}', context '{context}', and palette '{palette}'")
    
    def _setup_style(self) -> None:
        """Set up the visualization style using seaborn and matplotlib."""
        # Set seaborn style
        sns.set_style(self.style)
        sns.set_context(self.context, font_scale=self.font_scale)
        sns.set_palette(self.palette)
        
        # Set matplotlib parameters
        plt.rcParams['font.family'] = self.font_family
        plt.rcParams['figure.figsize'] = self.figure_size
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['savefig.dpi'] = self.dpi
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.1
    
    def create_figure(self, nrows: int = 1, 
                     ncols: int = 1, 
                     figsize: Optional[Tuple[float, float]] = None,
                     sharex: bool = False,
                     sharey: bool = False,
                     constrained_layout: bool = True,
                     **kwargs) -> Tuple[Figure, Union[Axes, np.ndarray]]:
        """
        Create a new matplotlib figure and axes.
        
        Args:
            nrows: Number of rows in the subplot grid
            ncols: Number of columns in the subplot grid
            figsize: Figure size in inches (width, height)
            sharex: Whether to share x-axes among subplots
            sharey: Whether to share y-axes among subplots
            constrained_layout: Whether to use constrained layout for better spacing
            **kwargs: Additional keyword arguments for plt.subplots
            
        Returns:
            Tuple of (Figure, Axes or array of Axes)
        """
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize or self.figure_size,
            sharex=sharex,
            sharey=sharey,
            constrained_layout=constrained_layout,
            **kwargs
        )
        
        return fig, axes
    
    def line_plot(self, data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray],
                 x: Optional[Union[str, List[float], np.ndarray]] = None,
                 y: Optional[Union[str, List[str]]] = None,
                 hue: Optional[str] = None,
                 style: Optional[str] = None,
                 markers: bool = False,
                 dashes: bool = True,
                 title: Optional[str] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 xlim: Optional[Tuple[float, float]] = None,
                 ylim: Optional[Tuple[float, float]] = None,
                 legend_title: Optional[str] = None,
                 palette: Optional[Union[str, List[str]]] = None,
                 figsize: Optional[Tuple[float, float]] = None,
                 ax: Optional[Axes] = None,
                 ci: Optional[Union[int, str]] = 95,
                 err_style: str = 'band',
                 grid: bool = True,
                 save_path: Optional[str] = None,
                 show_plot: bool = True,
                 **kwargs) -> Tuple[Figure, Axes]:
        """
        Create a publication-quality line plot.
        
        Args:
            data: Data to plot (DataFrame, dictionary, or array)
            x: x-axis data or column name
            y: y-axis data or column name(s)
            hue: Variable for color grouping
            style: Variable for line style grouping
            markers: Whether to include markers on lines
            dashes: Whether to use different dash patterns for lines
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            xlim: x-axis limits (min, max)
            ylim: y-axis limits (min, max)
            legend_title: Title for the legend
            palette: Color palette name or list of colors
            figsize: Figure size in inches (width, height)
            ax: Existing axes to plot on
            ci: Confidence interval level (0-100) or 'sd' for standard deviation
            err_style: Error bar style ('band', 'bars')
            grid: Whether to show grid lines
            save_path: Path to save the figure (png, jpg, pdf, svg)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for seaborn.lineplot
            
        Returns:
            Tuple of (Figure, Axes)
        """
        # Process data based on type
        if isinstance(data, dict):
            # Convert dictionary to DataFrame
            if x is None:
                # Use keys as x-axis
                df = pd.DataFrame({
                    'x': np.repeat(list(data.keys()), [len(v) if hasattr(v, '__len__') else 1 for v in data.values()]),
                    'y': np.concatenate([np.array(v).flatten() if hasattr(v, '__len__') else [v] for v in data.values()])
                })
                x = 'x'
                y = 'y'
            else:
                # Use specified x data
                df = pd.DataFrame(data)
        
        elif isinstance(data, np.ndarray):
            # Convert array to DataFrame
            if data.ndim == 1:
                # 1D array, use as y-axis
                if x is None:
                    # Use range as x-axis
                    df = pd.DataFrame({
                        'x': np.arange(len(data)),
                        'y': data
                    })
                    x = 'x'
                    y = 'y'
                else:
                    # Use provided x data
                    df = pd.DataFrame({
                        'x': x,
                        'y': data
                    })
                    x = 'x'
                    y = 'y'
            
            elif data.ndim == 2:
                # 2D array
                if x is None:
                    # Use first column as x-axis
                    df = pd.DataFrame(data, columns=[f'Column{i}' for i in range(data.shape[1])])
                    x = 'Column0'
                    if y is None:
                        # Use remaining columns as y-axis
                        y = [f'Column{i}' for i in range(1, data.shape[1])]
                else:
                    # Use provided x data
                    df = pd.DataFrame(data, columns=[f'Column{i}' for i in range(data.shape[1])])
                    if y is None:
                        # Use all columns as y-axis
                        y = [f'Column{i}' for i in range(data.shape[1])]
            
            else:
                raise ValueError("NumPy array must be 1D or 2D")
        
        else:
            # Use DataFrame as is
            df = data
        
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        else:
            fig = ax.figure
        
        # Create line plot
        if isinstance(y, (list, tuple)) and len(y) > 1 and hue is None:
            # Multiple y columns, melt DataFrame
            id_vars = [x] if x is not None else []
            if hue is not None:
                id_vars.append(hue)
            if style is not None:
                id_vars.append(style)
            
            # Melt DataFrame to long format
            df_melt = pd.melt(df, id_vars=id_vars, value_vars=y, var_name='variable', value_name='value')
            
            # Create line plot with melted DataFrame
            sns.lineplot(
                data=df_melt,
                x=x,
                y='value',
                hue='variable',
                style=style,
                markers=markers,
                dashes=dashes,
                ci=ci,
                err_style=err_style,
                palette=palette,
                ax=ax,
                **kwargs
            )
        else:
            # Single y column or hue is specified
            sns.lineplot(
                data=df,
                x=x,
                y=y,
                hue=hue,
                style=style,
                markers=markers,
                dashes=dashes,
                ci=ci,
                err_style=err_style,
                palette=palette,
                ax=ax,
                **kwargs
            )
        
        # Set plot attributes
        if title is not None:
            ax.set_title(title, fontweight='bold')
        
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        
        if xlim is not None:
            ax.set_xlim(xlim)
        
        if ylim is not None:
            ax.set_ylim(ylim)
        
        if legend_title is not None:
            if ax.get_legend() is not None:
                ax.get_legend().set_title(legend_title)
        
        # Set grid
        ax.grid(grid, linestyle='--', alpha=0.7)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig, ax
    
    def scatter_plot(self, data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray],
                    x: Union[str, List[float], np.ndarray],
                    y: Union[str, List[float], np.ndarray],
                    hue: Optional[str] = None,
                    size: Optional[Union[str, List[float]]] = None,
                    style: Optional[str] = None,
                    title: Optional[str] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    xlim: Optional[Tuple[float, float]] = None,
                    ylim: Optional[Tuple[float, float]] = None,
                    legend_title: Optional[str] = None,
                    palette: Optional[Union[str, List[str]]] = None,
                    figsize: Optional[Tuple[float, float]] = None,
                    ax: Optional[Axes] = None,
                    alpha: float = 0.8,
                    grid: bool = True,
                    add_regression: bool = False,
                    reg_ci: int = 95,
                    add_corr: bool = False,
                    corr_method: str = 'pearson',
                    save_path: Optional[str] = None,
                    show_plot: bool = True,
                    **kwargs) -> Tuple[Figure, Axes]:
        """
        Create a publication-quality scatter plot.
        
        Args:
            data: Data to plot (DataFrame, dictionary, or array)
            x: x-axis data or column name
            y: y-axis data or column name
            hue: Variable for color grouping
            size: Variable for point size
            style: Variable for point style
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            xlim: x-axis limits (min, max)
            ylim: y-axis limits (min, max)
            legend_title: Title for the legend
            palette: Color palette name or list of colors
            figsize: Figure size in inches (width, height)
            ax: Existing axes to plot on
            alpha: Transparency level (0-1)
            grid: Whether to show grid lines
            add_regression: Whether to add regression line
            reg_ci: Confidence interval for regression line (0-100)
            add_corr: Whether to add correlation coefficient annotation
            corr_method: Correlation method ('pearson', 'spearman', 'kendall')
            save_path: Path to save the figure (png, jpg, pdf, svg)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for seaborn.scatterplot
            
        Returns:
            Tuple of (Figure, Axes)
        """
        # Process data based on type
        if isinstance(data, dict):
            # Convert dictionary to DataFrame
            df = pd.DataFrame(data)
        
        elif isinstance(data, np.ndarray):
            # Convert array to DataFrame
            if data.ndim == 2 and data.shape[1] >= 2:
                # Use first two columns as x and y
                df = pd.DataFrame(data, columns=[f'Column{i}' for i in range(data.shape[1])])
                if isinstance(x, str) and x not in df.columns:
                    x = 'Column0'
                if isinstance(y, str) and y not in df.columns:
                    y = 'Column1'
            else:
                raise ValueError("NumPy array must have at least 2 columns for scatter plot")
        
        else:
            # Use DataFrame as is
            df = data
        
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        else:
            fig = ax.figure
        
        # Create scatter plot
        if add_regression:
            # Use regplot for regression line
            sns.regplot(
                data=df,
                x=x,
                y=y,
                scatter=True,
                ci=reg_ci,
                scatter_kws={'alpha': alpha},
                line_kws={'color': 'red'},
                ax=ax
            )
            
            # Add additional scatter plot if hue, size, or style is specified
            if hue is not None or size is not None or style is not None:
                sns.scatterplot(
                    data=df,
                    x=x,
                    y=y,
                    hue=hue,
                    size=size,
                    style=style,
                    palette=palette,
                    alpha=alpha,
                    ax=ax,
                    **kwargs
                )
        else:
            # Use scatterplot
            sns.scatterplot(
                data=df,
                x=x,
                y=y,
                hue=hue,
                size=size,
                style=style,
                palette=palette,
                alpha=alpha,
                ax=ax,
                **kwargs
            )
        
        # Add correlation coefficient annotation
        if add_corr:
            # Extract data
            x_data = df[x] if isinstance(x, str) else x
            y_data = df[y] if isinstance(y, str) else y
            
            # Calculate correlation coefficient
            if corr_method == 'pearson':
                corr, p_value = stats.pearsonr(x_data, y_data)
                corr_name = "Pearson's r"
            elif corr_method == 'spearman':
                corr, p_value = stats.spearmanr(x_data, y_data)
                corr_name = "Spearman's ρ"
            elif corr_method == 'kendall':
                corr, p_value = stats.kendalltau(x_data, y_data)
                corr_name = "Kendall's τ"
            else:
                raise ValueError(f"Unknown correlation method: {corr_method}")
            
            # Add annotation
            significance = ''
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            
            ax.annotate(
                f"{corr_name} = {corr:.3f}{significance}\np = {p_value:.3e}",
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                ha='left',
                va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
            )
        
        # Set plot attributes
        if title is not None:
            ax.set_title(title, fontweight='bold')
        
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        
        if xlim is not None:
            ax.set_xlim(xlim)
        
        if ylim is not None:
            ax.set_ylim(ylim)
        
        if legend_title is not None:
            if ax.get_legend() is not None:
                ax.get_legend().set_title(legend_title)
        
        # Set grid
        ax.grid(grid, linestyle='--', alpha=0.7)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig, ax
    
    def bar_plot(self, data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray],
               x: Optional[Union[str, List[Any]]] = None,
               y: Optional[Union[str, List[float]]] = None,
               hue: Optional[str] = None,
               title: Optional[str] = None,
               xlabel: Optional[str] = None,
               ylabel: Optional[str] = None,
               xlim: Optional[Tuple[float, float]] = None,
               ylim: Optional[Tuple[float, float]] = None,
               legend_title: Optional[str] = None,
               palette: Optional[Union[str, List[str]]] = None,
               figsize: Optional[Tuple[float, float]] = None,
               ax: Optional[Axes] = None,
               orientation: str = 'vertical',
               error_bars: Optional[str] = 'sd',
               grid: bool = True,
               add_values: bool = False,
               value_format: str = '{:.1f}',
               save_path: Optional[str] = None,
               show_plot: bool = True,
               **kwargs) -> Tuple[Figure, Axes]:
        """
        Create a publication-quality bar plot.
        
        Args:
            data: Data to plot (DataFrame, dictionary, or array)
            x: x-axis data or column name
            y: y-axis data or column name
            hue: Variable for color grouping
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            xlim: x-axis limits (min, max)
            ylim: y-axis limits (min, max)
            legend_title: Title for the legend
            palette: Color palette name or list of colors
            figsize: Figure size in inches (width, height)
            ax: Existing axes to plot on
            orientation: Bar orientation ('vertical' or 'horizontal')
            error_bars: Error bar type ('sd', 'se', 'ci', None)
            grid: Whether to show grid lines
            add_values: Whether to add value labels on bars
            value_format: Format string for value labels
            save_path: Path to save the figure (png, jpg, pdf, svg)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for seaborn.barplot
            
        Returns:
            Tuple of (Figure, Axes)
        """
        # Process data based on type
        if isinstance(data, dict):
            # Convert dictionary to DataFrame
            if x is None and y is None:
                # Use keys as x-axis and values as y-axis
                df = pd.DataFrame({
                    'x': list(data.keys()),
                    'y': list(data.values())
                })
                x = 'x'
                y = 'y'
            else:
                # Use specified x and y
                df = pd.DataFrame(data)
        
        elif isinstance(data, np.ndarray):
            # Convert array to DataFrame
            if data.ndim == 1:
                # 1D array, use as y-axis
                if x is None:
                    # Use range as x-axis
                    df = pd.DataFrame({
                        'x': [f'Item {i}' for i in range(len(data))],
                        'y': data
                    })
                    x = 'x'
                    y = 'y'
                else:
                    # Use provided x data
                    df = pd.DataFrame({
                        'x': x,
                        'y': data
                    })
                    x = 'x'
                    y = 'y'
            
            elif data.ndim == 2:
                # 2D array
                df = pd.DataFrame(data, columns=[f'Column{i}' for i in range(data.shape[1])])
                if x is None:
                    x = 'Column0'
                if y is None:
                    y = 'Column1'
            
            else:
                raise ValueError("NumPy array must be 1D or 2D")
        
        else:
            # Use DataFrame as is
            df = data
        
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        else:
            fig = ax.figure
        
        # Create bar plot
        if orientation == 'vertical':
            # Vertical bars
            bars = sns.barplot(
                data=df,
                x=x,
                y=y,
                hue=hue,
                palette=palette,
                errorbar=error_bars,
                ax=ax,
                **kwargs
            )
        else:
            # Horizontal bars (swap x and y)
            bars = sns.barplot(
                data=df,
                x=y,
                y=x,
                hue=hue,
                palette=palette,
                errorbar=error_bars,
                orient='h',
                ax=ax,
                **kwargs
            )
        
        # Add value labels on bars
        if add_values:
            if orientation == 'vertical':
                # Vertical bars
                for i, p in enumerate(ax.patches):
                    value = p.get_height()
                    ax.annotate(
                        value_format.format(value),
                        (p.get_x() + p.get_width() / 2., value),
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        fontweight='bold',
                        rotation=0
                    )
            else:
                # Horizontal bars
                for i, p in enumerate(ax.patches):
                    value = p.get_width()
                    ax.annotate(
                        value_format.format(value),
                        (value, p.get_y() + p.get_height() / 2.),
                        ha='left',
                        va='center',
                        fontsize=8,
                        fontweight='bold',
                        rotation=0
                    )
        
        # Set plot attributes
        if title is not None:
            ax.set_title(title, fontweight='bold')
        
        if orientation == 'vertical':
            # Vertical bars
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            
            if xlim is not None:
                ax.set_xlim(xlim)
            
            if ylim is not None:
                ax.set_ylim(ylim)
        else:
            # Horizontal bars (swap x and y labels)
            if xlabel is not None:
                ax.set_ylabel(xlabel)
            
            if ylabel is not None:
                ax.set_xlabel(ylabel)
            
            if xlim is not None:
                ax.set_ylim(xlim)
            
            if ylim is not None:
                ax.set_xlim(ylim)
        
        if legend_title is not None:
            if ax.get_legend() is not None:
                ax.get_legend().set_title(legend_title)
        
        # Set grid
        ax.grid(grid, linestyle='--', alpha=0.7)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig, ax
    
    def box_plot(self, data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray],
               x: Optional[Union[str, List[Any]]] = None,
               y: Optional[Union[str, List[float]]] = None,
               hue: Optional[str] = None,
               title: Optional[str] = None,
               xlabel: Optional[str] = None,
               ylabel: Optional[str] = None,
               xlim: Optional[Tuple[float, float]] = None,
               ylim: Optional[Tuple[float, float]] = None,
               legend_title: Optional[str] = None,
               palette: Optional[Union[str, List[str]]] = None,
               figsize: Optional[Tuple[float, float]] = None,
               ax: Optional[Axes] = None,
               orientation: str = 'vertical',
               grid: bool = True,
               add_stripplot: bool = False,
               add_swarmplot: bool = False,
               add_stats: bool = False,
               save_path: Optional[str] = None,
               show_plot: bool = True,
               **kwargs) -> Tuple[Figure, Axes]:
        """
        Create a publication-quality box plot.
        
        Args:
            data: Data to plot (DataFrame, dictionary, or array)
            x: x-axis data or column name
            y: y-axis data or column name
            hue: Variable for color grouping
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            xlim: x-axis limits (min, max)
            ylim: y-axis limits (min, max)
            legend_title: Title for the legend
            palette: Color palette name or list of colors
            figsize: Figure size in inches (width, height)
            ax: Existing axes to plot on
            orientation: Box orientation ('vertical' or 'horizontal')
            grid: Whether to show grid lines
            add_stripplot: Whether to add individual data points as strip plot
            add_swarmplot: Whether to add individual data points as swarm plot
            add_stats: Whether to add statistical annotations
            save_path: Path to save the figure (png, jpg, pdf, svg)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for seaborn.boxplot
            
        Returns:
            Tuple of (Figure, Axes)
        """
        # Process data based on type
        if isinstance(data, dict):
            # Convert dictionary to DataFrame
            if x is None and y is None:
                # Use keys as categories and values as distributions
                categories = []
                distributions = []
                
                for key, values in data.items():
                    if hasattr(values, '__len__'):
                        categories.extend([key] * len(values))
                        distributions.extend(values)
                    else:
                        categories.append(key)
                        distributions.append(values)
                
                df = pd.DataFrame({
                    'category': categories,
                    'value': distributions
                })
                
                if orientation == 'vertical':
                    x = 'category'
                    y = 'value'
                else:
                    y = 'category'
                    x = 'value'
            else:
                # Use specified x and y
                df = pd.DataFrame(data)
        
        elif isinstance(data, np.ndarray):
            # Convert array to DataFrame
            if data.ndim == 1:
                # 1D array, create single box
                df = pd.DataFrame({'value': data})
                if orientation == 'vertical':
                    y = 'value'
                else:
                    x = 'value'
            
            elif data.ndim == 2:
                # 2D array, create box for each column
                df = pd.DataFrame(data, columns=[f'Column{i}' for i in range(data.shape[1])])
                
                # Melt to long format
                df = pd.melt(df, var_name='Column', value_name='Value')
                
                if orientation == 'vertical':
                    x = 'Column'
                    y = 'Value'
                else:
                    y = 'Column'
                    x = 'Value'
            
            else:
                raise ValueError("NumPy array must be 1D or 2D")
        
        else:
            # Use DataFrame as is
            df = data
        
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        else:
            fig = ax.figure
        
        # Create box plot
        if orientation == 'vertical':
            # Vertical boxes
            sns.boxplot(
                data=df,
                x=x,
                y=y,
                hue=hue,
                palette=palette,
                ax=ax,
                **kwargs
            )
            
            # Add individual data points
            if add_stripplot:
                sns.stripplot(
                    data=df,
                    x=x,
                    y=y,
                    hue=hue,
                    palette=palette,
                    dodge=True,
                    alpha=0.5,
                    size=3,
                    ax=ax
                )
            
            if add_swarmplot:
                sns.swarmplot(
                    data=df,
                    x=x,
                    y=y,
                    hue=hue,
                    palette=palette,
                    dodge=True,
                    alpha=0.5,
                    size=3,
                    ax=ax
                )
        else:
            # Horizontal boxes (swap x and y)
            sns.boxplot(
                data=df,
                x=x,
                y=y,
                hue=hue,
                palette=palette,
                orient='h',
                ax=ax,
                **kwargs
            )
            
            # Add individual data points
            if add_stripplot:
                sns.stripplot(
                    data=df,
                    x=x,
                    y=y,
                    hue=hue,
                    palette=palette,
                    dodge=True,
                    alpha=0.5,
                    size=3,
                    ax=ax
                )
            
            if add_swarmplot:
                sns.swarmplot(
                    data=df,
                    x=x,
                    y=y,
                    hue=hue,
                    palette=palette,
                    dodge=True,
                    alpha=0.5,
                    size=3,
                    ax=ax
                )
        
        # Add statistical annotations
        if add_stats and x is not None and y is not None:
            # Get unique categories
            if orientation == 'vertical':
                categories = df[x].unique() if isinstance(x, str) else np.unique(x)
                value_col = y if isinstance(y, str) else 'value'
            else:
                categories = df[y].unique() if isinstance(y, str) else np.unique(y)
                value_col = x if isinstance(x, str) else 'value'
            
            # Calculate statistics for each category
            stats_text = []
            for cat in categories:
                if orientation == 'vertical':
                    values = df[df[x] == cat][value_col]
                else:
                    values = df[df[y] == cat][value_col]
                
                # Calculate statistics
                mean = np.mean(values)
                median = np.median(values)
                std = np.std(values)
                
                # Format statistics
                stats_text.append(f"{cat}:\nMean: {mean:.2f}\nMedian: {median:.2f}\nStd: {std:.2f}")
            
            # Add text annotation
            ax.annotate(
                '\n\n'.join(stats_text),
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                ha='left',
                va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                fontsize=8
            )
        
        # Set plot attributes
        if title is not None:
            ax.set_title(title, fontweight='bold')
        
        if orientation == 'vertical':
            # Vertical boxes
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            
            if xlim is not None:
                ax.set_xlim(xlim)
            
            if ylim is not None:
                ax.set_ylim(ylim)
        else:
            # Horizontal boxes (swap x and y labels)
            if xlabel is not None:
                ax.set_ylabel(xlabel)
            
            if ylabel is not None:
                ax.set_xlabel(ylabel)
            
            if xlim is not None:
                ax.set_ylim(xlim)
            
            if ylim is not None:
                ax.set_xlim(ylim)
        
        if legend_title is not None:
            if ax.get_legend() is not None:
                ax.get_legend().set_title(legend_title)
        
        # Set grid
        ax.grid(grid, linestyle='--', alpha=0.7)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig, ax
    
    def heatmap(self, data: Union[pd.DataFrame, np.ndarray],
              title: Optional[str] = None,
              xlabel: Optional[str] = None,
              ylabel: Optional[str] = None,
              cmap: Optional[str] = 'viridis',
              center: Optional[float] = None,
              vmin: Optional[float] = None,
              vmax: Optional[float] = None,
              figsize: Optional[Tuple[float, float]] = None,
              ax: Optional[Axes] = None,
              cbar_label: Optional[str] = None,
              annot: bool = True,
              fmt: str = '.2f',
              linewidths: float = 0.5,
              linecolor: str = 'white',
              save_path: Optional[str] = None,
              show_plot: bool = True,
              **kwargs) -> Tuple[Figure, Axes]:
        """
        Create a publication-quality heatmap.
        
        Args:
            data: Data to plot (DataFrame or array)
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            cmap: Colormap name
            center: Value at which to center the colormap
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            figsize: Figure size in inches (width, height)
            ax: Existing axes to plot on
            cbar_label: Label for the color bar
            annot: Whether to annotate cells with values
            fmt: Format string for annotations
            linewidths: Width of lines between cells
            linecolor: Color of lines between cells
            save_path: Path to save the figure (png, jpg, pdf, svg)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for seaborn.heatmap
            
        Returns:
            Tuple of (Figure, Axes)
        """
        # Process data based on type
        if isinstance(data, np.ndarray):
            # Convert array to DataFrame
            if data.ndim == 2:
                df = pd.DataFrame(
                    data,
                    index=[f'Row {i}' for i in range(data.shape[0])],
                    columns=[f'Col {i}' for i in range(data.shape[1])]
                )
            else:
                raise ValueError("NumPy array must be 2D for heatmap")
        else:
            # Use DataFrame as is
            df = data
        
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        else:
            fig = ax.figure
        
        # Create heatmap
        hm = sns.heatmap(
            df,
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            annot=annot,
            fmt=fmt,
            linewidths=linewidths,
            linecolor=linecolor,
            cbar_kws={'label': cbar_label} if cbar_label else None,
            ax=ax,
            **kwargs
        )
        
        # Set plot attributes
        if title is not None:
            ax.set_title(title, fontweight='bold')
        
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig, ax
    
    def correlation_matrix(self, data: Union[pd.DataFrame, np.ndarray],
                         method: str = 'pearson',
                         title: Optional[str] = None,
                         cmap: Optional[str] = 'coolwarm',
                         figsize: Optional[Tuple[float, float]] = None,
                         ax: Optional[Axes] = None,
                         annot: bool = True,
                         fmt: str = '.2f',
                         mask_upper: bool = False,
                         mask_diagonal: bool = False,
                         save_path: Optional[str] = None,
                         show_plot: bool = True,
                         **kwargs) -> Tuple[Figure, Axes]:
        """
        Create a publication-quality correlation matrix heatmap.
        
        Args:
            data: Data to plot (DataFrame or array)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            title: Plot title
            cmap: Colormap name
            figsize: Figure size in inches (width, height)
            ax: Existing axes to plot on
            annot: Whether to annotate cells with correlation values
            fmt: Format string for annotations
            mask_upper: Whether to mask the upper triangle
            mask_diagonal: Whether to mask the diagonal
            save_path: Path to save the figure (png, jpg, pdf, svg)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for seaborn.heatmap
            
        Returns:
            Tuple of (Figure, Axes)
        """
        # Process data based on type
        if isinstance(data, np.ndarray):
            # Convert array to DataFrame
            if data.ndim == 2:
                df = pd.DataFrame(
                    data,
                    columns=[f'Var {i}' for i in range(data.shape[1])]
                )
            else:
                raise ValueError("NumPy array must be 2D for correlation matrix")
        else:
            # Use DataFrame as is
            df = data
        
        # Calculate correlation matrix
        corr_matrix = df.corr(method=method)
        
        # Create mask if needed
        mask = None
        if mask_upper or mask_diagonal:
            mask = np.zeros_like(corr_matrix, dtype=bool)
            if mask_upper:
                # Mask upper triangle
                mask[np.triu_indices_from(mask, k=1)] = True
            if mask_diagonal:
                # Mask diagonal
                mask[np.diag_indices_from(mask)] = True
        
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        else:
            fig = ax.figure
        
        # Create heatmap
        hm = sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            center=0,
            annot=annot,
            fmt=fmt,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': f'{method.capitalize()} Correlation'},
            ax=ax,
            **kwargs
        )
        
        # Set plot attributes
        if title is not None:
            ax.set_title(title, fontweight='bold')
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig, ax
    
    def pair_plot(self, data: Union[pd.DataFrame, np.ndarray],
                vars: Optional[List[str]] = None,
                hue: Optional[str] = None,
                title: Optional[str] = None,
                palette: Optional[Union[str, List[str]]] = None,
                figsize: Optional[Tuple[float, float]] = None,
                diag_kind: str = 'kde',
                plot_kind: str = 'scatter',
                markers: Optional[Union[str, List[str]]] = None,
                height: float = 2.5,
                aspect: float = 1,
                save_path: Optional[str] = None,
                show_plot: bool = True,
                **kwargs) -> sns.PairGrid:
        """
        Create a publication-quality pair plot (scatter plot matrix).
        
        Args:
            data: Data to plot (DataFrame or array)
            vars: List of variables to include
            hue: Variable for color grouping
            title: Plot title
            palette: Color palette name or list of colors
            figsize: Figure size in inches (width, height)
            diag_kind: Kind of plot for diagonal ('hist', 'kde')
            plot_kind: Kind of plot for off-diagonal ('scatter', 'reg')
            markers: Marker style(s) for scatter plots
            height: Height (in inches) of each facet
            aspect: Aspect ratio of each facet
            save_path: Path to save the figure (png, jpg, pdf, svg)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for seaborn.pairplot
            
        Returns:
            Seaborn PairGrid object
        """
        # Process data based on type
        if isinstance(data, np.ndarray):
            # Convert array to DataFrame
            if data.ndim == 2:
                df = pd.DataFrame(
                    data,
                    columns=[f'Var {i}' for i in range(data.shape[1])]
                )
                if vars is None:
                    vars = df.columns.tolist()
            else:
                raise ValueError("NumPy array must be 2D for pair plot")
        else:
            # Use DataFrame as is
            df = data
        
        # Create pair plot
        g = sns.pairplot(
            df,
            vars=vars,
            hue=hue,
            palette=palette,
            diag_kind=diag_kind,
            kind=plot_kind,
            markers=markers,
            height=height,
            aspect=aspect,
            **kwargs
        )
        
        # Set plot attributes
        if title is not None:
            g.fig.suptitle(title, fontweight='bold', y=1.02)
        
        # Adjust layout
        g.fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(g.fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(g.fig)
        
        return g
    
    def violin_plot(self, data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray],
                  x: Optional[Union[str, List[Any]]] = None,
                  y: Optional[Union[str, List[float]]] = None,
                  hue: Optional[str] = None,
                  title: Optional[str] = None,
                  xlabel: Optional[str] = None,
                  ylabel: Optional[str] = None,
                  xlim: Optional[Tuple[float, float]] = None,
                  ylim: Optional[Tuple[float, float]] = None,
                  legend_title: Optional[str] = None,
                  palette: Optional[Union[str, List[str]]] = None,
                  figsize: Optional[Tuple[float, float]] = None,
                  ax: Optional[Axes] = None,
                  orientation: str = 'vertical',
                  grid: bool = True,
                  inner: str = 'box',
                  split: bool = False,
                  add_boxplot: bool = False,
                  add_stripplot: bool = False,
                  save_path: Optional[str] = None,
                  show_plot: bool = True,
                  **kwargs) -> Tuple[Figure, Axes]:
        """
        Create a publication-quality violin plot.
        
        Args:
            data: Data to plot (DataFrame, dictionary, or array)
            x: x-axis data or column name
            y: y-axis data or column name
            hue: Variable for color grouping
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            xlim: x-axis limits (min, max)
            ylim: y-axis limits (min, max)
            legend_title: Title for the legend
            palette: Color palette name or list of colors
            figsize: Figure size in inches (width, height)
            ax: Existing axes to plot on
            orientation: Violin orientation ('vertical' or 'horizontal')
            grid: Whether to show grid lines
            inner: Inner representation ('box', 'quartile', 'point', 'stick', None)
            split: Whether to split violins when hue is specified
            add_boxplot: Whether to overlay box plots
            add_stripplot: Whether to overlay strip plots
            save_path: Path to save the figure (png, jpg, pdf, svg)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for seaborn.violinplot
            
        Returns:
            Tuple of (Figure, Axes)
        """
        # Process data based on type
        if isinstance(data, dict):
            # Convert dictionary to DataFrame
            if x is None and y is None:
                # Use keys as categories and values as distributions
                categories = []
                distributions = []
                
                for key, values in data.items():
                    if hasattr(values, '__len__'):
                        categories.extend([key] * len(values))
                        distributions.extend(values)
                    else:
                        categories.append(key)
                        distributions.append(values)
                
                df = pd.DataFrame({
                    'category': categories,
                    'value': distributions
                })
                
                if orientation == 'vertical':
                    x = 'category'
                    y = 'value'
                else:
                    y = 'category'
                    x = 'value'
            else:
                # Use specified x and y
                df = pd.DataFrame(data)
        
        elif isinstance(data, np.ndarray):
            # Convert array to DataFrame
            if data.ndim == 1:
                # 1D array, create single violin
                df = pd.DataFrame({'value': data})
                if orientation == 'vertical':
                    y = 'value'
                else:
                    x = 'value'
            
            elif data.ndim == 2:
                # 2D array, create violin for each column
                df = pd.DataFrame(data, columns=[f'Column{i}' for i in range(data.shape[1])])
                
                # Melt to long format
                df = pd.melt(df, var_name='Column', value_name='Value')
                
                if orientation == 'vertical':
                    x = 'Column'
                    y = 'Value'
                else:
                    y = 'Column'
                    x = 'Value'
            
            else:
                raise ValueError("NumPy array must be 1D or 2D")
        
        else:
            # Use DataFrame as is
            df = data
        
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        else:
            fig = ax.figure
        
        # Create violin plot
        if orientation == 'vertical':
            # Vertical violins
            sns.violinplot(
                data=df,
                x=x,
                y=y,
                hue=hue,
                palette=palette,
                inner=inner,
                split=split,
                ax=ax,
                **kwargs
            )
            
            # Add box plot overlay
            if add_boxplot:
                sns.boxplot(
                    data=df,
                    x=x,
                    y=y,
                    hue=hue,
                    palette=palette,
                    width=0.15,
                    boxprops={'zorder': 2, 'alpha': 0.7},
                    ax=ax
                )
            
            # Add strip plot overlay
            if add_stripplot:
                sns.stripplot(
                    data=df,
                    x=x,
                    y=y,
                    hue=hue,
                    palette=palette,
                    dodge=True,
                    alpha=0.5,
                    size=3,
                    ax=ax
                )
        else:
            # Horizontal violins (swap x and y)
            sns.violinplot(
                data=df,
                x=x,
                y=y,
                hue=hue,
                palette=palette,
                inner=inner,
                split=split,
                orient='h',
                ax=ax,
                **kwargs
            )
            
            # Add box plot overlay
            if add_boxplot:
                sns.boxplot(
                    data=df,
                    x=x,
                    y=y,
                    hue=hue,
                    palette=palette,
                    width=0.15,
                    orient='h',
                    boxprops={'zorder': 2, 'alpha': 0.7},
                    ax=ax
                )
            
            # Add strip plot overlay
            if add_stripplot:
                sns.stripplot(
                    data=df,
                    x=x,
                    y=y,
                    hue=hue,
                    palette=palette,
                    dodge=True,
                    alpha=0.5,
                    size=3,
                    orient='h',
                    ax=ax
                )
        
        # Set plot attributes
        if title is not None:
            ax.set_title(title, fontweight='bold')
        
        if orientation == 'vertical':
            # Vertical violins
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            
            if xlim is not None:
                ax.set_xlim(xlim)
            
            if ylim is not None:
                ax.set_ylim(ylim)
        else:
            # Horizontal violins (swap x and y labels)
            if xlabel is not None:
                ax.set_ylabel(xlabel)
            
            if ylabel is not None:
                ax.set_xlabel(ylabel)
            
            if xlim is not None:
                ax.set_ylim(xlim)
            
            if ylim is not None:
                ax.set_xlim(ylim)
        
        if legend_title is not None:
            if ax.get_legend() is not None:
                ax.get_legend().set_title(legend_title)
        
        # Set grid
        ax.grid(grid, linestyle='--', alpha=0.7)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig, ax
    
    def histogram(self, data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray, List[float]],
                column: Optional[str] = None,
                bins: int = 30,
                title: Optional[str] = None,
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = 'Frequency',
                xlim: Optional[Tuple[float, float]] = None,
                ylim: Optional[Tuple[float, float]] = None,
                color: Optional[str] = None,
                figsize: Optional[Tuple[float, float]] = None,
                ax: Optional[Axes] = None,
                grid: bool = True,
                kde: bool = False,
                rug: bool = False,
                add_mean_line: bool = False,
                add_median_line: bool = False,
                add_stats: bool = False,
                save_path: Optional[str] = None,
                show_plot: bool = True,
                **kwargs) -> Tuple[Figure, Axes]:
        """
        Create a publication-quality histogram.
        
        Args:
            data: Data to plot (DataFrame, dictionary, array, or list)
            column: Column name for DataFrame input
            bins: Number of bins
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            xlim: x-axis limits (min, max)
            ylim: y-axis limits (min, max)
            color: Bar color
            figsize: Figure size in inches (width, height)
            ax: Existing axes to plot on
            grid: Whether to show grid lines
            kde: Whether to add kernel density estimate
            rug: Whether to add rug plot
            add_mean_line: Whether to add vertical line at mean
            add_median_line: Whether to add vertical line at median
            add_stats: Whether to add statistical annotations
            save_path: Path to save the figure (png, jpg, pdf, svg)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for seaborn.histplot
            
        Returns:
            Tuple of (Figure, Axes)
        """
        # Process data based on type
        if isinstance(data, pd.DataFrame):
            # DataFrame input
            if column is not None:
                if column in data.columns:
                    plot_data = data[column]
                    x_name = column
                else:
                    raise ValueError(f"Column '{column}' not found in DataFrame")
            else:
                # Use first numeric column
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    plot_data = data[numeric_cols[0]]
                    x_name = numeric_cols[0]
                    logger.info(f"No column specified, using first numeric column: {x_name}")
                else:
                    raise ValueError("No numeric columns found in DataFrame")
        
        elif isinstance(data, dict):
            # Dictionary input
            if column is not None:
                if column in data:
                    plot_data = data[column]
                    x_name = column
                else:
                    raise ValueError(f"Key '{column}' not found in dictionary")
            else:
                # Use first key
                first_key = list(data.keys())[0]
                plot_data = data[first_key]
                x_name = first_key
                logger.info(f"No key specified, using first key: {x_name}")
        
        elif isinstance(data, (np.ndarray, list)):
            # Array or list input
            plot_data = data
            x_name = 'value'
        
        else:
            raise ValueError("Data must be a pandas DataFrame, dictionary, NumPy array, or list")
        
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        else:
            fig = ax.figure
        
        # Create histogram
        sns.histplot(
            plot_data,
            bins=bins,
            color=color,
            kde=kde,
            ax=ax,
            **kwargs
        )
        
        # Add rug plot
        if rug:
            sns.rugplot(
                plot_data,
                color=color,
                alpha=0.5,
                ax=ax
            )
        
        # Add mean line
        if add_mean_line:
            mean_value = np.mean(plot_data)
            ax.axvline(
                mean_value,
                color='red',
                linestyle='--',
                linewidth=1.5,
                label=f'Mean: {mean_value:.2f}'
            )
        
        # Add median line
        if add_median_line:
            median_value = np.median(plot_data)
            ax.axvline(
                median_value,
                color='green',
                linestyle='-.',
                linewidth=1.5,
                label=f'Median: {median_value:.2f}'
            )
        
        # Add statistical annotations
        if add_stats:
            # Calculate statistics
            mean = np.mean(plot_data)
            median = np.median(plot_data)
            std = np.std(plot_data)
            min_val = np.min(plot_data)
            max_val = np.max(plot_data)
            q1 = np.percentile(plot_data, 25)
            q3 = np.percentile(plot_data, 75)
            
            # Format statistics
            stats_text = (
                f"Mean: {mean:.2f}\n"
                f"Median: {median:.2f}\n"
                f"Std Dev: {std:.2f}\n"
                f"Min: {min_val:.2f}\n"
                f"Max: {max_val:.2f}\n"
                f"Q1: {q1:.2f}\n"
                f"Q3: {q3:.2f}"
            )
            
            # Add text annotation
            ax.annotate(
                stats_text,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                ha='left',
                va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
            )
        
        # Set plot attributes
        if title is not None:
            ax.set_title(title, fontweight='bold')
        
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x_name)
        
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        
        if xlim is not None:
            ax.set_xlim(xlim)
        
        if ylim is not None:
            ax.set_ylim(ylim)
        
        # Add legend if needed
        if add_mean_line or add_median_line:
            ax.legend()
        
        # Set grid
        ax.grid(grid, linestyle='--', alpha=0.7)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig, ax
    
    def multi_panel_figure(self, nrows: int = 2,
                         ncols: int = 2,
                         figsize: Optional[Tuple[float, float]] = None,
                         sharex: bool = False,
                         sharey: bool = False,
                         panel_labels: Optional[List[str]] = None,
                         panel_label_params: Optional[Dict[str, Any]] = None,
                         title: Optional[str] = None,
                         tight_layout: bool = True,
                         save_path: Optional[str] = None,
                         show_plot: bool = True) -> Tuple[Figure, np.ndarray]:
        """
        Create a multi-panel figure for publication.
        
        Args:
            nrows: Number of rows
            ncols: Number of columns
            figsize: Figure size in inches (width, height)
            sharex: Whether to share x-axes among subplots
            sharey: Whether to share y-axes among subplots
            panel_labels: List of labels for panels (e.g., ['A', 'B', 'C', 'D'])
            panel_label_params: Dictionary of parameters for panel labels
            title: Figure title
            tight_layout: Whether to use tight layout
            save_path: Path to save the figure (png, jpg, pdf, svg)
            show_plot: Whether to display the plot
            
        Returns:
            Tuple of (Figure, array of Axes)
        """
        # Calculate figure size if not provided
        if figsize is None:
            figsize = (self.figure_size[0] * ncols, self.figure_size[1] * nrows)
        
        # Create figure and axes
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey
        )
        
        # Set default panel label parameters
        default_label_params = {
            'fontsize': 14,
            'fontweight': 'bold',
            'ha': 'left',
            'va': 'top',
            'xy': (-0.1, 1.1),
            'xycoords': 'axes fraction'
        }
        
        # Update with user-provided parameters
        if panel_label_params is not None:
            default_label_params.update(panel_label_params)
        
        # Add panel labels if provided
        if panel_labels is not None:
            # Ensure axes is always a 2D array
            if nrows == 1 and ncols == 1:
                axes_array = np.array([[axes]])
            elif nrows == 1:
                axes_array = np.array([axes])
            elif ncols == 1:
                axes_array = np.array([[ax] for ax in axes])
            else:
                axes_array = axes
            
            # Add labels to each panel
            for i in range(nrows):
                for j in range(ncols):
                    idx = i * ncols + j
                    if idx < len(panel_labels):
                        axes_array[i, j].annotate(
                            panel_labels[idx],
                            **default_label_params
                        )
        
        # Set figure title
        if title is not None:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        if tight_layout:
            fig.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig, axes
    
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
    
    def figure_to_base64(self, fig: Figure, format: str = 'png', dpi: Optional[int] = None) -> str:
        """
        Convert figure to base64 encoded string.
        
        Args:
            fig: Matplotlib Figure object
            format: Image format ('png', 'jpg', 'svg', 'pdf')
            dpi: Resolution in dots per inch
            
        Returns:
            Base64 encoded string
        """
        # Save figure to in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format=format, dpi=dpi or self.dpi, bbox_inches='tight')
        buf.seek(0)
        
        # Encode as base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f"data:image/{format};base64,{img_str}"
