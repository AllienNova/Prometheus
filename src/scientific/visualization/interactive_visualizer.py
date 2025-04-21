"""
Interactive visualization module for scientific data.

This module provides classes for creating interactive scientific visualizations
using Plotly, including interactive plots, dashboards, and 3D visualizations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging
import os
import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

# Configure logging
logger = logging.getLogger(__name__)

# Set default plotly template
pio.templates.default = "plotly_white"

class InteractiveVisualizer:
    """
    Class for creating interactive scientific visualizations using Plotly.
    
    This class provides methods for creating interactive visualizations such as
    line plots, scatter plots, bar charts, 3D plots, and dashboards.
    """
    
    def __init__(self, theme: str = 'default',
                width: int = 900,
                height: int = 600,
                font_family: str = 'Arial, sans-serif',
                font_size: int = 12):
        """
        Initialize the InteractiveVisualizer.
        
        Args:
            theme: Visual theme for plots ('default', 'dark', 'light', 'presentation')
            width: Default plot width in pixels
            height: Default plot height in pixels
            font_family: Font family for text elements
            font_size: Base font size for text elements
        """
        self.width = width
        self.height = height
        self.font_family = font_family
        self.font_size = font_size
        
        # Set theme
        self._set_theme(theme)
        
        logger.info(f"Initialized InteractiveVisualizer with theme '{theme}' and dimensions {width}x{height}")
    
    def _set_theme(self, theme: str) -> None:
        """
        Set the visual theme for plots.
        
        Args:
            theme: Theme name ('default', 'dark', 'light', 'presentation')
        """
        self.theme = theme
        
        if theme == 'default':
            pio.templates.default = "plotly_white"
            self.colorscale = 'Viridis'
            self.paper_bgcolor = 'white'
            self.plot_bgcolor = 'white'
            self.font_color = 'black'
            self.grid_color = 'lightgray'
            
        elif theme == 'dark':
            pio.templates.default = "plotly_dark"
            self.colorscale = 'Plasma'
            self.paper_bgcolor = '#2E3440'
            self.plot_bgcolor = '#2E3440'
            self.font_color = 'white'
            self.grid_color = '#3B4252'
            
        elif theme == 'light':
            pio.templates.default = "plotly_white"
            self.colorscale = 'Blues'
            self.paper_bgcolor = '#F8F9FA'
            self.plot_bgcolor = '#F8F9FA'
            self.font_color = '#212529'
            self.grid_color = '#E9ECEF'
            
        elif theme == 'presentation':
            pio.templates.default = "plotly"
            self.colorscale = 'Rainbow'
            self.paper_bgcolor = 'white'
            self.plot_bgcolor = 'white'
            self.font_color = 'black'
            self.grid_color = 'lightgray'
            
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
                 color_sequence: Optional[List[str]] = None,
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 save_path: Optional[str] = None,
                 show_plot: bool = True,
                 **kwargs) -> go.Figure:
        """
        Create an interactive line plot.
        
        Args:
            data: Data to plot (DataFrame, dictionary, or array)
            x: x-axis data or column name
            y: y-axis data or column name(s)
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            legend_title: Title for the legend
            color_sequence: List of colors for lines
            width: Plot width in pixels
            height: Plot height in pixels
            save_path: Path to save the figure (html or png/jpg/pdf)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for plotly express
            
        Returns:
            Plotly Figure object
        """
        # Process data based on type
        if isinstance(data, pd.DataFrame):
            # DataFrame input
            if x is None:
                # Use index as x-axis
                x_data = data.index
                x_name = 'index'
            else:
                # Use specified column as x-axis
                x_data = data[x]
                x_name = x
            
            if y is None:
                # Use all numeric columns as y-axis
                y_columns = data.select_dtypes(include=[np.number]).columns
            elif isinstance(y, str):
                # Use single column as y-axis
                y_columns = [y]
            else:
                # Use multiple columns as y-axis
                y_columns = y
            
            # Create figure
            fig = go.Figure()
            
            # Add traces for each y column
            for i, col in enumerate(y_columns):
                color = color_sequence[i % len(color_sequence)] if color_sequence else None
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=data[col],
                        mode='lines',
                        name=col,
                        line=dict(color=color),
                        **kwargs
                    )
                )
            
        elif isinstance(data, dict):
            # Dictionary input
            if x is None:
                # Use range as x-axis
                max_len = max(len(v) for v in data.values())
                x_data = np.arange(max_len)
                x_name = 'index'
            elif isinstance(x, str) and x in data:
                # Use specified key as x-axis
                x_data = data[x]
                x_name = x
                # Remove x from keys to plot
                keys_to_plot = [k for k in data.keys() if k != x]
            else:
                # Use provided x data
                x_data = x
                x_name = 'x'
                keys_to_plot = list(data.keys())
            
            # Create figure
            fig = go.Figure()
            
            # Add traces for each key
            if y is None:
                # Plot all keys
                for i, key in enumerate(data.keys()):
                    if key != x or not isinstance(x, str):  # Skip x key if it's in the dict
                        color = color_sequence[i % len(color_sequence)] if color_sequence else None
                        fig.add_trace(
                            go.Scatter(
                                x=x_data,
                                y=data[key],
                                mode='lines',
                                name=key,
                                line=dict(color=color),
                                **kwargs
                            )
                        )
            elif isinstance(y, str):
                # Plot single key
                if y in data:
                    fig.add_trace(
                        go.Scatter(
                            x=x_data,
                            y=data[y],
                            mode='lines',
                            name=y,
                            **kwargs
                        )
                    )
                else:
                    logger.warning(f"Key '{y}' not found in data dictionary")
            else:
                # Plot multiple keys
                for i, key in enumerate(y):
                    if key in data:
                        color = color_sequence[i % len(color_sequence)] if color_sequence else None
                        fig.add_trace(
                            go.Scatter(
                                x=x_data,
                                y=data[key],
                                mode='lines',
                                name=key,
                                line=dict(color=color),
                                **kwargs
                            )
                        )
                    else:
                        logger.warning(f"Key '{key}' not found in data dictionary")
        
        elif isinstance(data, np.ndarray):
            # NumPy array input
            if data.ndim == 1:
                # 1D array, use as y-axis
                if x is None:
                    # Use range as x-axis
                    x_data = np.arange(len(data))
                    x_name = 'index'
                else:
                    # Use provided x data
                    x_data = x
                    x_name = 'x'
                
                # Create figure
                fig = go.Figure()
                
                # Add trace
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=data,
                        mode='lines',
                        name='data',
                        **kwargs
                    )
                )
                
            elif data.ndim == 2:
                # 2D array
                if x is None:
                    # Use first column as x-axis
                    x_data = data[:, 0]
                    x_name = 'Column 0'
                    # Use remaining columns as y-axis
                    y_data = data[:, 1:]
                    y_names = [f'Column {i}' for i in range(1, data.shape[1])]
                else:
                    # Use provided x data
                    x_data = x
                    x_name = 'x'
                    # Use all columns as y-axis
                    y_data = data
                    y_names = [f'Column {i}' for i in range(data.shape[1])]
                
                # Create figure
                fig = go.Figure()
                
                # Add traces for each column
                for i in range(y_data.shape[1]):
                    color = color_sequence[i % len(color_sequence)] if color_sequence else None
                    fig.add_trace(
                        go.Scatter(
                            x=x_data,
                            y=y_data[:, i],
                            mode='lines',
                            name=y_names[i],
                            line=dict(color=color),
                            **kwargs
                        )
                    )
            
            else:
                raise ValueError("NumPy array must be 1D or 2D")
        
        else:
            raise ValueError("Data must be a pandas DataFrame, dictionary, or NumPy array")
        
        # Set plot attributes
        fig.update_layout(
            title=title,
            xaxis_title=xlabel if xlabel is not None else x_name,
            yaxis_title=ylabel,
            legend_title=legend_title,
            width=width or self.width,
            height=height or self.height,
            font=dict(
                family=self.font_family,
                size=self.font_size,
                color=self.font_color
            ),
            paper_bgcolor=self.paper_bgcolor,
            plot_bgcolor=self.plot_bgcolor,
            hovermode='closest'
        )
        
        # Update grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.grid_color)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.grid_color)
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            fig.show()
        
        return fig
    
    def scatter_plot(self, data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray],
                    x: Union[str, List[float], np.ndarray],
                    y: Union[str, List[float], np.ndarray],
                    color: Optional[Union[str, List[Any]]] = None,
                    size: Optional[Union[str, List[float]]] = None,
                    text: Optional[Union[str, List[str]]] = None,
                    title: str = 'Scatter Plot',
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    color_label: Optional[str] = None,
                    size_label: Optional[str] = None,
                    colorscale: Optional[str] = None,
                    width: Optional[int] = None,
                    height: Optional[int] = None,
                    save_path: Optional[str] = None,
                    show_plot: bool = True,
                    **kwargs) -> go.Figure:
        """
        Create an interactive scatter plot.
        
        Args:
            data: Data to plot (DataFrame, dictionary, or array)
            x: x-axis data or column name
            y: y-axis data or column name
            color: Variable for color mapping
            size: Variable for size mapping
            text: Variable for hover text
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            color_label: Label for color scale
            size_label: Label for size scale
            colorscale: Colorscale name
            width: Plot width in pixels
            height: Plot height in pixels
            save_path: Path to save the figure (html or png/jpg/pdf)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for plotly express
            
        Returns:
            Plotly Figure object
        """
        # Process data based on type
        if isinstance(data, pd.DataFrame):
            # DataFrame input - use plotly express for simplicity
            fig = px.scatter(
                data,
                x=x,
                y=y,
                color=color,
                size=size,
                text=text,
                title=title,
                labels={
                    x: xlabel if xlabel is not None else x,
                    y: ylabel if ylabel is not None else y,
                    color: color_label if color_label is not None else color,
                    size: size_label if size_label is not None else size
                },
                color_continuous_scale=colorscale or self.colorscale,
                width=width or self.width,
                height=height or self.height,
                **kwargs
            )
            
        elif isinstance(data, dict):
            # Dictionary input
            if isinstance(x, str) and x in data:
                x_data = data[x]
                x_name = x
            else:
                x_data = x
                x_name = 'x'
            
            if isinstance(y, str) and y in data:
                y_data = data[y]
                y_name = y
            else:
                y_data = y
                y_name = 'y'
            
            # Process color data
            if color is not None:
                if isinstance(color, str) and color in data:
                    color_data = data[color]
                    color_name = color
                else:
                    color_data = color
                    color_name = 'color'
            else:
                color_data = None
                color_name = None
            
            # Process size data
            if size is not None:
                if isinstance(size, str) and size in data:
                    size_data = data[size]
                    size_name = size
                else:
                    size_data = size
                    size_name = 'size'
            else:
                size_data = None
                size_name = None
            
            # Process text data
            if text is not None:
                if isinstance(text, str) and text in data:
                    text_data = data[text]
                else:
                    text_data = text
            else:
                text_data = None
            
            # Create figure
            fig = go.Figure()
            
            # Add scatter trace
            scatter_kwargs = {}
            if color_data is not None:
                scatter_kwargs['marker'] = dict(
                    color=color_data,
                    colorscale=colorscale or self.colorscale,
                    showscale=True,
                    colorbar=dict(title=color_label if color_label is not None else color_name)
                )
            
            if size_data is not None:
                if 'marker' not in scatter_kwargs:
                    scatter_kwargs['marker'] = dict()
                scatter_kwargs['marker']['size'] = size_data
            
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    text=text_data,
                    **scatter_kwargs,
                    **kwargs
                )
            )
            
            # Set plot attributes
            fig.update_layout(
                title=title,
                xaxis_title=xlabel if xlabel is not None else x_name,
                yaxis_title=ylabel if ylabel is not None else y_name,
                width=width or self.width,
                height=height or self.height,
                font=dict(
                    family=self.font_family,
                    size=self.font_size,
                    color=self.font_color
                ),
                paper_bgcolor=self.paper_bgcolor,
                plot_bgcolor=self.plot_bgcolor,
                hovermode='closest'
            )
        
        elif isinstance(data, np.ndarray):
            # NumPy array input
            if data.ndim == 2 and data.shape[1] >= 2:
                # Use first two columns as x and y
                x_data = data[:, 0] if x is None else x
                y_data = data[:, 1] if y is None else y
                
                # Process color data
                if color is not None:
                    if data.shape[1] >= 3 and color is None:
                        color_data = data[:, 2]
                        color_name = 'Column 2'
                    else:
                        color_data = color
                        color_name = 'color'
                else:
                    color_data = None
                    color_name = None
                
                # Process size data
                if size is not None:
                    if data.shape[1] >= 4 and size is None:
                        size_data = data[:, 3]
                        size_name = 'Column 3'
                    else:
                        size_data = size
                        size_name = 'size'
                else:
                    size_data = None
                    size_name = None
                
                # Create figure
                fig = go.Figure()
                
                # Add scatter trace
                scatter_kwargs = {}
                if color_data is not None:
                    scatter_kwargs['marker'] = dict(
                        color=color_data,
                        colorscale=colorscale or self.colorscale,
                        showscale=True,
                        colorbar=dict(title=color_label if color_label is not None else color_name)
                    )
                
                if size_data is not None:
                    if 'marker' not in scatter_kwargs:
                        scatter_kwargs['marker'] = dict()
                    scatter_kwargs['marker']['size'] = size_data
                
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='markers',
                        text=text,
                        **scatter_kwargs,
                        **kwargs
                    )
                )
                
                # Set plot attributes
                fig.update_layout(
                    title=title,
                    xaxis_title=xlabel if xlabel is not None else 'X',
                    yaxis_title=ylabel if ylabel is not None else 'Y',
                    width=width or self.width,
                    height=height or self.height,
                    font=dict(
                        family=self.font_family,
                        size=self.font_size,
                        color=self.font_color
                    ),
                    paper_bgcolor=self.paper_bgcolor,
                    plot_bgcolor=self.plot_bgcolor,
                    hovermode='closest'
                )
            else:
                raise ValueError("NumPy array must have at least 2 columns for scatter plot")
        
        else:
            raise ValueError("Data must be a pandas DataFrame, dictionary, or NumPy array")
        
        # Update grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.grid_color)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.grid_color)
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            fig.show()
        
        return fig
    
    def bar_plot(self, data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray],
               x: Optional[Union[str, List[Any]]] = None,
               y: Optional[Union[str, List[float]]] = None,
               color: Optional[Union[str, List[Any]]] = None,
               title: str = 'Bar Plot',
               xlabel: Optional[str] = None,
               ylabel: Optional[str] = None,
               color_label: Optional[str] = None,
               colorscale: Optional[str] = None,
               orientation: str = 'vertical',
               width: Optional[int] = None,
               height: Optional[int] = None,
               save_path: Optional[str] = None,
               show_plot: bool = True,
               **kwargs) -> go.Figure:
        """
        Create an interactive bar plot.
        
        Args:
            data: Data to plot (DataFrame, dictionary, or array)
            x: x-axis data or column name
            y: y-axis data or column name
            color: Variable for color mapping
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            color_label: Label for color scale
            colorscale: Colorscale name
            orientation: Bar orientation ('vertical' or 'horizontal')
            width: Plot width in pixels
            height: Plot height in pixels
            save_path: Path to save the figure (html or png/jpg/pdf)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for plotly express
            
        Returns:
            Plotly Figure object
        """
        # Process data based on type
        if isinstance(data, pd.DataFrame):
            # DataFrame input - use plotly express for simplicity
            if orientation == 'vertical':
                fig = px.bar(
                    data,
                    x=x,
                    y=y,
                    color=color,
                    title=title,
                    labels={
                        x: xlabel if xlabel is not None else x,
                        y: ylabel if ylabel is not None else y,
                        color: color_label if color_label is not None else color
                    },
                    color_continuous_scale=colorscale or self.colorscale,
                    width=width or self.width,
                    height=height or self.height,
                    **kwargs
                )
            else:
                # Horizontal bars (swap x and y)
                fig = px.bar(
                    data,
                    x=y,
                    y=x,
                    color=color,
                    title=title,
                    labels={
                        y: ylabel if ylabel is not None else y,
                        x: xlabel if xlabel is not None else x,
                        color: color_label if color_label is not None else color
                    },
                    color_continuous_scale=colorscale or self.colorscale,
                    orientation='h',
                    width=width or self.width,
                    height=height or self.height,
                    **kwargs
                )
            
        elif isinstance(data, dict):
            # Dictionary input
            if x is None:
                # Use keys as x-axis
                x_data = list(data.keys())
                x_name = 'category'
                
                if y is None:
                    # Use values as y-axis
                    y_data = list(data.values())
                    y_name = 'value'
                else:
                    # Use specified key as y-axis
                    if isinstance(y, str) and y in data:
                        y_data = data[y]
                        y_name = y
                    else:
                        y_data = y
                        y_name = 'y'
            else:
                # Use provided x data
                x_data = x
                x_name = 'x'
                
                if y is None:
                    # Use first key's values as y-axis
                    first_key = list(data.keys())[0]
                    y_data = data[first_key]
                    y_name = first_key
                else:
                    # Use specified key as y-axis
                    if isinstance(y, str) and y in data:
                        y_data = data[y]
                        y_name = y
                    else:
                        y_data = y
                        y_name = 'y'
            
            # Process color data
            if color is not None:
                if isinstance(color, str) and color in data:
                    color_data = data[color]
                    color_name = color
                else:
                    color_data = color
                    color_name = 'color'
            else:
                color_data = None
                color_name = None
            
            # Create figure
            fig = go.Figure()
            
            # Add bar trace
            bar_kwargs = {}
            if color_data is not None:
                bar_kwargs['marker'] = dict(
                    color=color_data,
                    colorscale=colorscale or self.colorscale,
                    showscale=True,
                    colorbar=dict(title=color_label if color_label is not None else color_name)
                )
            
            if orientation == 'vertical':
                fig.add_trace(
                    go.Bar(
                        x=x_data,
                        y=y_data,
                        **bar_kwargs,
                        **kwargs
                    )
                )
            else:
                # Horizontal bars (swap x and y)
                fig.add_trace(
                    go.Bar(
                        x=y_data,
                        y=x_data,
                        orientation='h',
                        **bar_kwargs,
                        **kwargs
                    )
                )
            
            # Set plot attributes
            if orientation == 'vertical':
                fig.update_layout(
                    title=title,
                    xaxis_title=xlabel if xlabel is not None else x_name,
                    yaxis_title=ylabel if ylabel is not None else y_name,
                    width=width or self.width,
                    height=height or self.height,
                    font=dict(
                        family=self.font_family,
                        size=self.font_size,
                        color=self.font_color
                    ),
                    paper_bgcolor=self.paper_bgcolor,
                    plot_bgcolor=self.plot_bgcolor
                )
            else:
                # Horizontal bars (swap axis labels)
                fig.update_layout(
                    title=title,
                    xaxis_title=ylabel if ylabel is not None else y_name,
                    yaxis_title=xlabel if xlabel is not None else x_name,
                    width=width or self.width,
                    height=height or self.height,
                    font=dict(
                        family=self.font_family,
                        size=self.font_size,
                        color=self.font_color
                    ),
                    paper_bgcolor=self.paper_bgcolor,
                    plot_bgcolor=self.plot_bgcolor
                )
        
        elif isinstance(data, np.ndarray):
            # NumPy array input
            if data.ndim == 1:
                # 1D array, use as y-axis
                if x is None:
                    # Use range as x-axis
                    x_data = np.arange(len(data))
                    x_name = 'index'
                else:
                    # Use provided x data
                    x_data = x
                    x_name = 'x'
                
                y_data = data
                y_name = 'value'
                
            elif data.ndim == 2:
                # 2D array
                if data.shape[1] == 1:
                    # Single column, use as y-axis
                    if x is None:
                        # Use range as x-axis
                        x_data = np.arange(len(data))
                        x_name = 'index'
                    else:
                        # Use provided x data
                        x_data = x
                        x_name = 'x'
                    
                    y_data = data[:, 0]
                    y_name = 'Column 0'
                    
                else:
                    # Multiple columns
                    if x is None:
                        # Use first column as x-axis
                        x_data = data[:, 0]
                        x_name = 'Column 0'
                        
                        # Use second column as y-axis
                        y_data = data[:, 1]
                        y_name = 'Column 1'
                        
                        # Use third column as color if available
                        if data.shape[1] >= 3 and color is None:
                            color_data = data[:, 2]
                            color_name = 'Column 2'
                        else:
                            color_data = color
                            color_name = 'color'
                    else:
                        # Use provided x data
                        x_data = x
                        x_name = 'x'
                        
                        # Use first column as y-axis
                        y_data = data[:, 0]
                        y_name = 'Column 0'
                        
                        # Use second column as color if available
                        if data.shape[1] >= 2 and color is None:
                            color_data = data[:, 1]
                            color_name = 'Column 1'
                        else:
                            color_data = color
                            color_name = 'color'
            else:
                raise ValueError("NumPy array must be 1D or 2D")
            
            # Create figure
            fig = go.Figure()
            
            # Add bar trace
            bar_kwargs = {}
            if 'color_data' in locals() and color_data is not None:
                bar_kwargs['marker'] = dict(
                    color=color_data,
                    colorscale=colorscale or self.colorscale,
                    showscale=True,
                    colorbar=dict(title=color_label if color_label is not None else color_name)
                )
            
            if orientation == 'vertical':
                fig.add_trace(
                    go.Bar(
                        x=x_data,
                        y=y_data,
                        **bar_kwargs,
                        **kwargs
                    )
                )
            else:
                # Horizontal bars (swap x and y)
                fig.add_trace(
                    go.Bar(
                        x=y_data,
                        y=x_data,
                        orientation='h',
                        **bar_kwargs,
                        **kwargs
                    )
                )
            
            # Set plot attributes
            if orientation == 'vertical':
                fig.update_layout(
                    title=title,
                    xaxis_title=xlabel if xlabel is not None else x_name,
                    yaxis_title=ylabel if ylabel is not None else y_name,
                    width=width or self.width,
                    height=height or self.height,
                    font=dict(
                        family=self.font_family,
                        size=self.font_size,
                        color=self.font_color
                    ),
                    paper_bgcolor=self.paper_bgcolor,
                    plot_bgcolor=self.plot_bgcolor
                )
            else:
                # Horizontal bars (swap axis labels)
                fig.update_layout(
                    title=title,
                    xaxis_title=ylabel if ylabel is not None else y_name,
                    yaxis_title=xlabel if xlabel is not None else x_name,
                    width=width or self.width,
                    height=height or self.height,
                    font=dict(
                        family=self.font_family,
                        size=self.font_size,
                        color=self.font_color
                    ),
                    paper_bgcolor=self.paper_bgcolor,
                    plot_bgcolor=self.plot_bgcolor
                )
        
        else:
            raise ValueError("Data must be a pandas DataFrame, dictionary, or NumPy array")
        
        # Update grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.grid_color)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.grid_color)
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            fig.show()
        
        return fig
    
    def histogram(self, data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray, List[float]],
                 column: Optional[str] = None,
                 bins: int = 30,
                 title: str = 'Histogram',
                 xlabel: Optional[str] = None,
                 ylabel: str = 'Count',
                 color: Optional[str] = None,
                 opacity: float = 0.7,
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 save_path: Optional[str] = None,
                 show_plot: bool = True,
                 **kwargs) -> go.Figure:
        """
        Create an interactive histogram.
        
        Args:
            data: Data to plot (DataFrame, dictionary, array, or list)
            column: Column name for DataFrame input
            bins: Number of bins
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            color: Bar color
            opacity: Bar opacity
            width: Plot width in pixels
            height: Plot height in pixels
            save_path: Path to save the figure (html or png/jpg/pdf)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for plotly express
            
        Returns:
            Plotly Figure object
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
            
            # Create histogram using plotly express
            fig = px.histogram(
                plot_data,
                nbins=bins,
                title=title,
                opacity=opacity,
                color_discrete_sequence=[color] if color else None,
                width=width or self.width,
                height=height or self.height,
                **kwargs
            )
            
            # Set axis labels
            fig.update_layout(
                xaxis_title=xlabel if xlabel is not None else x_name,
                yaxis_title=ylabel
            )
        
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
            
            # Create figure
            fig = go.Figure()
            
            # Add histogram trace
            fig.add_trace(
                go.Histogram(
                    x=plot_data,
                    nbinsx=bins,
                    marker_color=color,
                    opacity=opacity,
                    **kwargs
                )
            )
            
            # Set plot attributes
            fig.update_layout(
                title=title,
                xaxis_title=xlabel if xlabel is not None else x_name,
                yaxis_title=ylabel,
                width=width or self.width,
                height=height or self.height,
                font=dict(
                    family=self.font_family,
                    size=self.font_size,
                    color=self.font_color
                ),
                paper_bgcolor=self.paper_bgcolor,
                plot_bgcolor=self.plot_bgcolor
            )
        
        elif isinstance(data, (np.ndarray, list)):
            # Array or list input
            plot_data = data
            x_name = 'value'
            
            # Create figure
            fig = go.Figure()
            
            # Add histogram trace
            fig.add_trace(
                go.Histogram(
                    x=plot_data,
                    nbinsx=bins,
                    marker_color=color,
                    opacity=opacity,
                    **kwargs
                )
            )
            
            # Set plot attributes
            fig.update_layout(
                title=title,
                xaxis_title=xlabel if xlabel is not None else x_name,
                yaxis_title=ylabel,
                width=width or self.width,
                height=height or self.height,
                font=dict(
                    family=self.font_family,
                    size=self.font_size,
                    color=self.font_color
                ),
                paper_bgcolor=self.paper_bgcolor,
                plot_bgcolor=self.plot_bgcolor
            )
        
        else:
            raise ValueError("Data must be a pandas DataFrame, dictionary, NumPy array, or list")
        
        # Update grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.grid_color)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.grid_color)
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            fig.show()
        
        return fig
    
    def box_plot(self, data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray],
               x: Optional[Union[str, List[Any]]] = None,
               y: Optional[Union[str, List[float]]] = None,
               color: Optional[Union[str, List[Any]]] = None,
               title: str = 'Box Plot',
               xlabel: Optional[str] = None,
               ylabel: Optional[str] = None,
               color_label: Optional[str] = None,
               orientation: str = 'vertical',
               points: str = 'outliers',
               width: Optional[int] = None,
               height: Optional[int] = None,
               save_path: Optional[str] = None,
               show_plot: bool = True,
               **kwargs) -> go.Figure:
        """
        Create an interactive box plot.
        
        Args:
            data: Data to plot (DataFrame, dictionary, or array)
            x: x-axis data or column name
            y: y-axis data or column name
            color: Variable for color mapping
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            color_label: Label for color legend
            orientation: Box orientation ('vertical' or 'horizontal')
            points: How to show points ('outliers', 'all', 'suspectedoutliers', False)
            width: Plot width in pixels
            height: Plot height in pixels
            save_path: Path to save the figure (html or png/jpg/pdf)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for plotly express
            
        Returns:
            Plotly Figure object
        """
        # Process data based on type
        if isinstance(data, pd.DataFrame):
            # DataFrame input - use plotly express for simplicity
            if orientation == 'vertical':
                fig = px.box(
                    data,
                    x=x,
                    y=y,
                    color=color,
                    title=title,
                    labels={
                        x: xlabel if xlabel is not None else x,
                        y: ylabel if ylabel is not None else y,
                        color: color_label if color_label is not None else color
                    },
                    points=points,
                    width=width or self.width,
                    height=height or self.height,
                    **kwargs
                )
            else:
                # Horizontal boxes (swap x and y)
                fig = px.box(
                    data,
                    x=y,
                    y=x,
                    color=color,
                    title=title,
                    labels={
                        y: ylabel if ylabel is not None else y,
                        x: xlabel if xlabel is not None else x,
                        color: color_label if color_label is not None else color
                    },
                    points=points,
                    width=width or self.width,
                    height=height or self.height,
                    **kwargs
                )
            
        elif isinstance(data, dict):
            # Dictionary input
            if x is None and y is None:
                # Use keys as categories and values as distributions
                categories = []
                distributions = []
                
                for key, values in data.items():
                    categories.extend([key] * len(values))
                    distributions.extend(values)
                
                # Create DataFrame for plotly express
                df = pd.DataFrame({
                    'category': categories,
                    'value': distributions
                })
                
                if orientation == 'vertical':
                    fig = px.box(
                        df,
                        x='category',
                        y='value',
                        color=color,
                        title=title,
                        labels={
                            'category': xlabel if xlabel is not None else 'Category',
                            'value': ylabel if ylabel is not None else 'Value',
                            color: color_label if color_label is not None else color
                        },
                        points=points,
                        width=width or self.width,
                        height=height or self.height,
                        **kwargs
                    )
                else:
                    # Horizontal boxes (swap x and y)
                    fig = px.box(
                        df,
                        x='value',
                        y='category',
                        color=color,
                        title=title,
                        labels={
                            'value': ylabel if ylabel is not None else 'Value',
                            'category': xlabel if xlabel is not None else 'Category',
                            color: color_label if color_label is not None else color
                        },
                        points=points,
                        width=width or self.width,
                        height=height or self.height,
                        **kwargs
                    )
            else:
                # Use specified x and y
                if isinstance(x, str) and x in data:
                    x_data = data[x]
                    x_name = x
                else:
                    x_data = x
                    x_name = 'x'
                
                if isinstance(y, str) and y in data:
                    y_data = data[y]
                    y_name = y
                else:
                    y_data = y
                    y_name = 'y'
                
                # Create figure
                fig = go.Figure()
                
                # Add box trace
                if orientation == 'vertical':
                    fig.add_trace(
                        go.Box(
                            x=x_data,
                            y=y_data,
                            boxpoints=points,
                            **kwargs
                        )
                    )
                else:
                    # Horizontal boxes (swap x and y)
                    fig.add_trace(
                        go.Box(
                            x=y_data,
                            y=x_data,
                            boxpoints=points,
                            **kwargs
                        )
                    )
                
                # Set plot attributes
                if orientation == 'vertical':
                    fig.update_layout(
                        title=title,
                        xaxis_title=xlabel if xlabel is not None else x_name,
                        yaxis_title=ylabel if ylabel is not None else y_name,
                        width=width or self.width,
                        height=height or self.height,
                        font=dict(
                            family=self.font_family,
                            size=self.font_size,
                            color=self.font_color
                        ),
                        paper_bgcolor=self.paper_bgcolor,
                        plot_bgcolor=self.plot_bgcolor
                    )
                else:
                    # Horizontal boxes (swap axis labels)
                    fig.update_layout(
                        title=title,
                        xaxis_title=ylabel if ylabel is not None else y_name,
                        yaxis_title=xlabel if xlabel is not None else x_name,
                        width=width or self.width,
                        height=height or self.height,
                        font=dict(
                            family=self.font_family,
                            size=self.font_size,
                            color=self.font_color
                        ),
                        paper_bgcolor=self.paper_bgcolor,
                        plot_bgcolor=self.plot_bgcolor
                    )
        
        elif isinstance(data, np.ndarray):
            # NumPy array input
            if data.ndim == 1:
                # 1D array, create single box
                if orientation == 'vertical':
                    fig = go.Figure(
                        go.Box(
                            y=data,
                            boxpoints=points,
                            **kwargs
                        )
                    )
                else:
                    # Horizontal box
                    fig = go.Figure(
                        go.Box(
                            x=data,
                            boxpoints=points,
                            **kwargs
                        )
                    )
                
                # Set plot attributes
                fig.update_layout(
                    title=title,
                    xaxis_title=xlabel,
                    yaxis_title=ylabel if ylabel is not None else 'Value' if orientation == 'vertical' else None,
                    width=width or self.width,
                    height=height or self.height,
                    font=dict(
                        family=self.font_family,
                        size=self.font_size,
                        color=self.font_color
                    ),
                    paper_bgcolor=self.paper_bgcolor,
                    plot_bgcolor=self.plot_bgcolor
                )
                
            elif data.ndim == 2:
                # 2D array, create box for each column
                # Convert to DataFrame
                df = pd.DataFrame(data, columns=[f'Column {i}' for i in range(data.shape[1])])
                
                # Melt to long format
                df_melt = pd.melt(df, var_name='Column', value_name='Value')
                
                if orientation == 'vertical':
                    fig = px.box(
                        df_melt,
                        x='Column',
                        y='Value',
                        color=color,
                        title=title,
                        labels={
                            'Column': xlabel if xlabel is not None else 'Column',
                            'Value': ylabel if ylabel is not None else 'Value',
                            color: color_label if color_label is not None else color
                        },
                        points=points,
                        width=width or self.width,
                        height=height or self.height,
                        **kwargs
                    )
                else:
                    # Horizontal boxes (swap x and y)
                    fig = px.box(
                        df_melt,
                        x='Value',
                        y='Column',
                        color=color,
                        title=title,
                        labels={
                            'Value': ylabel if ylabel is not None else 'Value',
                            'Column': xlabel if xlabel is not None else 'Column',
                            color: color_label if color_label is not None else color
                        },
                        points=points,
                        width=width or self.width,
                        height=height or self.height,
                        **kwargs
                    )
            
            else:
                raise ValueError("NumPy array must be 1D or 2D")
        
        else:
            raise ValueError("Data must be a pandas DataFrame, dictionary, or NumPy array")
        
        # Update grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.grid_color)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.grid_color)
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            fig.show()
        
        return fig
    
    def heatmap(self, data: Union[pd.DataFrame, np.ndarray],
              title: str = 'Heatmap',
              xlabel: Optional[str] = None,
              ylabel: Optional[str] = None,
              colorscale: Optional[str] = None,
              colorbar_title: str = '',
              width: Optional[int] = None,
              height: Optional[int] = None,
              save_path: Optional[str] = None,
              show_plot: bool = True,
              **kwargs) -> go.Figure:
        """
        Create an interactive heatmap.
        
        Args:
            data: Data to plot (DataFrame or array)
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            colorscale: Colorscale name
            colorbar_title: Title for the color scale bar
            width: Plot width in pixels
            height: Plot height in pixels
            save_path: Path to save the figure (html or png/jpg/pdf)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for go.Heatmap
            
        Returns:
            Plotly Figure object
        """
        # Process data based on type
        if isinstance(data, pd.DataFrame):
            # DataFrame input
            z_data = data.values
            x_labels = data.columns.tolist()
            y_labels = data.index.tolist()
            
            # Set default axis labels if not provided
            if xlabel is None:
                xlabel = 'Columns'
            
            if ylabel is None:
                ylabel = 'Index'
            
        elif isinstance(data, np.ndarray):
            # NumPy array input
            if data.ndim == 2:
                z_data = data
                x_labels = [f'Col {i}' for i in range(data.shape[1])]
                y_labels = [f'Row {i}' for i in range(data.shape[0])]
                
                # Set default axis labels if not provided
                if xlabel is None:
                    xlabel = 'Columns'
                
                if ylabel is None:
                    ylabel = 'Rows'
                
            else:
                raise ValueError("NumPy array must be 2D for heatmap")
        
        else:
            raise ValueError("Data must be a pandas DataFrame or 2D NumPy array")
        
        # Create figure
        fig = go.Figure()
        
        # Add heatmap trace
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=x_labels,
                y=y_labels,
                colorscale=colorscale or self.colorscale,
                colorbar=dict(title=colorbar_title),
                **kwargs
            )
        )
        
        # Set plot attributes
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=width or self.width,
            height=height or self.height,
            font=dict(
                family=self.font_family,
                size=self.font_size,
                color=self.font_color
            ),
            paper_bgcolor=self.paper_bgcolor,
            plot_bgcolor=self.plot_bgcolor
        )
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            fig.show()
        
        return fig
    
    def scatter_3d(self, data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray],
                  x: Union[str, List[float], np.ndarray],
                  y: Union[str, List[float], np.ndarray],
                  z: Union[str, List[float], np.ndarray],
                  color: Optional[Union[str, List[Any]]] = None,
                  size: Optional[Union[str, List[float]]] = None,
                  text: Optional[Union[str, List[str]]] = None,
                  title: str = '3D Scatter Plot',
                  xlabel: Optional[str] = None,
                  ylabel: Optional[str] = None,
                  zlabel: Optional[str] = None,
                  color_label: Optional[str] = None,
                  size_label: Optional[str] = None,
                  colorscale: Optional[str] = None,
                  width: Optional[int] = None,
                  height: Optional[int] = None,
                  save_path: Optional[str] = None,
                  show_plot: bool = True,
                  **kwargs) -> go.Figure:
        """
        Create an interactive 3D scatter plot.
        
        Args:
            data: Data to plot (DataFrame, dictionary, or array)
            x: x-axis data or column name
            y: y-axis data or column name
            z: z-axis data or column name
            color: Variable for color mapping
            size: Variable for size mapping
            text: Variable for hover text
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            zlabel: z-axis label
            color_label: Label for color scale
            size_label: Label for size scale
            colorscale: Colorscale name
            width: Plot width in pixels
            height: Plot height in pixels
            save_path: Path to save the figure (html or png/jpg/pdf)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for plotly express
            
        Returns:
            Plotly Figure object
        """
        # Process data based on type
        if isinstance(data, pd.DataFrame):
            # DataFrame input - use plotly express for simplicity
            fig = px.scatter_3d(
                data,
                x=x,
                y=y,
                z=z,
                color=color,
                size=size,
                text=text,
                title=title,
                labels={
                    x: xlabel if xlabel is not None else x,
                    y: ylabel if ylabel is not None else y,
                    z: zlabel if zlabel is not None else z,
                    color: color_label if color_label is not None else color,
                    size: size_label if size_label is not None else size
                },
                color_continuous_scale=colorscale or self.colorscale,
                width=width or self.width,
                height=height or self.height,
                **kwargs
            )
            
        elif isinstance(data, dict):
            # Dictionary input
            if isinstance(x, str) and x in data:
                x_data = data[x]
                x_name = x
            else:
                x_data = x
                x_name = 'x'
            
            if isinstance(y, str) and y in data:
                y_data = data[y]
                y_name = y
            else:
                y_data = y
                y_name = 'y'
            
            if isinstance(z, str) and z in data:
                z_data = data[z]
                z_name = z
            else:
                z_data = z
                z_name = 'z'
            
            # Process color data
            if color is not None:
                if isinstance(color, str) and color in data:
                    color_data = data[color]
                    color_name = color
                else:
                    color_data = color
                    color_name = 'color'
            else:
                color_data = None
                color_name = None
            
            # Process size data
            if size is not None:
                if isinstance(size, str) and size in data:
                    size_data = data[size]
                    size_name = size
                else:
                    size_data = size
                    size_name = 'size'
            else:
                size_data = None
                size_name = None
            
            # Process text data
            if text is not None:
                if isinstance(text, str) and text in data:
                    text_data = data[text]
                else:
                    text_data = text
            else:
                text_data = None
            
            # Create figure
            fig = go.Figure()
            
            # Add scatter3d trace
            scatter_kwargs = {}
            if color_data is not None:
                scatter_kwargs['marker'] = dict(
                    color=color_data,
                    colorscale=colorscale or self.colorscale,
                    showscale=True,
                    colorbar=dict(title=color_label if color_label is not None else color_name)
                )
            
            if size_data is not None:
                if 'marker' not in scatter_kwargs:
                    scatter_kwargs['marker'] = dict()
                scatter_kwargs['marker']['size'] = size_data
            
            fig.add_trace(
                go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode='markers',
                    text=text_data,
                    **scatter_kwargs,
                    **kwargs
                )
            )
            
            # Set plot attributes
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title=xlabel if xlabel is not None else x_name,
                    yaxis_title=ylabel if ylabel is not None else y_name,
                    zaxis_title=zlabel if zlabel is not None else z_name
                ),
                width=width or self.width,
                height=height or self.height,
                font=dict(
                    family=self.font_family,
                    size=self.font_size,
                    color=self.font_color
                ),
                paper_bgcolor=self.paper_bgcolor
            )
        
        elif isinstance(data, np.ndarray):
            # NumPy array input
            if data.ndim == 2 and data.shape[1] >= 3:
                # Use first three columns as x, y, and z
                x_data = data[:, 0] if x is None else x
                y_data = data[:, 1] if y is None else y
                z_data = data[:, 2] if z is None else z
                
                # Process color data
                if color is not None:
                    if data.shape[1] >= 4 and color is None:
                        color_data = data[:, 3]
                        color_name = 'Column 3'
                    else:
                        color_data = color
                        color_name = 'color'
                else:
                    color_data = None
                    color_name = None
                
                # Process size data
                if size is not None:
                    if data.shape[1] >= 5 and size is None:
                        size_data = data[:, 4]
                        size_name = 'Column 4'
                    else:
                        size_data = size
                        size_name = 'size'
                else:
                    size_data = None
                    size_name = None
                
                # Create figure
                fig = go.Figure()
                
                # Add scatter3d trace
                scatter_kwargs = {}
                if color_data is not None:
                    scatter_kwargs['marker'] = dict(
                        color=color_data,
                        colorscale=colorscale or self.colorscale,
                        showscale=True,
                        colorbar=dict(title=color_label if color_label is not None else color_name)
                    )
                
                if size_data is not None:
                    if 'marker' not in scatter_kwargs:
                        scatter_kwargs['marker'] = dict()
                    scatter_kwargs['marker']['size'] = size_data
                
                fig.add_trace(
                    go.Scatter3d(
                        x=x_data,
                        y=y_data,
                        z=z_data,
                        mode='markers',
                        text=text,
                        **scatter_kwargs,
                        **kwargs
                    )
                )
                
                # Set plot attributes
                fig.update_layout(
                    title=title,
                    scene=dict(
                        xaxis_title=xlabel if xlabel is not None else 'X',
                        yaxis_title=ylabel if ylabel is not None else 'Y',
                        zaxis_title=zlabel if zlabel is not None else 'Z'
                    ),
                    width=width or self.width,
                    height=height or self.height,
                    font=dict(
                        family=self.font_family,
                        size=self.font_size,
                        color=self.font_color
                    ),
                    paper_bgcolor=self.paper_bgcolor
                )
            else:
                raise ValueError("NumPy array must have at least 3 columns for 3D scatter plot")
        
        else:
            raise ValueError("Data must be a pandas DataFrame, dictionary, or NumPy array")
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            fig.show()
        
        return fig
    
    def surface_plot(self, data: Union[pd.DataFrame, np.ndarray],
                    title: str = 'Surface Plot',
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    zlabel: Optional[str] = None,
                    colorscale: Optional[str] = None,
                    colorbar_title: str = '',
                    width: Optional[int] = None,
                    height: Optional[int] = None,
                    save_path: Optional[str] = None,
                    show_plot: bool = True,
                    **kwargs) -> go.Figure:
        """
        Create an interactive 3D surface plot.
        
        Args:
            data: Data to plot (DataFrame or array)
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            zlabel: z-axis label
            colorscale: Colorscale name
            colorbar_title: Title for the color scale bar
            width: Plot width in pixels
            height: Plot height in pixels
            save_path: Path to save the figure (html or png/jpg/pdf)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments for go.Surface
            
        Returns:
            Plotly Figure object
        """
        # Process data based on type
        if isinstance(data, pd.DataFrame):
            # DataFrame input
            z_data = data.values
            x_labels = data.columns.tolist()
            y_labels = data.index.tolist()
            
            # Set default axis labels if not provided
            if xlabel is None:
                xlabel = 'Columns'
            
            if ylabel is None:
                ylabel = 'Index'
            
            if zlabel is None:
                zlabel = 'Value'
            
        elif isinstance(data, np.ndarray):
            # NumPy array input
            if data.ndim == 2:
                z_data = data
                x_labels = np.arange(data.shape[1])
                y_labels = np.arange(data.shape[0])
                
                # Set default axis labels if not provided
                if xlabel is None:
                    xlabel = 'X'
                
                if ylabel is None:
                    ylabel = 'Y'
                
                if zlabel is None:
                    zlabel = 'Z'
                
            else:
                raise ValueError("NumPy array must be 2D for surface plot")
        
        else:
            raise ValueError("Data must be a pandas DataFrame or 2D NumPy array")
        
        # Create figure
        fig = go.Figure()
        
        # Add surface trace
        fig.add_trace(
            go.Surface(
                z=z_data,
                x=x_labels,
                y=y_labels,
                colorscale=colorscale or self.colorscale,
                colorbar=dict(title=colorbar_title),
                **kwargs
            )
        )
        
        # Set plot attributes
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                zaxis_title=zlabel
            ),
            width=width or self.width,
            height=height or self.height,
            font=dict(
                family=self.font_family,
                size=self.font_size,
                color=self.font_color
            ),
            paper_bgcolor=self.paper_bgcolor
        )
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show plot if requested
        if show_plot:
            fig.show()
        
        return fig
    
    def create_dashboard(self, title: str = 'Dashboard',
                        rows: int = 2,
                        cols: int = 2,
                        subplot_titles: Optional[List[str]] = None,
                        shared_xaxes: bool = False,
                        shared_yaxes: bool = False,
                        width: Optional[int] = None,
                        height: Optional[int] = None,
                        save_path: Optional[str] = None,
                        show_dashboard: bool = True) -> Tuple[go.Figure, List[int]]:
        """
        Create an interactive dashboard with multiple subplots.
        
        Args:
            title: Dashboard title
            rows: Number of rows
            cols: Number of columns
            subplot_titles: List of titles for each subplot
            shared_xaxes: Whether to share x-axes across subplots
            shared_yaxes: Whether to share y-axes across subplots
            width: Dashboard width in pixels
            height: Dashboard height in pixels
            save_path: Path to save the dashboard (html or png/jpg/pdf)
            show_dashboard: Whether to display the dashboard
            
        Returns:
            Tuple of (Plotly Figure object, List of subplot indices)
        """
        # Create subplot titles if not provided
        if subplot_titles is None:
            subplot_titles = [f'Plot {i+1}' for i in range(rows * cols)]
        
        # Create figure with subplots
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes
        )
        
        # Set plot attributes
        fig.update_layout(
            title=title,
            width=width or self.width * cols,
            height=height or self.height * rows,
            font=dict(
                family=self.font_family,
                size=self.font_size,
                color=self.font_color
            ),
            paper_bgcolor=self.paper_bgcolor,
            plot_bgcolor=self.plot_bgcolor
        )
        
        # Create list of subplot indices
        subplot_indices = [(i+1, j+1) for i in range(rows) for j in range(cols)]
        
        # Save figure if path is provided
        if save_path is not None:
            self._save_figure(fig, save_path)
        
        # Show dashboard if requested
        if show_dashboard:
            fig.show()
        
        return fig, subplot_indices
    
    def add_to_dashboard(self, dashboard: go.Figure,
                        plot_type: str,
                        data: Union[pd.DataFrame, Dict[str, List[float]], np.ndarray],
                        row: int,
                        col: int,
                        **kwargs) -> go.Figure:
        """
        Add a plot to a dashboard.
        
        Args:
            dashboard: Plotly Figure object created by create_dashboard
            plot_type: Type of plot to add ('scatter', 'line', 'bar', 'box', 'histogram', 'heatmap')
            data: Data to plot
            row: Row index (1-based)
            col: Column index (1-based)
            **kwargs: Additional keyword arguments for the specific plot type
            
        Returns:
            Updated Plotly Figure object
        """
        # Process data and create trace based on plot type
        if plot_type == 'scatter':
            # Extract required parameters
            x = kwargs.pop('x')
            y = kwargs.pop('y')
            
            # Process data based on type
            if isinstance(data, pd.DataFrame):
                x_data = data[x] if isinstance(x, str) else x
                y_data = data[y] if isinstance(y, str) else y
            elif isinstance(data, dict):
                x_data = data[x] if isinstance(x, str) and x in data else x
                y_data = data[y] if isinstance(y, str) and y in data else y
            elif isinstance(data, np.ndarray):
                if data.ndim == 2 and data.shape[1] >= 2:
                    x_data = data[:, 0] if x is None else x
                    y_data = data[:, 1] if y is None else y
                else:
                    raise ValueError("NumPy array must have at least 2 columns for scatter plot")
            else:
                raise ValueError("Data must be a pandas DataFrame, dictionary, or NumPy array")
            
            # Add scatter trace to dashboard
            dashboard.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    **kwargs
                ),
                row=row,
                col=col
            )
        
        elif plot_type == 'line':
            # Extract required parameters
            x = kwargs.pop('x', None)
            y = kwargs.pop('y', None)
            
            # Process data based on type
            if isinstance(data, pd.DataFrame):
                if x is None:
                    # Use index as x-axis
                    x_data = data.index
                else:
                    # Use specified column as x-axis
                    x_data = data[x] if isinstance(x, str) else x
                
                if y is None:
                    # Use first numeric column as y-axis
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        y_data = data[numeric_cols[0]]
                    else:
                        raise ValueError("No numeric columns found in DataFrame")
                else:
                    # Use specified column as y-axis
                    y_data = data[y] if isinstance(y, str) else y
            
            elif isinstance(data, dict):
                if x is None:
                    # Use range as x-axis
                    first_key = list(data.keys())[0]
                    x_data = np.arange(len(data[first_key]))
                else:
                    # Use specified key as x-axis
                    x_data = data[x] if isinstance(x, str) and x in data else x
                
                if y is None:
                    # Use first key as y-axis
                    first_key = list(data.keys())[0]
                    y_data = data[first_key]
                else:
                    # Use specified key as y-axis
                    y_data = data[y] if isinstance(y, str) and y in data else y
            
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    # 1D array, use as y-axis
                    x_data = np.arange(len(data)) if x is None else x
                    y_data = data
                
                elif data.ndim == 2:
                    # 2D array
                    if x is None:
                        # Use first column as x-axis
                        x_data = data[:, 0]
                        # Use second column as y-axis
                        y_data = data[:, 1] if data.shape[1] >= 2 else np.arange(data.shape[0])
                    else:
                        # Use provided x data
                        x_data = x
                        # Use first column as y-axis
                        y_data = data[:, 0]
                
                else:
                    raise ValueError("NumPy array must be 1D or 2D")
            
            else:
                raise ValueError("Data must be a pandas DataFrame, dictionary, or NumPy array")
            
            # Add line trace to dashboard
            dashboard.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    **kwargs
                ),
                row=row,
                col=col
            )
        
        elif plot_type == 'bar':
            # Extract required parameters
            x = kwargs.pop('x', None)
            y = kwargs.pop('y', None)
            orientation = kwargs.pop('orientation', 'vertical')
            
            # Process data based on type
            if isinstance(data, pd.DataFrame):
                if orientation == 'vertical':
                    if x is None or y is None:
                        raise ValueError("Both x and y must be specified for DataFrame bar plot")
                    
                    x_data = data[x] if isinstance(x, str) else x
                    y_data = data[y] if isinstance(y, str) else y
                else:
                    # Horizontal bars (swap x and y)
                    if x is None or y is None:
                        raise ValueError("Both x and y must be specified for DataFrame bar plot")
                    
                    y_data = data[x] if isinstance(x, str) else x
                    x_data = data[y] if isinstance(y, str) else y
            
            elif isinstance(data, dict):
                if orientation == 'vertical':
                    if x is None:
                        # Use keys as x-axis
                        x_data = list(data.keys())
                        y_data = list(data.values())
                    else:
                        # Use provided x data
                        x_data = x
                        # Use specified key as y-axis
                        y_data = data[y] if isinstance(y, str) and y in data else y
                else:
                    # Horizontal bars (swap x and y)
                    if x is None:
                        # Use keys as y-axis
                        y_data = list(data.keys())
                        x_data = list(data.values())
                    else:
                        # Use provided x data as y-axis
                        y_data = x
                        # Use specified key as x-axis
                        x_data = data[y] if isinstance(y, str) and y in data else y
            
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    # 1D array
                    if orientation == 'vertical':
                        x_data = np.arange(len(data)) if x is None else x
                        y_data = data
                    else:
                        # Horizontal bars (swap x and y)
                        y_data = np.arange(len(data)) if x is None else x
                        x_data = data
                
                elif data.ndim == 2:
                    # 2D array
                    if orientation == 'vertical':
                        if x is None:
                            # Use first column as x-axis
                            x_data = data[:, 0]
                            # Use second column as y-axis
                            y_data = data[:, 1] if data.shape[1] >= 2 else np.arange(data.shape[0])
                        else:
                            # Use provided x data
                            x_data = x
                            # Use first column as y-axis
                            y_data = data[:, 0]
                    else:
                        # Horizontal bars (swap x and y)
                        if x is None:
                            # Use first column as y-axis
                            y_data = data[:, 0]
                            # Use second column as x-axis
                            x_data = data[:, 1] if data.shape[1] >= 2 else np.arange(data.shape[0])
                        else:
                            # Use provided x data as y-axis
                            y_data = x
                            # Use first column as x-axis
                            x_data = data[:, 0]
                
                else:
                    raise ValueError("NumPy array must be 1D or 2D")
            
            else:
                raise ValueError("Data must be a pandas DataFrame, dictionary, or NumPy array")
            
            # Add bar trace to dashboard
            if orientation == 'vertical':
                dashboard.add_trace(
                    go.Bar(
                        x=x_data,
                        y=y_data,
                        **kwargs
                    ),
                    row=row,
                    col=col
                )
            else:
                # Horizontal bars
                dashboard.add_trace(
                    go.Bar(
                        x=x_data,
                        y=y_data,
                        orientation='h',
                        **kwargs
                    ),
                    row=row,
                    col=col
                )
        
        elif plot_type == 'box':
            # Extract required parameters
            x = kwargs.pop('x', None)
            y = kwargs.pop('y', None)
            orientation = kwargs.pop('orientation', 'vertical')
            
            # Process data based on type
            if isinstance(data, pd.DataFrame):
                if orientation == 'vertical':
                    if y is None:
                        raise ValueError("y must be specified for DataFrame box plot")
                    
                    x_data = data[x] if x is not None and isinstance(x, str) else None
                    y_data = data[y] if isinstance(y, str) else y
                else:
                    # Horizontal boxes (swap x and y)
                    if x is None:
                        raise ValueError("x must be specified for horizontal DataFrame box plot")
                    
                    y_data = data[x] if isinstance(x, str) else x
                    x_data = data[y] if y is not None and isinstance(y, str) else None
            
            elif isinstance(data, dict):
                if orientation == 'vertical':
                    if y is None:
                        # Use first key's values as y-axis
                        first_key = list(data.keys())[0]
                        y_data = data[first_key]
                    else:
                        # Use specified key as y-axis
                        y_data = data[y] if isinstance(y, str) and y in data else y
                    
                    x_data = data[x] if x is not None and isinstance(x, str) and x in data else None
                else:
                    # Horizontal boxes (swap x and y)
                    if x is None:
                        # Use first key's values as x-axis
                        first_key = list(data.keys())[0]
                        x_data = data[first_key]
                    else:
                        # Use specified key as x-axis
                        x_data = data[x] if isinstance(x, str) and x in data else x
                    
                    y_data = data[y] if y is not None and isinstance(y, str) and y in data else None
            
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    # 1D array
                    if orientation == 'vertical':
                        y_data = data
                        x_data = None
                    else:
                        # Horizontal boxes (swap x and y)
                        x_data = data
                        y_data = None
                
                elif data.ndim == 2:
                    # 2D array
                    if orientation == 'vertical':
                        if y is None:
                            # Use first column as y-axis
                            y_data = data[:, 0]
                        else:
                            # Use specified column as y-axis
                            y_data = y
                        
                        if x is None:
                            x_data = None
                        else:
                            # Use specified column as x-axis
                            x_data = x
                    else:
                        # Horizontal boxes (swap x and y)
                        if x is None:
                            # Use first column as x-axis
                            x_data = data[:, 0]
                        else:
                            # Use specified column as x-axis
                            x_data = x
                        
                        if y is None:
                            y_data = None
                        else:
                            # Use specified column as y-axis
                            y_data = y
                
                else:
                    raise ValueError("NumPy array must be 1D or 2D")
            
            else:
                raise ValueError("Data must be a pandas DataFrame, dictionary, or NumPy array")
            
            # Add box trace to dashboard
            if orientation == 'vertical':
                dashboard.add_trace(
                    go.Box(
                        x=x_data,
                        y=y_data,
                        **kwargs
                    ),
                    row=row,
                    col=col
                )
            else:
                # Horizontal boxes
                dashboard.add_trace(
                    go.Box(
                        x=x_data,
                        y=y_data,
                        **kwargs
                    ),
                    row=row,
                    col=col
                )
        
        elif plot_type == 'histogram':
            # Extract required parameters
            x = kwargs.pop('x', None)
            y = kwargs.pop('y', None)
            orientation = kwargs.pop('orientation', 'vertical')
            
            # Process data based on type
            if isinstance(data, pd.DataFrame):
                if orientation == 'vertical':
                    if x is None:
                        # Use first numeric column as x-axis
                        numeric_cols = data.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            x_data = data[numeric_cols[0]]
                        else:
                            raise ValueError("No numeric columns found in DataFrame")
                    else:
                        # Use specified column as x-axis
                        x_data = data[x] if isinstance(x, str) else x
                    
                    y_data = None
                else:
                    # Horizontal histogram (swap x and y)
                    if y is None:
                        # Use first numeric column as y-axis
                        numeric_cols = data.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            y_data = data[numeric_cols[0]]
                        else:
                            raise ValueError("No numeric columns found in DataFrame")
                    else:
                        # Use specified column as y-axis
                        y_data = data[y] if isinstance(y, str) else y
                    
                    x_data = None
            
            elif isinstance(data, dict):
                if orientation == 'vertical':
                    if x is None:
                        # Use first key's values as x-axis
                        first_key = list(data.keys())[0]
                        x_data = data[first_key]
                    else:
                        # Use specified key as x-axis
                        x_data = data[x] if isinstance(x, str) and x in data else x
                    
                    y_data = None
                else:
                    # Horizontal histogram (swap x and y)
                    if y is None:
                        # Use first key's values as y-axis
                        first_key = list(data.keys())[0]
                        y_data = data[first_key]
                    else:
                        # Use specified key as y-axis
                        y_data = data[y] if isinstance(y, str) and y in data else y
                    
                    x_data = None
            
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    # 1D array
                    if orientation == 'vertical':
                        x_data = data
                        y_data = None
                    else:
                        # Horizontal histogram (swap x and y)
                        y_data = data
                        x_data = None
                
                elif data.ndim == 2:
                    # 2D array
                    if orientation == 'vertical':
                        if x is None:
                            # Use first column as x-axis
                            x_data = data[:, 0]
                        else:
                            # Use specified column as x-axis
                            x_data = x
                        
                        y_data = None
                    else:
                        # Horizontal histogram (swap x and y)
                        if y is None:
                            # Use first column as y-axis
                            y_data = data[:, 0]
                        else:
                            # Use specified column as y-axis
                            y_data = y
                        
                        x_data = None
                
                else:
                    raise ValueError("NumPy array must be 1D or 2D")
            
            else:
                raise ValueError("Data must be a pandas DataFrame, dictionary, or NumPy array")
            
            # Add histogram trace to dashboard
            if orientation == 'vertical':
                dashboard.add_trace(
                    go.Histogram(
                        x=x_data,
                        **kwargs
                    ),
                    row=row,
                    col=col
                )
            else:
                # Horizontal histogram
                dashboard.add_trace(
                    go.Histogram(
                        y=y_data,
                        **kwargs
                    ),
                    row=row,
                    col=col
                )
        
        elif plot_type == 'heatmap':
            # Process data based on type
            if isinstance(data, pd.DataFrame):
                z_data = data.values
                x_labels = data.columns.tolist()
                y_labels = data.index.tolist()
            
            elif isinstance(data, np.ndarray):
                if data.ndim == 2:
                    z_data = data
                    x_labels = np.arange(data.shape[1])
                    y_labels = np.arange(data.shape[0])
                else:
                    raise ValueError("NumPy array must be 2D for heatmap")
            
            else:
                raise ValueError("Data must be a pandas DataFrame or 2D NumPy array")
            
            # Add heatmap trace to dashboard
            dashboard.add_trace(
                go.Heatmap(
                    z=z_data,
                    x=x_labels,
                    y=y_labels,
                    colorscale=kwargs.pop('colorscale', self.colorscale),
                    **kwargs
                ),
                row=row,
                col=col
            )
        
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        return dashboard
    
    def _save_figure(self, fig: go.Figure, save_path: str) -> None:
        """
        Save figure to file.
        
        Args:
            fig: Plotly Figure object
            save_path: Path to save the figure
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Determine file format based on extension
        file_ext = os.path.splitext(save_path)[1].lower()
        
        if file_ext == '.html':
            # Save as interactive HTML
            fig.write_html(save_path)
            logger.info(f"Saved interactive figure to {save_path}")
        
        elif file_ext in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']:
            # Save as static image
            fig.write_image(save_path)
            logger.info(f"Saved static figure to {save_path}")
        
        elif file_ext == '.json':
            # Save as JSON
            fig.write_json(save_path)
            logger.info(f"Saved figure as JSON to {save_path}")
        
        else:
            # Default to HTML
            html_path = f"{os.path.splitext(save_path)[0]}.html"
            fig.write_html(html_path)
            logger.info(f"Saved interactive figure to {html_path} (unknown extension {file_ext})")
