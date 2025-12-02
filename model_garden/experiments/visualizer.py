"""Flexible visualization module for hyperparameter exploration results.

This module provides tools for creating various plots to analyze the impact
of different parameters on performance metrics.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure


class ExplorationVisualizer:
    """Create visualizations for hyperparameter exploration results."""
    
    def __init__(
        self,
        results_file: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize visualizer.
        
        Args:
            results_file: Path to JSON file with experiment results
            output_dir: Directory to save plots (defaults to same dir as results)
        """
        self.results_file = Path(results_file)
        
        if output_dir is None:
            self.output_dir = self.results_file.parent / "plots"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        with open(self.results_file) as f:
            data = json.load(f)
            if isinstance(data, dict) and 'results' in data:
                self.results = data['results']
                self.experiment_name = data.get('experiment_name', 'Experiment')
            else:
                self.results = data
                self.experiment_name = self.results_file.stem
        
        # Extract all parameters and metrics
        self.parameters = self._extract_unique_keys('config')
        self.metrics = self._extract_unique_keys('metrics')
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def _extract_unique_keys(self, field: str) -> List[str]:
        """Extract all unique keys from a field across all results."""
        keys = set()
        for result in self.results:
            if field in result:
                keys.update(result[field].keys())
        return sorted(keys)
    
    def _extract_data(
        self,
        parameter: Optional[str] = None,
        metric: Optional[str] = None
    ) -> Tuple[List[Any], List[float]]:
        """Extract parameter values and corresponding metric values."""
        param_values = []
        metric_values = []
        
        for result in self.results:
            if parameter and parameter in result['config']:
                param_val = result['config'][parameter]
            else:
                param_val = None
            
            if metric and metric in result['metrics']:
                metric_val = result['metrics'][metric]
            else:
                metric_val = None
            
            if metric_val is not None:
                param_values.append(param_val)
                metric_values.append(metric_val)
        
        return param_values, metric_values
    
    def plot_parameter_impact(
        self,
        parameter: str,
        metric: str,
        minimize: bool = False,
        title: Optional[str] = None,
        save: bool = True
    ) -> Figure:
        """
        Plot the impact of a single parameter on a metric.
        
        Shows mean ± std for each parameter value.
        """
        param_values, metric_values = self._extract_data(parameter, metric)
        
        # Group by parameter value
        grouped = {}
        for pv, mv in zip(param_values, metric_values):
            if pv not in grouped:
                grouped[pv] = []
            grouped[pv].append(mv)
        
        # Calculate statistics
        sorted_params = sorted(grouped.keys())
        means = [np.mean(grouped[p]) for p in sorted_params]
        stds = [np.std(grouped[p]) for p in sorted_params]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = range(len(sorted_params))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                     color='coral' if minimize else 'skyblue',
                     edgecolor='black')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([self._format_value(p) for p in sorted_params])
        ax.set_xlabel(parameter, fontweight='bold', fontsize=12)
        ax.set_ylabel(f'{metric} {"(lower is better)" if minimize else ""}', 
                     fontweight='bold', fontsize=12)
        
        if title is None:
            title = f'Impact of {parameter} on {metric}'
        ax.set_title(title, fontweight='bold', fontsize=14)
        
        ax.grid(axis='y', alpha=0.3)
        
        if minimize:
            ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save:
            filename = f"{parameter}_vs_{metric}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        return fig
    
    def plot_heatmap(
        self,
        param1: str,
        param2: str,
        metric: str,
        minimize: bool = False,
        title: Optional[str] = None,
        save: bool = True
    ) -> Figure:
        """
        Plot a 2D heatmap showing the impact of two parameters on a metric.
        """
        # Extract data
        data_dict = {}
        for result in self.results:
            p1 = result['config'].get(param1)
            p2 = result['config'].get(param2)
            m = result['metrics'].get(metric)
            
            if p1 is not None and p2 is not None and m is not None:
                key = (p1, p2)
                if key not in data_dict:
                    data_dict[key] = []
                data_dict[key].append(m)
        
        # Create grid
        p1_values = sorted(set(k[0] for k in data_dict.keys()))
        p2_values = sorted(set(k[1] for k in data_dict.keys()))
        
        grid = np.zeros((len(p2_values), len(p1_values)))
        for i, p2_val in enumerate(p2_values):
            for j, p1_val in enumerate(p1_values):
                key = (p1_val, p2_val)
                if key in data_dict:
                    grid[i, j] = np.mean(data_dict[key])
                else:
                    grid[i, j] = np.nan
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cmap = 'RdYlGn_r' if minimize else 'RdYlGn'
        im = ax.imshow(grid, cmap=cmap, aspect='auto')
        
        ax.set_xticks(range(len(p1_values)))
        ax.set_yticks(range(len(p2_values)))
        ax.set_xticklabels([self._format_value(v) for v in p1_values])
        ax.set_yticklabels([self._format_value(v) for v in p2_values])
        
        ax.set_xlabel(param1, fontweight='bold', fontsize=12)
        ax.set_ylabel(param2, fontweight='bold', fontsize=12)
        
        if title is None:
            title = f'{metric}: {param1} × {param2}'
        ax.set_title(title, fontweight='bold', fontsize=14)
        
        # Add values to cells
        for i in range(len(p2_values)):
            for j in range(len(p1_values)):
                if not np.isnan(grid[i, j]):
                    text = ax.text(j, i, f'{grid[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=10)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{metric} {"(lower is better)" if minimize else ""}', 
                      fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filename = f"heatmap_{param1}_vs_{param2}_{metric}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        return fig
    
    def plot_scatter(
        self,
        x_metric: str,
        y_metric: str,
        color_by: Optional[str] = None,
        size_by: Optional[str] = None,
        title: Optional[str] = None,
        save: bool = True
    ) -> Figure:
        """
        Create a scatter plot comparing two metrics, optionally colored/sized by parameter.
        """
        x_values = [r['metrics'].get(x_metric) for r in self.results]
        y_values = [r['metrics'].get(y_metric) for r in self.results]
        
        # Filter out None values
        valid_indices = [i for i, (x, y) in enumerate(zip(x_values, y_values)) 
                        if x is not None and y is not None]
        x_values = [x_values[i] for i in valid_indices]
        y_values = [y_values[i] for i in valid_indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare color and size
        if color_by:
            c_values = [self.results[i]['config'].get(color_by) or 
                       self.results[i]['metrics'].get(color_by) 
                       for i in valid_indices]
        else:
            c_values = 'skyblue'
        
        if size_by:
            s_values = [self.results[i]['config'].get(size_by) or 
                       self.results[i]['metrics'].get(size_by) 
                       for i in valid_indices]
            s_values = np.array(s_values) * 10  # Scale for visibility
        else:
            s_values = 100
        
        scatter = ax.scatter(x_values, y_values, c=c_values, s=s_values, 
                           alpha=0.6, edgecolors='black', cmap='viridis')
        
        ax.set_xlabel(x_metric, fontweight='bold', fontsize=12)
        ax.set_ylabel(y_metric, fontweight='bold', fontsize=12)
        
        if title is None:
            title = f'{y_metric} vs {x_metric}'
        ax.set_title(title, fontweight='bold', fontsize=14)
        
        ax.grid(alpha=0.3)
        
        if color_by and isinstance(c_values, list):
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_by, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filename = f"scatter_{x_metric}_vs_{y_metric}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        return fig
    
    def plot_distribution(
        self,
        metric: str,
        bins: int = 20,
        show_baseline: Optional[float] = None,
        title: Optional[str] = None,
        save: bool = True
    ) -> Figure:
        """Plot the distribution of a metric across all runs."""
        _, metric_values = self._extract_data(metric=metric)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(metric_values, bins=bins, alpha=0.7, color='skyblue', 
               edgecolor='black')
        
        mean_val = np.mean(metric_values)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.2f}')
        
        if show_baseline is not None:
            ax.axvline(show_baseline, color='blue', linestyle=':', linewidth=2,
                      label=f'Baseline: {show_baseline:.2f}')
        
        ax.set_xlabel(metric, fontweight='bold', fontsize=12)
        ax.set_ylabel('Frequency', fontweight='bold', fontsize=12)
        
        if title is None:
            title = f'Distribution of {metric}'
        ax.set_title(title, fontweight='bold', fontsize=14)
        
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f"distribution_{metric}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        return fig
    
    def plot_top_configurations(
        self,
        metric: str,
        n: int = 10,
        minimize: bool = False,
        display_params: Optional[List[str]] = None,
        title: Optional[str] = None,
        save: bool = True
    ) -> Figure:
        """Plot the top N configurations by a metric."""
        # Sort results
        sorted_results = sorted(
            self.results,
            key=lambda r: r['metrics'].get(metric, float('inf') if minimize else float('-inf')),
            reverse=not minimize
        )[:n]
        
        # Create labels
        if display_params is None:
            display_params = self.parameters
        
        labels = []
        values = []
        for result in sorted_results:
            label_parts = [f"{p}:{self._format_value(result['config'].get(p))}" 
                          for p in display_params if p in result['config']]
            # Split into 2 lines: put half parameters on each line
            mid = (len(label_parts) + 1) // 2
            line1 = ' '.join(label_parts[:mid])
            line2 = ' '.join(label_parts[mid:]) if mid < len(label_parts) else ''
            label = f"{line1}\n{line2}" if line2 else line1
            labels.append(label)
            values.append(result['metrics'].get(metric, 0))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, max(6, n * 0.5)))
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, n)) if minimize else \
                 plt.cm.RdYlGn(np.linspace(0.3, 0.8, n))
        
        y_pos = range(len(labels))
        ax.barh(y_pos, values, color=colors, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(f'{metric} {"(lower is better)" if minimize else ""}', 
                     fontweight='bold', fontsize=12)
        
        if title is None:
            title = f'Top {n} Configurations by {metric}'
        ax.set_title(title, fontweight='bold', fontsize=14)
        
        # Add value labels
        for i, v in enumerate(values):
            x_pos = v * 0.02 if minimize else v + (max(values) * 0.02)
            align = 'left' if minimize else 'left'
            ax.text(x_pos, i, f'{v:.2f}', va='center', fontweight='bold', ha=align)
        
        ax.grid(axis='x', alpha=0.3)
        
        if minimize:
            ax.invert_xaxis()
        
        plt.tight_layout()
        
        if save:
            filename = f"top_{n}_{metric}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        return fig
    
    def create_full_report(
        self,
        primary_metric: str,
        minimize: bool = False,
        secondary_metrics: Optional[List[str]] = None
    ):
        """
        Create a comprehensive report with multiple visualizations.
        """
        print(f"\n{'='*60}")
        print(f"Creating visualization report for: {self.experiment_name}")
        print(f"{'='*60}\n")
        
        # 1. Distribution of primary metric
        self.plot_distribution(primary_metric, 
                             title=f'{primary_metric} Distribution (Primary Metric)')
        
        # 2. Impact of each parameter on primary metric
        for param in self.parameters:
            self.plot_parameter_impact(param, primary_metric, minimize=minimize)
        
        # 3. Heatmaps for parameter pairs
        if len(self.parameters) >= 2:
            param_pairs = []
            params_list = list(self.parameters)
            for i in range(len(params_list)):
                for j in range(i + 1, len(params_list)):
                    param_pairs.append((params_list[i], params_list[j]))
            
            for p1, p2 in param_pairs[:6]:  # Limit to 6 heatmaps
                self.plot_heatmap(p1, p2, primary_metric, minimize=minimize)
        
        # 4. Top configurations
        self.plot_top_configurations(primary_metric, n=10, minimize=minimize)
        
        # 5. Secondary metrics
        if secondary_metrics:
            for metric in secondary_metrics:
                self.plot_distribution(metric, title=f'{metric} Distribution')
                
                # Scatter: primary vs secondary
                self.plot_scatter(primary_metric, metric,
                                title=f'{metric} vs {primary_metric}')
        
        print(f"\n{'='*60}")
        print(f"Report complete! Plots saved to: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if isinstance(value, float):
            if value < 0.001:
                return f'{value:.0e}'
            elif value < 1:
                return f'{value:.3f}'
            else:
                return f'{value:.2f}'
        return str(value)
