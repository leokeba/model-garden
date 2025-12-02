"""Generic hyperparameter exploration module for vision fine-tuning.

This module provides a flexible framework for systematically testing different
training parameters and analyzing their impact on performance metrics.
"""

import json
import itertools
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


class HyperparameterSpace:
    """Define the search space for hyperparameters."""
    
    def __init__(self):
        self.params: Dict[str, List[Any]] = {}
        self.fixed_params: Dict[str, Any] = {}
    
    def add_parameter(self, name: str, values: List[Any]) -> 'HyperparameterSpace':
        """Add a parameter to explore with multiple values."""
        self.params[name] = values
        return self
    
    def add_fixed_parameter(self, name: str, value: Any) -> 'HyperparameterSpace':
        """Add a fixed parameter that won't be varied."""
        self.fixed_params[name] = value
        return self
    
    def grid_search(self) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters (grid search)."""
        if not self.params:
            return [self.fixed_params.copy()]
        
        param_names = list(self.params.keys())
        param_values = [self.params[name] for name in param_names]
        
        configurations = []
        for combination in itertools.product(*param_values):
            config = self.fixed_params.copy()
            config.update(dict(zip(param_names, combination)))
            configurations.append(config)
        
        return configurations
    
    def random_search(self, n_samples: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate random combinations of parameters."""
        import random
        if seed is not None:
            random.seed(seed)
        
        configurations = []
        for _ in range(n_samples):
            config = self.fixed_params.copy()
            for name, values in self.params.items():
                config[name] = random.choice(values)
            configurations.append(config)
        
        return configurations


class ExperimentResult:
    """Store results from a single experiment run."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.metrics = metrics
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'config': self.config,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentResult':
        """Create from dictionary."""
        result = cls(data['config'], data['metrics'], data.get('metadata', {}))
        result.timestamp = data.get('timestamp', datetime.now().isoformat())
        return result


class HyperparameterExplorer:
    """Main class for running hyperparameter exploration experiments."""
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: Union[str, Path] = "experiments"
    ):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[ExperimentResult] = []
        self.results_file = self.output_dir / f"{experiment_name}_results.json"
        
        # Load existing results if available
        if self.results_file.exists():
            self._load_results()
    
    def _load_results(self):
        """Load existing results from file."""
        try:
            with open(self.results_file) as f:
                data = json.load(f)
                self.results = [ExperimentResult.from_dict(r) for r in data]
            console.print(f"[green]Loaded {len(self.results)} existing results[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load existing results: {e}[/yellow]")
    
    def _save_results(self):
        """Save results to file."""
        with open(self.results_file, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
    
    def run_exploration(
        self,
        space: HyperparameterSpace,
        train_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        search_strategy: str = "grid",
        n_samples: Optional[int] = None,
        resume: bool = True,
        save_frequency: int = 1
    ) -> List[ExperimentResult]:
        """
        Run hyperparameter exploration.
        
        Args:
            space: HyperparameterSpace defining the search space
            train_fn: Function that takes config dict and returns metrics dict
            search_strategy: "grid" or "random"
            n_samples: Number of samples for random search
            resume: Whether to skip configurations already tested
            save_frequency: Save results every N experiments
        
        Returns:
            List of ExperimentResult objects
        """
        # Generate configurations
        if search_strategy == "grid":
            configurations = space.grid_search()
        elif search_strategy == "random":
            if n_samples is None:
                raise ValueError("n_samples required for random search")
            configurations = space.random_search(n_samples)
        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}")
        
        console.print(f"\n[bold cyan]Starting {self.experiment_name}[/bold cyan]")
        console.print(f"Total configurations to test: {len(configurations)}")
        
        # Filter out already tested configurations if resuming
        if resume:
            tested_configs = {self._config_hash(r.config) for r in self.results}
            configurations = [c for c in configurations if self._config_hash(c) not in tested_configs]
            console.print(f"Skipping {len(tested_configs)} already tested configurations")
            console.print(f"Remaining: {len(configurations)}")
        
        if not configurations:
            console.print("[yellow]No new configurations to test![/yellow]")
            return self.results
        
        # Run experiments
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"Running experiments...",
                total=len(configurations)
            )
            
            for idx, config in enumerate(configurations, 1):
                try:
                    # Display current configuration
                    config_str = ", ".join(f"{k}={v}" for k, v in config.items() 
                                          if k not in space.fixed_params)
                    progress.update(task, description=f"[cyan]Run {idx}/{len(configurations)}: {config_str}")
                    
                    # Run training
                    start_time = time.time()
                    metrics = train_fn(config)
                    elapsed_time = time.time() - start_time
                    
                    # Store result
                    result = ExperimentResult(
                        config=config,
                        metrics=metrics,
                        metadata={'elapsed_time': elapsed_time}
                    )
                    self.results.append(result)
                    
                    # Save periodically
                    if idx % save_frequency == 0:
                        self._save_results()
                    
                    progress.advance(task)
                    
                except Exception as e:
                    console.print(f"[red]Error in run {idx}: {e}[/red]")
                    console.print_exception()
                    continue
        
        # Final save
        self._save_results()
        console.print(f"\n[green]✓ Completed {len(configurations)} experiments[/green]")
        console.print(f"Results saved to: {self.results_file}")
        
        return self.results
    
    def _config_hash(self, config: Dict[str, Any]) -> str:
        """Create a hash string for a configuration."""
        items = sorted(config.items())
        return str(items)
    
    def get_best_results(
        self,
        metric: str,
        n: int = 10,
        minimize: bool = False
    ) -> List[ExperimentResult]:
        """Get top N results by a specific metric."""
        sorted_results = sorted(
            self.results,
            key=lambda r: r.metrics.get(metric, float('inf') if minimize else float('-inf')),
            reverse=not minimize
        )
        return sorted_results[:n]
    
    def print_summary(self, primary_metric: str, minimize: bool = False):
        """Print a summary table of results."""
        if not self.results:
            console.print("[yellow]No results to display[/yellow]")
            return
        
        # Get all unique parameter names
        param_names = set()
        for result in self.results:
            param_names.update(result.config.keys())
        param_names = sorted(param_names)
        
        # Get all unique metric names
        metric_names = set()
        for result in self.results:
            metric_names.update(result.metrics.keys())
        metric_names = sorted(metric_names)
        
        # Create summary table
        table = Table(title=f"{self.experiment_name} - Summary", show_lines=True)
        
        # Add parameter columns
        for param in param_names:
            table.add_column(param, style="cyan")
        
        # Add metric columns
        for metric in metric_names:
            style = "green" if metric == primary_metric else "white"
            table.add_column(metric, style=style)
        
        # Add time column
        table.add_column("Time (min)", style="dim")
        
        # Add top 10 results
        top_results = self.get_best_results(primary_metric, n=10, minimize=minimize)
        
        for result in top_results:
            row = []
            # Add parameter values
            for param in param_names:
                value = result.config.get(param, "N/A")
                if isinstance(value, float):
                    row.append(f"{value:.0e}")
                else:
                    row.append(str(value))
            
            # Add metric values
            for metric in metric_names:
                value = result.metrics.get(metric, float('nan'))
                if isinstance(value, float):
                    row.append(f"{value:.2f}")
                else:
                    row.append(str(value))
            
            # Add time
            elapsed = result.metadata.get('elapsed_time', 0)
            row.append(f"{elapsed/60:.1f}")
            
            table.add_row(*row)
        
        console.print("\n")
        console.print(table)
        
        # Print statistics
        console.print(f"\n[bold]Statistics:[/bold]")
        console.print(f"Total runs: {len(self.results)}")
        
        if primary_metric in metric_names:
            values = [r.metrics.get(primary_metric) for r in self.results 
                     if primary_metric in r.metrics]
            if values:
                import numpy as np
                console.print(f"{primary_metric}: {np.mean(values):.2f} ± {np.std(values):.2f}")
                console.print(f"Best {primary_metric}: {min(values) if minimize else max(values):.2f}")
    
    def export_for_plotting(self) -> Dict[str, Any]:
        """Export results in a format suitable for plotting."""
        return {
            'experiment_name': self.experiment_name,
            'results': [r.to_dict() for r in self.results],
            'n_results': len(self.results)
        }
