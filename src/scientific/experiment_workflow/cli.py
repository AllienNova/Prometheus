"""
Experiment workflow CLI for the Prometheus AI Automation Platform.

This module provides a command-line interface for managing scientific experiments
using the experiment workflow system.
"""

import os
import sys
import argparse
import logging
import json
import datetime
import tabulate
from typing import List, Dict, Any, Optional

from .workflow import (
    Experiment,
    ExperimentStatus,
    ExperimentWorkflowManager,
    get_experiment_manager
)
from .templates import (
    get_available_templates,
    create_experiment_from_template
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_parser() -> argparse.ArgumentParser:
    """
    Set up the command-line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Prometheus AI Experiment Workflow CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  experiment-cli list
  
  # Create a new experiment from template
  experiment-cli create --template data_analysis --name "My Data Analysis" --author "John Doe"
  
  # Execute an experiment
  experiment-cli execute --id 12345678-1234-5678-1234-567812345678
  
  # View experiment details
  experiment-cli view --id 12345678-1234-5678-1234-567812345678
  
  # Export experiment results
  experiment-cli export --id 12345678-1234-5678-1234-567812345678 --format html
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List experiments')
    list_parser.add_argument('--status', help='Filter by status')
    list_parser.add_argument('--author', help='Filter by author')
    list_parser.add_argument('--tag', help='Filter by tag')
    list_parser.add_argument('--format', choices=['table', 'json'], default='table',
                            help='Output format (default: table)')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new experiment')
    create_parser.add_argument('--template', required=True, help='Template to use')
    create_parser.add_argument('--name', required=True, help='Experiment name')
    create_parser.add_argument('--description', default='', help='Experiment description')
    create_parser.add_argument('--author', default='', help='Experiment author')
    create_parser.add_argument('--params', help='JSON string of additional parameters')
    
    # Templates command
    templates_parser = subparsers.add_parser('templates', help='List available templates')
    
    # Execute command
    execute_parser = subparsers.add_parser('execute', help='Execute an experiment')
    execute_parser.add_argument('--id', required=True, help='Experiment ID')
    execute_parser.add_argument('--sync', action='store_true', help='Execute synchronously')
    
    # Abort command
    abort_parser = subparsers.add_parser('abort', help='Abort a running experiment')
    abort_parser.add_argument('--id', required=True, help='Experiment ID')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View experiment details')
    view_parser.add_argument('--id', required=True, help='Experiment ID')
    view_parser.add_argument('--format', choices=['table', 'json'], default='table',
                           help='Output format (default: table)')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export experiment results')
    export_parser.add_argument('--id', required=True, help='Experiment ID')
    export_parser.add_argument('--format', choices=['json', 'csv', 'html'], default='json',
                             help='Export format (default: json)')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple experiments')
    compare_parser.add_argument('--ids', required=True, help='Comma-separated list of experiment IDs')
    compare_parser.add_argument('--format', choices=['table', 'json'], default='table',
                              help='Output format (default: table)')
    
    # Clone command
    clone_parser = subparsers.add_parser('clone', help='Clone an experiment')
    clone_parser.add_argument('--id', required=True, help='Experiment ID')
    clone_parser.add_argument('--name', help='New experiment name')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete an experiment')
    delete_parser.add_argument('--id', required=True, help='Experiment ID')
    delete_parser.add_argument('--force', action='store_true', help='Force deletion without confirmation')
    
    return parser

def format_experiment_list(experiments: List[Experiment], format_type: str) -> str:
    """
    Format a list of experiments for display.
    
    Args:
        experiments: List of experiments
        format_type: Output format ('table' or 'json')
        
    Returns:
        Formatted string
    """
    if format_type == 'json':
        # Convert to JSON
        data = [
            {
                'id': exp.id,
                'name': exp.name,
                'author': exp.author,
                'status': exp.status.value,
                'created_at': exp.created_at.isoformat() if exp.created_at else None,
                'tags': exp.tags
            }
            for exp in experiments
        ]
        return json.dumps(data, indent=2)
    else:
        # Format as table
        headers = ['ID', 'Name', 'Author', 'Status', 'Created', 'Tags']
        rows = [
            [
                exp.id,
                exp.name,
                exp.author,
                exp.status.value,
                exp.created_at.strftime('%Y-%m-%d %H:%M') if exp.created_at else '',
                ', '.join(exp.tags)
            ]
            for exp in experiments
        ]
        return tabulate.tabulate(rows, headers=headers, tablefmt='grid')

def format_experiment_details(experiment: Experiment, format_type: str) -> str:
    """
    Format experiment details for display.
    
    Args:
        experiment: Experiment object
        format_type: Output format ('table' or 'json')
        
    Returns:
        Formatted string
    """
    if format_type == 'json':
        # Convert to JSON
        return json.dumps(experiment.to_dict(), indent=2)
    else:
        # Format as tables
        result = []
        
        # Basic information
        result.append("Experiment Information:")
        basic_info = [
            ['ID', experiment.id],
            ['Name', experiment.name],
            ['Description', experiment.description],
            ['Author', experiment.author],
            ['Status', experiment.status.value],
            ['Created', experiment.created_at.strftime('%Y-%m-%d %H:%M') if experiment.created_at else ''],
            ['Started', experiment.started_at.strftime('%Y-%m-%d %H:%M') if experiment.started_at else ''],
            ['Completed', experiment.completed_at.strftime('%Y-%m-%d %H:%M') if experiment.completed_at else ''],
            ['Tags', ', '.join(experiment.tags)]
        ]
        result.append(tabulate.tabulate(basic_info, tablefmt='grid'))
        result.append("")
        
        # Parameters
        result.append("Parameters:")
        param_rows = [[k, str(v)] for k, v in experiment.parameters.items()]
        result.append(tabulate.tabulate(param_rows, headers=['Parameter', 'Value'], tablefmt='grid'))
        result.append("")
        
        # Metrics
        if experiment.metrics:
            result.append("Metrics:")
            metric_rows = [[k, str(v)] for k, v in experiment.metrics.items()]
            result.append(tabulate.tabulate(metric_rows, headers=['Metric', 'Value'], tablefmt='grid'))
            result.append("")
        
        # Steps
        result.append("Steps:")
        step_rows = [
            [
                step.name,
                step.status,
                step.started_at.strftime('%Y-%m-%d %H:%M') if step.started_at else '',
                step.completed_at.strftime('%Y-%m-%d %H:%M') if step.completed_at else '',
                step.error if step.error else ''
            ]
            for step in experiment.steps
        ]
        result.append(tabulate.tabulate(
            step_rows,
            headers=['Step', 'Status', 'Started', 'Completed', 'Error'],
            tablefmt='grid'
        ))
        result.append("")
        
        # Artifacts
        if experiment.artifacts:
            result.append("Artifacts:")
            artifact_rows = [
                [
                    artifact.name,
                    artifact.type,
                    artifact.description,
                    artifact.created_at.strftime('%Y-%m-%d %H:%M') if artifact.created_at else ''
                ]
                for artifact in experiment.artifacts
            ]
            result.append(tabulate.tabulate(
                artifact_rows,
                headers=['Name', 'Type', 'Description', 'Created'],
                tablefmt='grid'
            ))
            result.append("")
        
        # Logs (last 10)
        if experiment.logs:
            result.append("Logs (last 10):")
            for log in experiment.logs[-10:]:
                result.append(f"  {log}")
        
        return "\n".join(result)

def format_comparison_results(comparison: Dict[str, Any], format_type: str) -> str:
    """
    Format experiment comparison results for display.
    
    Args:
        comparison: Comparison results
        format_type: Output format ('table' or 'json')
        
    Returns:
        Formatted string
    """
    if format_type == 'json':
        # Convert to JSON
        return json.dumps(comparison, indent=2)
    else:
        # Format as tables
        result = []
        
        # Experiments
        result.append("Experiments:")
        exp_rows = [
            [
                exp['id'],
                exp['name'],
                exp['status'],
                exp['created_at'],
                exp['completed_at'] or ''
            ]
            for exp in comparison['experiments']
        ]
        result.append(tabulate.tabulate(
            exp_rows,
            headers=['ID', 'Name', 'Status', 'Created', 'Completed'],
            tablefmt='grid'
        ))
        result.append("")
        
        # Metrics
        if comparison['metrics']:
            result.append("Metrics:")
            for metric_name, metric_values in comparison['metrics'].items():
                result.append(f"  {metric_name}:")
                metric_rows = [
                    [exp_id, exp_name, value]
                    for exp_id, exp_name, value in metric_values
                ]
                result.append(tabulate.tabulate(
                    metric_rows,
                    headers=['Experiment ID', 'Experiment Name', 'Value'],
                    tablefmt='simple'
                ))
                result.append("")
        
        # Parameters
        if comparison['parameters']:
            result.append("Parameters:")
            for param_name, param_values in comparison['parameters'].items():
                result.append(f"  {param_name}:")
                param_rows = [
                    [exp_id, exp_name, str(value)]
                    for exp_id, exp_name, value in param_values
                ]
                result.append(tabulate.tabulate(
                    param_rows,
                    headers=['Experiment ID', 'Experiment Name', 'Value'],
                    tablefmt='simple'
                ))
                result.append("")
        
        return "\n".join(result)

def handle_list_command(args) -> int:
    """
    Handle the 'list' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    manager = get_experiment_manager()
    
    # Convert status string to enum if provided
    status = None
    if args.status:
        try:
            status = ExperimentStatus(args.status)
        except ValueError:
            print(f"Error: Invalid status '{args.status}'")
            return 1
    
    # List experiments
    experiments = manager.list_experiments(
        status=status,
        author=args.author,
        tag=args.tag
    )
    
    if not experiments:
        print("No experiments found.")
        return 0
    
    # Format and print results
    print(format_experiment_list(experiments, args.format))
    return 0

def handle_create_command(args) -> int:
    """
    Handle the 'create' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    # Check if template exists
    templates = get_available_templates()
    if args.template not in templates:
        print(f"Error: Unknown template '{args.template}'")
        print("Available templates:")
        for name, desc in templates.items():
            print(f"  {name}: {desc}")
        return 1
    
    # Parse additional parameters
    kwargs = {}
    if args.params:
        try:
            kwargs = json.loads(args.params)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in --params")
            return 1
    
    # Create experiment
    experiment = create_experiment_from_template(
        template_name=args.template,
        experiment_name=args.name,
        description=args.description,
        author=args.author,
        **kwargs
    )
    
    if experiment is None:
        print(f"Error: Failed to create experiment")
        return 1
    
    # Save experiment
    manager = get_experiment_manager()
    manager.experiments[experiment.id] = experiment
    filepath = manager.save_experiment(experiment.id)
    
    print(f"Created experiment '{experiment.name}' with ID {experiment.id}")
    print(f"Saved to {filepath}")
    return 0

def handle_templates_command(args) -> int:
    """
    Handle the 'templates' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    templates = get_available_templates()
    
    print("Available templates:")
    for name, desc in templates.items():
        print(f"  {name}: {desc}")
    
    return 0

def handle_execute_command(args) -> int:
    """
    Handle the 'execute' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    manager = get_experiment_manager()
    
    # Get experiment
    experiment = manager.get_experiment(args.id)
    if experiment is None:
        print(f"Error: Experiment with ID {args.id} not found")
        return 1
    
    # Execute experiment
    print(f"Executing experiment '{experiment.name}'...")
    result = manager.execute_experiment(args.id, async_mode=not args.sync)
    
    if args.sync:
        if result:
            print(f"Experiment completed successfully")
            return 0
        else:
            print(f"Experiment failed")
            return 1
    else:
        print(f"Experiment execution started")
        return 0

def handle_abort_command(args) -> int:
    """
    Handle the 'abort' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    manager = get_experiment_manager()
    
    # Get experiment
    experiment = manager.get_experiment(args.id)
    if experiment is None:
        print(f"Error: Experiment with ID {args.id} not found")
        return 1
    
    # Abort experiment
    result = manager.abort_experiment(args.id)
    
    if result:
        print(f"Aborted experiment '{experiment.name}'")
        return 0
    else:
        print(f"Failed to abort experiment (not running)")
        return 1

def handle_view_command(args) -> int:
    """
    Handle the 'view' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    manager = get_experiment_manager()
    
    # Get experiment
    experiment = manager.get_experiment(args.id)
    if experiment is None:
        print(f"Error: Experiment with ID {args.id} not found")
        return 1
    
    # Format and print details
    print(format_experiment_details(experiment, args.format))
    return 0

def handle_export_command(args) -> int:
    """
    Handle the 'export' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    manager = get_experiment_manager()
    
    # Get experiment
    experiment = manager.get_experiment(args.id)
    if experiment is None:
        print(f"Error: Experiment with ID {args.id} not found")
        return 1
    
    # Export results
    filepath = manager.export_experiment_results(args.id, format=args.format)
    
    if filepath:
        print(f"Exported experiment results to {filepath}")
        return 0
    else:
        print(f"Failed to export experiment results")
        return 1

def handle_compare_command(args) -> int:
    """
    Handle the 'compare' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    manager = get_experiment_manager()
    
    # Parse experiment IDs
    experiment_ids = args.ids.split(',')
    
    # Compare experiments
    comparison = manager.compare_experiments(experiment_ids)
    
    if not comparison:
        print(f"Error: No valid experiments to compare")
        return 1
    
    # Format and print comparison
    print(format_comparison_results(comparison, args.format))
    return 0

def handle_clone_command(args) -> int:
    """
    Handle the 'clone' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    manager = get_experiment_manager()
    
    # Get experiment
    experiment = manager.get_experiment(args.id)
    if experiment is None:
        print(f"Error: Experiment with ID {args.id} not found")
        return 1
    
    # Clone experiment
    clone = manager.clone_experiment(args.id, args.name)
    
    if clone:
        print(f"Cloned experiment '{experiment.name}' to '{clone.name}' with ID {clone.id}")
        return 0
    else:
        print(f"Failed to clone experiment")
        return 1

def handle_delete_command(args) -> int:
    """
    Handle the 'delete' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    manager = get_experiment_manager()
    
    # Get experiment
    experiment = manager.get_experiment(args.id)
    if experiment is None:
        print(f"Error: Experiment with ID {args.id} not found")
        return 1
    
    # Confirm deletion
    if not args.force:
        confirm = input(f"Are you sure you want to delete experiment '{experiment.name}'? (y/N) ")
        if confirm.lower() != 'y':
            print("Deletion cancelled")
            return 0
    
    # Delete experiment
    result = manager.delete_experiment(args.id)
    
    if result:
        print(f"Deleted experiment '{experiment.name}'")
        return 0
    else:
        print(f"Failed to delete experiment")
        return 1

def main() -> int:
    """
    Main entry point for the CLI.
    
    Returns:
        Exit code
    """
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Initialize experiment manager
    manager = get_experiment_manager()
    manager.load_all_experiments()
    
    # Handle command
    if args.command == 'list':
        return handle_list_command(args)
    elif args.command == 'create':
        return handle_create_command(args)
    elif args.command == 'templates':
        return handle_templates_command(args)
    elif args.command == 'execute':
        return handle_execute_command(args)
    elif args.command == 'abort':
        return handle_abort_command(args)
    elif args.command == 'view':
        return handle_view_command(args)
    elif args.command == 'export':
        return handle_export_command(args)
    elif args.command == 'compare':
        return handle_compare_command(args)
    elif args.command == 'clone':
        return handle_clone_command(args)
    elif args.command == 'delete':
        return handle_delete_command(args)
    else:
        print(f"Error: Unknown command '{args.command}'")
        return 1

if __name__ == '__main__':
    sys.exit(main())
