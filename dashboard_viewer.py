"""Simple dashboard viewer for ZenML pipelines using Python client."""
import os
from dotenv import load_dotenv
load_dotenv()

from zenml.client import Client
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

def show_pipelines():
    """Display all pipelines with runs."""
    client = Client()

    # Get all pipelines
    pipelines = client.list_pipelines()

    if not pipelines:
        console.print("[yellow]No pipelines found![/yellow]")
        return

    console.print(Panel.fit(
        f"[bold cyan]ZenML Pipelines Overview[/bold cyan]\n"
        f"Connected to: {os.getenv('ZENML_SERVER_URL', 'local')}",
        box=box.DOUBLE
    ))

    for pipeline in pipelines:
        # Create table for this pipeline
        table = Table(
            title=f"[bold green]{pipeline.name}[/bold green]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )

        table.add_column("Run Name", style="cyan", no_wrap=False)
        table.add_column("Status", justify="center")
        table.add_column("Started", style="dim")
        table.add_column("Duration", justify="right")

        # Get runs for this pipeline
        try:
            runs = client.list_pipeline_runs(pipeline_id=pipeline.id, size=10)

            if not runs:
                table.add_row(
                    "[dim]No runs yet[/dim]",
                    "-",
                    "-",
                    "-"
                )
            else:
                for run in runs:
                    # Status with color
                    status = run.status.value if hasattr(run.status, 'value') else str(run.status)
                    status_colored = {
                        'completed': '[green]✓ COMPLETED[/green]',
                        'failed': '[red]✗ FAILED[/red]',
                        'running': '[yellow]⚙ RUNNING[/yellow]',
                    }.get(status.lower(), f'[white]{status}[/white]')

                    # Format datetime
                    started = run.created.strftime("%Y-%m-%d %H:%M:%S") if run.created else "N/A"

                    # Calculate duration (if available)
                    duration = "N/A"
                    if hasattr(run, 'end_time') and run.end_time and run.created:
                        delta = run.end_time - run.created
                        duration = str(delta).split('.')[0]  # Remove microseconds

                    table.add_row(
                        run.name,
                        status_colored,
                        started,
                        duration
                    )

        except Exception as e:
            table.add_row(
                f"[red]Error fetching runs: {e}[/red]",
                "-",
                "-",
                "-"
            )

        console.print(table)
        console.print()  # Empty line between pipelines

def show_detailed_run(pipeline_name: str = None, run_id: str = None):
    """Show detailed information about a specific run."""
    client = Client()

    if run_id:
        # Get specific run
        try:
            run = client.get_pipeline_run(run_id)
        except:
            console.print(f"[red]Run {run_id} not found[/red]")
            return
    elif pipeline_name:
        # Get latest run for pipeline
        pipelines = [p for p in client.list_pipelines() if p.name == pipeline_name]
        if not pipelines:
            console.print(f"[red]Pipeline {pipeline_name} not found[/red]")
            return

        runs = client.list_pipeline_runs(pipeline_id=pipelines[0].id, size=1)
        if not runs:
            console.print(f"[yellow]No runs found for {pipeline_name}[/yellow]")
            return
        run = runs[0]
    else:
        console.print("[red]Please provide pipeline_name or run_id[/red]")
        return

    # Display run details
    console.print(Panel.fit(
        f"[bold cyan]Run Details[/bold cyan]\n"
        f"Pipeline: {run.pipeline.name if hasattr(run, 'pipeline') else 'N/A'}\n"
        f"Run ID: {run.id}\n"
        f"Status: {run.status}",
        box=box.DOUBLE
    ))

    # Show steps
    try:
        steps = run.steps
        if steps:
            step_table = Table(title="Pipeline Steps", box=box.ROUNDED)
            step_table.add_column("Step", style="cyan")
            step_table.add_column("Status", justify="center")
            step_table.add_column("Started")
            step_table.add_column("Duration")

            for step_name, step_info in steps.items():
                status = step_info.status if hasattr(step_info, 'status') else 'unknown'
                started = step_info.created.strftime("%H:%M:%S") if hasattr(step_info, 'created') and step_info.created else "N/A"

                step_table.add_row(
                    step_name,
                    str(status),
                    started,
                    "N/A"
                )

            console.print(step_table)
    except Exception as e:
        console.print(f"[yellow]Could not fetch step details: {e}[/yellow]")

if __name__ == "__main__":
    import sys

    console.print("\n[bold]ZenML Pipeline Dashboard Viewer[/bold]\n", style="bold blue")

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "run" and len(sys.argv) > 2:
            # Show specific run details
            show_detailed_run(run_id=sys.argv[2])
        elif command == "pipeline" and len(sys.argv) > 2:
            # Show latest run for pipeline
            show_detailed_run(pipeline_name=sys.argv[2])
        else:
            console.print("[red]Unknown command. Use: python dashboard_viewer.py [run <run_id> | pipeline <name>][/red]")
    else:
        # Show all pipelines
        show_pipelines()

    console.print("\n[dim]Tip: Use 'python dashboard_viewer.py pipeline <name>' for details[/dim]\n")
