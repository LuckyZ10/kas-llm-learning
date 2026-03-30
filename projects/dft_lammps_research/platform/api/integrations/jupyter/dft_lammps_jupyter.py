"""
Jupyter Notebook Integration

Magic commands and widgets for using DFT+LAMMPS API in Jupyter notebooks.

Installation:
    %pip install dft-lammps-jupyter

Usage:
    %load_ext dft_lammps_jupyter
    
    # Set API key
    %dftlammps_api_key your-api-key-here
    
    # Create and run calculations
    %%dftlammps_calc project_id=proj_123
    structure = read("Li2S.cif")
    result = run_dft(structure, params={"ecut": 500})
"""

__version__ = "1.0.0"

import json
import os
from typing import Optional, Dict, Any
from IPython.core.magic import register_line_magic, register_cell_magic
from IPython.display import display, HTML, JSON
from IPython import get_ipython
import ipywidgets as widgets

from dft_lammps import Client

# Global client instance
_client: Optional[Client] = None


def get_client() -> Client:
    """Get or create API client"""
    global _client
    if _client is None:
        api_key = os.getenv("DFT_LAMMPS_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not set. Use %dftlammps_api_key or set DFT_LAMMPS_API_KEY env var."
            )
        _client = Client(api_key=api_key)
    return _client


@register_line_magic
def dftlammps_api_key(key):
    """Set API key for DFT+LAMMPS"""
    global _client
    os.environ["DFT_LAMMPS_API_KEY"] = key
    _client = Client(api_key=key)
    print("✓ API key set successfully")


@register_line_magic
def dftlammps_status(line):
    """Check API status and show usage"""
    try:
        client = get_client()
        health = client.health()
        usage = client.usage()
        
        html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 20px; border-radius: 10px; font-family: sans-serif;">
            <h3>🔬 DFT+LAMMPS API Status</h3>
            <p><strong>Status:</strong> <span style="color: #90EE90;">{health.get('status', 'unknown').upper()}</span></p>
            <p><strong>Version:</strong> {health.get('version', 'unknown')}</p>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <h4>Usage Today</h4>
            <p>Requests: {usage.get('requests', {}).get('total', 0)}</p>
            <p>Calculations: {usage.get('calculations', {}).get('total', 0)}</p>
            <p>Storage: {usage.get('storage', {}).get('used_gb', 0)} GB / {usage.get('storage', {}).get('total_gb', 1)} GB</p>
        </div>
        """
        display(HTML(html))
    except Exception as e:
        display(HTML(f"""
        <div style="background: #ffebee; color: #c62828; padding: 20px; border-radius: 10px;">
            <h3>❌ Connection Error</h3>
            <p>{str(e)}</p>
        </div>
        """))


@register_cell_magic
def dftlammps_calc(line, cell):
    """
    Execute calculation in a project context.
    
    Usage:
        %%dftlammps_calc project_id=proj_123 type=dft
        # Python code to prepare calculation
        structure = load_structure("Li2S.cif")
        result = submit_calculation(structure)
    """
    # Parse arguments
    args = {}
    for arg in line.split():
        if "=" in arg:
            key, value = arg.split("=", 1)
            args[key] = value
    
    project_id = args.get("project_id")
    calc_type = args.get("type", "dft")
    
    if not project_id:
        print("❌ Error: project_id required")
        return
    
    try:
        client = get_client()
        
        # Create progress widget
        progress = widgets.FloatProgress(
            value=0,
            min=0,
            max=100,
            description='Submitting...',
            bar_style='info',
            style={'bar_color': '#667eea'}
        )
        display(progress)
        
        # Execute cell code to get structure
        namespace = {}
        exec(cell, namespace)
        
        # Get structure from namespace
        structure = namespace.get("structure")
        
        if structure is None:
            progress.close()
            print("❌ Error: No 'structure' variable found in cell")
            return
        
        # Submit calculation
        progress.value = 50
        progress.description = 'Processing...'
        
        calc = client.calculations.submit(
            project_id=project_id,
            structure=structure,
            calculation_type=calc_type,
            parameters=namespace.get("params", {})
        )
        
        progress.value = 100
        progress.description = 'Complete!'
        progress.bar_style = 'success'
        
        # Display result
        display(HTML(f"""
        <div style="background: #e8f5e9; padding: 15px; border-radius: 8px; margin-top: 10px;">
            <h4>✓ Calculation Submitted</h4>
            <p><strong>ID:</strong> <code>{calc.id}</code></p>
            <p><strong>Status:</strong> {calc.status}</p>
            <p><strong>Type:</strong> {calc.calculation_type}</p>
            <a href="#" onclick="alert('Poll with: client.calculations.get(\'{calc.id}\')')">
               Check Status →
            </a>
        </div>
        """))
        
    except Exception as e:
        display(HTML(f"""
        <div style="background: #ffebee; color: #c62828; padding: 15px; border-radius: 8px;">
            <h4>❌ Error</h4>
            <pre>{str(e)}</pre>
        </div>
        """))


class ProjectExplorer:
    """Interactive project explorer widget"""
    
    def __init__(self):
        self.client = get_client()
        self.projects = []
        self.selected_project = None
        
    def display(self):
        """Display the explorer UI"""
        # Load projects
        try:
            result = self.client.projects.list()
            self.projects = result.get("items", [])
        except Exception as e:
            display(HTML(f"""
            <div style="color: red;">Failed to load projects: {e}</div>
            """))
            return
        
        # Create dropdown
        project_options = [(p["name"], p["id"]) for p in self.projects]
        
        dropdown = widgets.Dropdown(
            options=project_options,
            description='Project:',
            style={'description_width': 'initial'}
        )
        
        # Info panel
        info_html = widgets.HTML(value="Select a project to view details")
        
        # Button
        refresh_btn = widgets.Button(
            description='🔄 Refresh',
            button_style='info'
        )
        
        new_project_btn = widgets.Button(
            description='➕ New Project',
            button_style='success'
        )
        
        # Layout
        buttons = widgets.HBox([refresh_btn, new_project_btn])
        ui = widgets.VBox([widgets.HBox([dropdown, buttons]), info_html])
        
        # Event handlers
        def on_project_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                project_id = change['new']
                self._update_info(project_id, info_html)
        
        def on_refresh(_):
            self.display()
        
        def on_new_project(_):
            self._show_new_project_dialog()
        
        dropdown.observe(on_project_change)
        refresh_btn.on_click(on_refresh)
        new_project_btn.on_click(on_new_project)
        
        display(ui)
    
    def _update_info(self, project_id: str, html_widget):
        """Update info panel"""
        try:
            project = self.client.projects.get(project_id)
            html = f"""
            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin-top: 10px;">
                <h4>{project.name}</h4>
                <p><strong>ID:</strong> <code>{project.id}</code></p>
                <p><strong>Status:</strong> <span style="color: {'green' if project.status == 'active' else 'orange'};">
                    {project.status}
                </span></p>
                <p><strong>Calculations:</strong> {project.completed_calculations} completed, 
                    {project.failed_calculations} failed</p>
                <p><strong>Created:</strong> {project.created_at}</p>
                <hr>
                <p><em>{project.description or 'No description'}</em></p>
            </div>
            """
            html_widget.value = html
        except Exception as e:
            html_widget.value = f"Error: {e}"
    
    def _show_new_project_dialog(self):
        """Show new project creation dialog"""
        name_input = widgets.Text(
            placeholder='Project name',
            description='Name:',
        )
        
        desc_input = widgets.Textarea(
            placeholder='Description',
            description='Desc:',
            layout=widgets.Layout(height='60px')
        )
        
        type_dropdown = widgets.Dropdown(
            options=['battery_screening', 'catalysis', 'phonon', 'custom'],
            value='battery_screening',
            description='Type:',
        )
        
        create_btn = widgets.Button(
            description='Create',
            button_style='success'
        )
        
        cancel_btn = widgets.Button(
            description='Cancel',
            button_style=''
        )
        
        dialog = widgets.VBox([
            widgets.HTML("<h4>Create New Project</h4>"),
            name_input,
            desc_input,
            type_dropdown,
            widgets.HBox([create_btn, cancel_btn])
        ])
        
        def on_create(_):
            try:
                project = self.client.projects.create(
                    name=name_input.value,
                    description=desc_input.value,
                    project_type=type_dropdown.value
                )
                dialog.close()
                display(HTML(f"✓ Project created: <code>{project.id}</code>"))
                self.display()  # Refresh
            except Exception as e:
                display(HTML(f"❌ Error: {e}"))
        
        def on_cancel(_):
            dialog.close()
        
        create_btn.on_click(on_create)
        cancel_btn.on_click(on_cancel)
        
        display(dialog)


def explore_projects():
    """Launch project explorer"""
    explorer = ProjectExplorer()
    explorer.display()


class CalculationMonitor:
    """Real-time calculation monitor widget"""
    
    def __init__(self, calculation_id: str):
        self.calculation_id = calculation_id
        self.client = get_client()
        
    def display(self):
        """Display monitoring widget"""
        status_label = widgets.Label(value="Monitoring...")
        progress = widgets.FloatProgress(value=0, min=0, max=100, description='Progress:')
        output = widgets.Output()
        
        ui = widgets.VBox([status_label, progress, output])
        display(ui)
        
        # Start monitoring loop
        import threading
        
        def monitor():
            while True:
                try:
                    calc = self.client.calculations.get(self.calculation_id)
                    
                    status_label.value = f"Status: {calc.status.upper()}"
                    
                    if calc.status == "completed":
                        progress.value = 100
                        progress.bar_style = 'success'
                        with output:
                            print("Results:", json.dumps(calc.results, indent=2))
                        break
                    elif calc.status == "failed":
                        progress.bar_style = 'danger'
                        with output:
                            print("Error:", calc.error_message)
                        break
                    else:
                        # Estimate progress
                        elapsed = (datetime.utcnow() - calc.created_at).total_seconds()
                        estimated_total = 300  # 5 min estimate
                        progress.value = min(95, (elapsed / estimated_total) * 100)
                    
                    time.sleep(5)
                    
                except Exception as e:
                    with output:
                        print(f"Error: {e}")
                    break
        
        thread = threading.Thread(target=monitor)
        thread.daemon = True
        thread.start()


def monitor_calculation(calculation_id: str):
    """Monitor a calculation in real-time"""
    monitor = CalculationMonitor(calculation_id)
    monitor.display()


# Export magic commands
def load_ipython_extension(ipython):
    """Load the extension in IPython"""
    ipython.register_magic_function(dftlammps_api_key, 'line')
    ipython.register_magic_function(dftlammps_status, 'line')
    ipython.register_magic_function(dftlammps_calc, 'cell')
    print("✓ DFT+LAMMPS Jupyter extension loaded")
    print("  Commands: %dftlammps_api_key, %dftlammps_status, %%dftlammps_calc")
