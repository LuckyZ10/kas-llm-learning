# Lab Automation Module

Comprehensive laboratory automation system for the DFT-LAMMPS research platform.

## Overview

This module provides complete laboratory automation capabilities including:

- **Equipment Abstraction Layer**: Unified interfaces for robotics, synthesis equipment, and characterization instruments
- **Synthesis Planning**: Intelligent planners for powder synthesis and thin film deposition
- **Characterization Data Parser**: Automated parsing of XRD, SEM, and electrochemical data
- **Closed-Loop Control**: PID, MPC, and adaptive feedback control systems
- **LIMS Integration**: Laboratory Information Management System connectivity
- **ROS2 Interface**: Robot Operating System 2 integration for robotic control
- **Complete Workflows**: End-to-end automation from synthesis to analysis

## Features

### Equipment Support

- **Robotic Arms**: Universal Robots (UR), FANUC, KUKA
- **Synthesis Equipment**: Furnaces, mixers, deposition systems
- **Characterization Instruments**: XRD, SEM, TEM, electrochemical workstations

### Synthesis Methods

- Powder: Solid-state, sol-gel, hydrothermal, co-precipitation
- Thin Film: CVD, PVD, ALD, spin coating, sputtering

### Characterization Techniques

- X-ray Diffraction (XRD) with peak detection and phase identification
- Scanning Electron Microscopy (SEM) with particle analysis
- Electrochemical: CV, EIS, GCD with automated analysis

### Control Systems

- PID controllers with auto-tuning
- Model Predictive Control (MPC)
- Bayesian optimization
- Multi-variable control

## Quick Start

### Installation

```bash
# Install the module
pip install -e .

# Install optional dependencies for ROS2 support
pip install rclpy

# Install optional dependencies for image analysis
pip install scikit-image Pillow
```

### Basic Usage

```python
import asyncio
from dftlammps.lab_automation import (
    EquipmentManager, RobotArm, PowderSynthesizer
)

async def main():
    # Create equipment manager
    manager = EquipmentManager()
    
    # Add robot
    robot = RobotArm(
        equipment_id="robot_001",
        name="UR5e Robot",
        model="UR5e"
    )
    manager.register(robot)
    
    # Initialize
    await manager.connect_all()
    await robot.initialize()
    
    # Move robot
    from dftlammps.lab_automation.equipment import Position3D
    target = Position3D(x=0.3, y=0.2, z=0.5)
    await robot.move_to(target)

# Run
asyncio.run(main())
```

### Synthesis Planning

```python
from dftlammps.lab_automation import PowderSynthesizer

async def synthesize():
    synthesizer = PowderSynthesizer()
    
    # List available recipes
    recipes = synthesizer.list_recipes()
    print(f"Available recipes: {recipes}")
    
    # Execute synthesis
    result = await synthesizer.execute_recipe(
        recipe_id="LCO_SSD_001",
        sample_id="LCO_001",
        equipment_manager=None
    )
    
    print(f"Synthesis complete: {result}")

asyncio.run(synthesize())
```

### Characterization Data Parsing

```python
import numpy as np
from dftlammps.lab_automation import XRDParser, XRDData

# Create or load XRD data
two_theta = np.linspace(10, 80, 1401)
intensity = np.random.rand(1401) * 100

xrd_data = XRDData(
    two_theta=two_theta,
    intensity=intensity,
    sample_id="SAMPLE_001"
)

# Analyze
parser = XRDParser()
peaks = parser.find_peaks(xrd_data)
phases = parser.phase_identification(peaks)

print(f"Found {len(peaks)} peaks")
print(f"Identified phases: {phases}")
```

## Module Structure

```
dftlammps/lab_automation/
├── __init__.py              # Package initialization
├── equipment.py             # Equipment abstraction layer
├── synthesis.py             # Synthesis planning
├── characterization.py      # Data parsers and analysis
├── control.py               # Feedback control systems
├── lims.py                  # LIMS integration
├── ros2_interface.py        # ROS2 robot control
├── workflows.py             # Complete workflow examples
├── examples/                # Usage examples
│   └── demo_workflow.py
└── tests/                   # Unit tests
    └── test_lab_automation.py
```

## API Reference

### Equipment

- `EquipmentManager`: Manages all laboratory equipment
- `RobotArm`: Robotic arm interface
- `SynthesisEquipment`: Synthesis equipment (furnaces, mixers)
- `CharacterizationInstrument`: Characterization instruments

### Synthesis

- `SynthesisPlanner`: Abstract base for synthesis planners
- `PowderSynthesizer`: Powder synthesis planner
- `ThinFilmDepositor`: Thin film deposition planner
- `SynthesisRecipe`: Recipe definition and management

### Characterization

- `XRDParser`: X-ray diffraction data parser
- `SEMParser`: SEM image parser
- `ElectrochemicalParser`: CV, EIS, GCD parser
- `DataAggregator`: Multi-technique data correlation

### Control

- `PIDController`: PID feedback controller
- `MPCController`: Model predictive controller
- `OptimizationEngine`: Parameter optimization
- `ControlLoop`: Main control loop manager

### LIMS

- `LIMSClient`: Abstract LIMS interface
- `MockLIMSClient`: Testing/mock implementation
- `SampleTracker`: Sample lifecycle management
- `ExperimentLogger`: Experiment logging

### ROS2

- `create_ros_interface`: Factory for ROS2 interfaces
- `RobotController`: High-level robot control
- `NavigationController`: Mobile robot navigation
- `EquipmentROSBridge`: Equipment-ROS2 bridge

## Configuration

### ROS2 Setup

To use real ROS2 integration:

1. Install ROS2 (Humble or later)
2. Source ROS2 environment: `source /opt/ros/humble/setup.bash`
3. Use `create_ros_interface(node_name, use_mock=False)`

### LIMS Configuration

```python
from dftlammps.lab_automation import RESTLIMSClient

lims = RESTLIMSClient(
    base_url="https://lims.example.com",
    api_key="your-api-key"
)

await lims.connect()
```

## Testing

```bash
# Run all tests
pytest dftlammps/lab_automation/tests/ -v

# Run specific test
pytest dftlammps/lab_automation/tests/test_lab_automation.py::test_pid_controller -v

# Run with coverage
pytest --cov=dftlammps.lab_automation dftlammps/lab_automation/tests/
```

## Examples

See `dftlammps/lab_automation/examples/demo_workflow.py` for complete examples:

1. Equipment control
2. Synthesis planning
3. Characterization parsing
4. Feedback control
5. LIMS integration
6. ROS2 robot control
7. Complete workflows

Run examples:

```bash
cd dftlammps/lab_automation/examples
python demo_workflow.py
```

## Advanced Workflows

### Closed-Loop Optimization

```python
from dftlammps.lab_automation.workflows import ClosedLoopOptimization

async def optimize():
    workflow = ClosedLoopOptimization()
    await workflow.setup()
    
    result = await workflow.optimize_synthesis(
        target_material="LiCoO2",
        target_property="capacity",
        target_value=150.0,
        max_iterations=10
    )
    
    print(f"Optimization complete: {result}")

asyncio.run(optimize())
```

### Complete Pipeline

```python
from dftlammps.lab_automation.workflows import CompleteWorkflow

async def full_pipeline():
    workflow = CompleteWorkflow()
    await workflow.setup()
    
    result = await workflow.run_full_pipeline(
        target_material="ZnO",
        material_type="thin_film",
        run_characterization=True,
        run_optimization=True
    )
    
    report = workflow.generate_report(result)
    print(report)

asyncio.run(full_pipeline())
```

## Troubleshooting

### ROS2 Not Available

If ROS2 is not installed, the module automatically uses mock mode. To force mock mode:

```python
from dftlammps.lab_automation.ros2_interface import create_ros_interface

ros = create_ros_interface("my_node", use_mock=True)
```

### Equipment Connection Issues

Check equipment connection parameters:

```python
robot = RobotArm(
    equipment_id="robot_001",
    name="UR5e",
    connection_params={'host': '192.168.1.100', 'port': 30002}
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Citation

If you use this module in your research, please cite:

```
DFT-LAMMPS Lab Automation Module
Version 1.0.0
Laboratory Automation for Materials Science Research
```

## Support

For issues and feature requests, please use the GitHub issue tracker.

## Roadmap

- [ ] Support for additional robot brands (ABB, Kawasaki)
- [ ] Machine learning-based synthesis optimization
- [ ] Real-time data visualization dashboard
- [ ] Integration with additional LIMS providers
- [ ] Support for cryo-EM and AFM data parsing
