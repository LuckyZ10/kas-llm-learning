# DFT+LAMMPS Research Platform - User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Management](#project-management)
3. [Workflow Design](#workflow-design)
4. [Monitoring](#monitoring)
5. [Screening Results](#screening-results)
6. [Structure Visualization](#structure-visualization)
7. [Report Generation](#report-generation)

---

## Getting Started

### Quick Start with Docker

```bash
cd webui_v2
docker-compose up -d
```

Access the application:
- Web Interface: http://localhost:3000
- API Documentation: http://localhost:8000/docs

### Login

Default credentials for development:
- Username: `researcher`
- Password: `researcher`

---

## Project Management

### Creating a Project

1. Navigate to the **Projects** page
2. Click **New Project**
3. Fill in project details:
   - **Name**: Project identifier
   - **Type**: Battery screening, alloy design, etc.
   - **Material System**: Chemical formula (e.g., Li-Mn-O)
   - **Description**: Optional project description
4. Click **Create**

### Project Dashboard

Each project has a dashboard showing:
- Project status (Draft, Active, Completed)
- Total structures
- Completed/failed calculations
- Associated workflows

---

## Workflow Design

### Workflow Types

- **Active Learning**: Automated training data selection
- **High-Throughput Screening**: Large-scale property calculation
- **ML Training**: Machine learning potential training
- **MD Simulation**: Molecular dynamics simulations
- **DFT Calculation**: Density functional theory calculations

### Creating a Workflow

1. Go to **Workflows** page
2. Drag nodes from the palette:
   - **Input**: Structure data
   - **DFT Calculation**: Quantum mechanical calculations
   - **MD Simulation**: Molecular dynamics
   - **ML Training**: Neural network training
   - **Analysis**: Results processing
3. Connect nodes by dragging between connection points
4. Configure each node by clicking and setting parameters
5. Save and execute

### Node Configuration

Each node has specific configuration options:

**DFT Calculation**:
- Functional (PBE, HSE06, etc.)
- k-point grid
- Energy cutoff
- Convergence criteria

**MD Simulation**:
- Temperature
- Time step
- Ensemble (NVT, NPT, NVE)
- Simulation length

**ML Training**:
- Model type (DeePMD, NEP)
- Training parameters
- Validation split

---

## Monitoring

### System Overview

The **Monitoring** page provides:
- Active workflows count
- Running tasks
- Resource usage (CPU hours, memory)
- System health status

### ML Training Progress

View real-time training metrics:
- Loss curves
- Force/energy RMSE
- Learning rate schedule
- Validation performance

### Active Learning Progress

Track AL iterations:
- Number of iterations completed
- Candidate structures per iteration
- Model deviation statistics
- Convergence status

### MD Simulation Metrics

Monitor running MD simulations:
- Temperature profile
- Energy conservation
- Pressure/volume
- Diffusion coefficients

---

## Screening Results

### Filtering Results

Use filters to find promising materials:
- Formula search (e.g., "LiMnO2")
- Property ranges (band gap, conductivity)
- ML/DFT calculated
- Project membership

### Visualization

- **Property Map**: Scatter plot of formation energy vs band gap
- **Parallel Coordinates**: Multi-dimensional data visualization
- **Structure Comparison**: Side-by-side analysis

### Exporting Results

Export top candidates in multiple formats:
- CSV (spreadsheet)
- PDF (report)
- CIF/POSCAR (structure files)

---

## Structure Visualization

### 3D Viewer

The structure viewer supports:
- Ball-and-stick representation
- Space-filling model
- Unit cell display
- Element coloring

### Controls

- **Left click + drag**: Rotate
- **Right click + drag**: Pan
- **Scroll**: Zoom
- **Double click**: Reset view

### Supported Formats

- CIF (Crystallographic Information File)
- POSCAR (VASP format)
- XYZ (Cartesian coordinates)
- LAMMPS dump

---

## Report Generation

### Report Types

**Project Report**:
- Complete project summary
- All workflows and tasks
- Statistics and timelines
- Configuration details

**Screening Report**:
- Top candidates ranked
- Property tables
- Distribution plots
- Methodology description

**Workflow Report**:
- Execution timeline
- Task details
- Resource usage
- Error logs (if any)

### Customizing Reports

Reports can be customized to include:
- Charts and figures
- Structure images
- Custom branding
- Executive summary

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl/Cmd + K` | Quick search |
| `Ctrl/Cmd + S` | Save workflow |
| `Esc` | Close modal/cancel |
| `?` | Show help |

## Troubleshooting

### Common Issues

**WebSocket Connection Failed**:
- Check if backend is running
- Verify network connectivity
- Refresh the page

**No Training Data Showing**:
- Verify `lcurve.out` exists in model directory
- Check file permissions
- Review backend logs

**3D Viewer Not Loading**:
- Enable WebGL in browser
- Update graphics drivers
- Try a different browser

### Support

For technical support:
- Check API documentation: `/docs`
- Review backend logs
- Contact system administrator
