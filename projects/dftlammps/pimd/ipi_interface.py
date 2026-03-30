"""
i-PI Interface for Path Integral Molecular Dynamics
====================================================

This module provides a comprehensive interface to i-PI for performing
path integral molecular dynamics (PIMD), ring polymer MD (RPMD), and
related quantum dynamics simulations.

i-PI is a Python package for path integral molecular dynamics simulations,
designed for ab initio simulations. It interfaces with external electronic
structure codes like VASP, Quantum ESPRESSO, LAMMPS, etc.

Classes:
--------
- IPIConfig: Configuration for i-PI simulations
- IPIInterface: Main interface to i-PI
- PIMDSimulation: Path Integral MD simulation
- RPMDSimulation: Ring Polymer MD simulation
- PIConvergenceChecker: Check PIMD convergence with bead number
- IPISocket: Socket interface for communication with clients

Functions:
----------
- run_pimd: Run PIMD simulation
- run_rpmd: Run RPMD simulation
- check_convergence: Check convergence of PIMD properties
- generate_input_xml: Generate i-PI input XML

References:
-----------
- Tuckerman, M. E. (2010). Statistical Mechanics: Theory and Molecular Simulation
- Ceriotti et al. (2010). Efficient calculation of free energy surfaces
- Marx & Parrinello (1996). Ab initio path integral molecular dynamics

Example:
--------
>>> from dftlammps.pimd import IPIConfig, PIMDSimulation
>>> config = IPIConfig(n_beads=32, temperature=300, mode='pimd')
>>> sim = PIMDSimulation(config)
>>> sim.run()
"""

import os
import re
import sys
import time
import socket
import struct
import logging
import tempfile
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from pathlib import Path
from enum import Enum, auto
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)


class IPIMode(Enum):
    """i-PI simulation modes."""
    PIMD = "pimd"           # Path Integral Molecular Dynamics
    RPMD = "rpmd"           # Ring Polymer Molecular Dynamics
    CMD = "cmd"             # Centroid Molecular Dynamics
    TRPMD = "trpmd"         # Thermostatted RPMD
    PIGLET = "piglet"       # Path Integral GLE Thermostat
    INSTANTON = "instanton" # Instanton optimization


class ThermostatType(Enum):
    """Thermostat types for PIMD."""
    LANGEVIN = "langevin"
    GLE = "gle"
    SVR = "svr"
    CSVR = "csvr"
    PILE = "pile"
    PILE_L = "pile_l"


class IntegratorType(Enum):
    """Integrator types."""
    VELOCITY_VERLET = "vv"
    VERLET = "verlet"
    LANGEVIN = "langevin"
    GLE = "gle"
    NVT = "nvt"
    NPT = "npt"


@dataclass
class IPIConfig:
    """Configuration for i-PI simulation.
    
    Attributes:
        n_beads: Number of path integral beads (P)
        temperature: Temperature in Kelvin
        mode: Simulation mode (PIMD, RPMD, CMD, etc.)
        timestep: Time step in femtoseconds
        n_steps: Number of MD steps
        thermostat: Thermostat type
        friction: Friction coefficient for Langevin (1/ps)
        cell_dimensions: Simulation cell (3x3 matrix or 3-vector)
        output_prefix: Prefix for output files
        restart_file: Path to restart file
        trajectory_file: Trajectory output file
        properties_file: Properties output file
        checkpoint_freq: Checkpoint frequency in steps
        trajectory_freq: Trajectory output frequency
        properties_freq: Properties calculation frequency
        client_address: Socket address for client (e.g., 'localhost')
        client_port: Socket port for client
        client_mode: Socket mode ('unix' or 'inet')
        ensemble: Thermodynamic ensemble ('nvt', 'npt', etc.)
        pressure: Pressure for NPT ensemble (GPa)
        barostat_tau: Barostat time constant (fs)
        centroid_friction: Centroid friction for RPMD (1/ps)
        bead_thermostat_tau: Bead thermostat time constant (fs)
        constraint: List of constraints (fix atoms, etc.)
        extras: Additional i-PI parameters
    """
    n_beads: int = 32
    temperature: float = 300.0
    mode: IPIMode = IPIMode.PIMD
    timestep: float = 0.5
    n_steps: int = 10000
    thermostat: ThermostatType = ThermostatType.PILE_L
    friction: float = 1.0
    cell_dimensions: Optional[np.ndarray] = None
    output_prefix: str = "ipi"
    restart_file: Optional[str] = None
    trajectory_file: str = "trajectory.xyz"
    properties_file: str = "properties.out"
    checkpoint_freq: int = 1000
    trajectory_freq: int = 10
    properties_freq: int = 1
    client_address: str = "localhost"
    client_port: int = 31415
    client_mode: str = "inet"
    ensemble: str = "nvt"
    pressure: float = 0.0
    barostat_tau: float = 100.0
    centroid_friction: float = 0.0
    bead_thermostat_tau: float = 100.0
    constraint: Optional[List[Dict]] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.n_beads < 1:
            raise ValueError(f"n_beads must be >= 1, got {self.n_beads}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.timestep <= 0:
            raise ValueError(f"timestep must be > 0, got {self.timestep}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {self.n_steps}")
        
        if self.cell_dimensions is not None:
            self.cell_dimensions = np.array(self.cell_dimensions)


@dataclass
class PIMDResults:
    """Results from PIMD simulation.
    
    Attributes:
        positions: Positions of all beads [n_frames, n_beads, n_atoms, 3]
        velocities: Velocities of all beads [n_frames, n_beads, n_atoms, 3]
        forces: Forces on all beads [n_frames, n_beads, n_atoms, 3]
        potential_energy: Potential energy time series
        kinetic_energy: Kinetic energy time series
        total_energy: Total energy time series
        temperature: Temperature time series
        pressure: Pressure time series
        centroid_positions: Centroid positions [n_frames, n_atoms, 3]
        bead_radii: Ring polymer radii [n_frames, n_atoms]
        primitive_energy: Primitive energy estimator
        virial_energy: Virial energy estimator
        centroid_virial_energy: Centroid virial energy estimator
        output_files: Dictionary of output file paths
        convergence_data: Convergence data with bead number
    """
    positions: Optional[np.ndarray] = None
    velocities: Optional[np.ndarray] = None
    forces: Optional[np.ndarray] = None
    potential_energy: Optional[np.ndarray] = None
    kinetic_energy: Optional[np.ndarray] = None
    total_energy: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    pressure: Optional[np.ndarray] = None
    centroid_positions: Optional[np.ndarray] = None
    bead_radii: Optional[np.ndarray] = None
    primitive_energy: Optional[np.ndarray] = None
    virial_energy: Optional[np.ndarray] = None
    centroid_virial_energy: Optional[np.ndarray] = None
    output_files: Dict[str, str] = field(default_factory=dict)
    convergence_data: Optional[Dict] = None
    
    def get_average_energy(self) -> Tuple[float, float]:
        """Get average total energy with standard error.
        
        Returns:
            (mean_energy, std_error)
        """
        if self.total_energy is None:
            raise ValueError("Total energy data not available")
        
        mean = np.mean(self.total_energy)
        # Standard error accounting for correlation
        n = len(self.total_energy)
        std = np.std(self.total_energy, ddof=1)
        stderr = std / np.sqrt(n)
        
        return mean, stderr
    
    def get_quantum_kinetic_energy(self) -> Tuple[float, float]:
        """Get quantum kinetic energy using virial estimator.
        
        Returns:
            (mean_ke, std_error)
        """
        if self.virial_energy is not None:
            data = self.virial_energy
        elif self.kinetic_energy is not None:
            data = self.kinetic_energy
        else:
            raise ValueError("Kinetic energy data not available")
        
        mean = np.mean(data)
        n = len(data)
        std = np.std(data, ddof=1)
        stderr = std / np.sqrt(n)
        
        return mean, stderr
    
    def calculate_bead_fluctuations(self) -> np.ndarray:
        """Calculate bead position fluctuations (measure of quantum spread).
        
        Returns:
            RMS bead fluctuations [n_atoms]
        """
        if self.positions is None:
            raise ValueError("Position data not available")
        
        # positions shape: [n_frames, n_beads, n_atoms, 3]
        centroid = np.mean(self.positions, axis=1, keepdims=True)
        fluctuations = self.positions - centroid
        rms_fluct = np.sqrt(np.mean(np.sum(fluctuations**2, axis=-1), axis=(0, 1)))
        
        return rms_fluct


class IPISocket:
    """Socket interface for i-PI communication.
    
    i-PI communicates with external codes (drivers) through sockets.
    This class handles the socket communication protocol.
    
    Protocol:
    ---------
    - i-PI acts as server, waits for clients to connect
    - Communication is message-based with specific header format
    - Supports both Unix domain sockets and Internet sockets
    
    Reference:
    ----------
    http://ipi-code.org/
    """
    
    # Message types
    MSG_STATUS = 0
    MSG_POSDATA = 1
    MSG_GETFORCE = 2
    MSG_FORCE = 3
    MSG_END = 4
    
    # Header size
    HDRLEN = 12
    
    def __init__(self, address: str = "localhost", port: int = 31415,
                 mode: str = "inet"):
        """Initialize socket interface.
        
        Args:
            address: Socket address (hostname for inet, path for unix)
            port: Port number for inet sockets
            mode: 'inet' or 'unix'
        """
        self.address = address
        self.port = port
        self.mode = mode
        self.socket = None
        self.conn = None
        self.hasdata = 0
        
    def create_socket(self):
        """Create server socket."""
        if self.mode == "unix":
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                os.remove(self.address)
            except FileNotFoundError:
                pass
            self.socket.bind(self.address)
        else:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.address, self.port))
        
        self.socket.listen(1)
        logger.info(f"Socket listening on {self.address}:{self.port}")
        
    def accept_connection(self):
        """Accept client connection."""
        self.conn, addr = self.socket.accept()
        logger.info(f"Connection from {addr}")
        
    def close(self):
        """Close socket connection."""
        if self.conn:
            self.conn.close()
        if self.socket:
            self.socket.close()
            
    def send_message(self, msg_type: int):
        """Send message header.
        
        Args:
            msg_type: Type of message
        """
        header = f"{msg_type:12d}"
        self.conn.sendall(header.encode())
        
    def receive_message(self) -> int:
        """Receive message header.
        
        Returns:
            Message type
        """
        header = self.conn.recv(self.HDRLEN)
        if len(header) == 0:
            return -1
        return int(header.decode())
    
    def send_data(self, data: np.ndarray):
        """Send numpy array data.
        
        Args:
            data: Array to send
        """
        # Send shape
        shape = np.array(data.shape, dtype=np.int32)
        self.conn.sendall(shape.tobytes())
        # Send data
        self.conn.sendall(data.astype(np.float64).tobytes())
        
    def receive_data(self) -> np.ndarray:
        """Receive numpy array data.
        
        Returns:
            Received array
        """
        # Receive shape
        shape_bytes = self.conn.recv(12)  # 3 * 4 bytes for int32
        shape = np.frombuffer(shape_bytes, dtype=np.int32)
        
        # Receive data
        size = np.prod(shape)
        data_bytes = self.conn.recv(size * 8)  # 8 bytes per float64
        data = np.frombuffer(data_bytes, dtype=np.float64).reshape(shape)
        
        return data


class IPIInterface:
    """Main interface to i-PI.
    
    This class manages the i-PI simulation workflow including:
    - Input XML generation
    - Socket communication setup
    - Process management
    - Output parsing
    
    Attributes:
        config: IPIConfig instance
        socket: IPISocket for communication
        process: i-PI subprocess
        working_dir: Working directory for simulation
    """
    
    def __init__(self, config: IPIConfig, working_dir: Optional[str] = None):
        """Initialize i-PI interface.
        
        Args:
            config: IPIConfig instance
            working_dir: Working directory (creates temp if None)
        """
        self.config = config
        self.working_dir = working_dir or tempfile.mkdtemp(prefix="ipi_")
        os.makedirs(self.working_dir, exist_ok=True)
        
        self.socket = None
        self.process = None
        self.input_file = None
        
    def generate_input_xml(self, atoms: Optional[Any] = None) -> str:
        """Generate i-PI input XML file.
        
        Args:
            atoms: ASE atoms object (optional, for initial structure)
            
        Returns:
            Path to generated XML file
        """
        # Create root element
        simulation = ET.Element("simulation")
        simulation.set("verbosity", "medium")
        simulation.set("mode", self.config.mode.value)
        
        # Output section
        output = ET.SubElement(simulation, "output")
        output.set("prefix", self.config.output_prefix)
        
        # Properties output
        properties = ET.SubElement(output, "properties")
        properties.set("stride", str(self.config.properties_freq))
        properties.set("filename", self.config.properties_file)
        
        # Standard properties
        for prop in ["time", "temperature", "kinetic_md", "potential", "total_energy",
                     "kinetic_cv", "volume", "pressure"]:
            ET.SubElement(properties, "property", name=prop)
        
        # Trajectory output
        trajectory = ET.SubElement(output, "trajectory")
        trajectory.set("stride", str(self.config.trajectory_freq))
        trajectory.set("filename", self.config.trajectory_file)
        trajectory.set("format", "xyz")
        trajectory.set("cell_units", "angstrom")
        
        # Checkpoint
        checkpoint = ET.SubElement(output, "checkpoint")
        checkpoint.set("stride", str(self.config.checkpoint_freq))
        checkpoint.set("filename", "checkpoint")
        
        # System section
        system = ET.SubElement(simulation, "system")
        
        # Initialize section
        initialize = ET.SubElement(system, "initialize")
        initialize.set("nbeads", str(self.config.n_beads))
        
        if atoms is not None:
            # Write initial positions to file
            init_file = os.path.join(self.working_dir, "init.xyz")
            self._write_xyz(atoms, init_file)
            initialize.set("file", init_file)
            initialize.set("format", "xyz")
            initialize.set("cell", self._cell_to_string(atoms.get_cell()))
        elif self.config.cell_dimensions is not None:
            initialize.set("cell", self._cell_to_string(self.config.cell_dimensions))
        
        # Ensemble
        ensemble = ET.SubElement(system, "ensemble")
        ensemble.set("mode", self.config.ensemble)
        
        temperature = ET.SubElement(ensemble, "temperature")
        temperature.set("units", "kelvin")
        temperature.text = str(self.config.temperature)
        
        if self.config.ensemble == "npt":
            pressure = ET.SubElement(ensemble, "pressure")
            pressure.set("units", "gigapascal")
            pressure.text = str(self.config.pressure)
        
        # Motion section
        motion = ET.SubElement(simulation, "motion")
        motion.set("mode", "dynamics")
        
        dynamics = ET.SubElement(motion, "dynamics")
        dynamics.set("mode", self._get_integrator())
        
        timestep = ET.SubElement(dynamics, "timestep")
        timestep.set("units", "femtosecond")
        timestep.text = str(self.config.timestep)
        
        # Thermostat
        thermostat = ET.SubElement(dynamics, "thermostat")
        thermostat.set("mode", self.config.thermostat.value)
        
        if self.config.thermostat in [ThermostatType.PILE, ThermostatType.PILE_L]:
            pile_lambda = ET.SubElement(thermostat, "pile_lambda")
            pile_lambda.text = "0.5"
        
        tau = ET.SubElement(thermostat, "tau")
        tau.set("units", "femtosecond")
        tau.text = str(self.config.bead_thermostat_tau)
        
        # Barostat for NPT
        if self.config.ensemble == "npt":
            barostat = ET.SubElement(dynamics, "barostat")
            barostat.set("mode", "isotropic")
            
            tau_baro = ET.SubElement(barostat, "tau")
            tau_baro.set("units", "femtosecond")
            tau_baro.text = str(self.config.barostat_tau)
        
        # FF socket for client
        ffsocket = ET.SubElement(simulation, "ffsocket")
        ffsocket.set("mode", self.config.client_mode)
        ffsocket.set("name", "lammps")
        
        if self.config.client_mode == "unix":
            address = ET.SubElement(ffsocket, "address")
            address.text = os.path.join(self.working_dir, "ipi_socket")
        else:
            address = ET.SubElement(ffsocket, "address")
            address.text = self.config.client_address
            port = ET.SubElement(ffsocket, "port")
            port.text = str(self.config.client_port)
        
        # Write XML
        xml_str = ET.tostring(simulation, encoding="unicode")
        xml_str = self._prettify_xml(xml_str)
        
        input_file = os.path.join(self.working_dir, "input.xml")
        with open(input_file, "w") as f:
            f.write(xml_str)
        
        self.input_file = input_file
        return input_file
    
    def _write_xyz(self, atoms: Any, filename: str):
        """Write ASE atoms to XYZ file."""
        from ase.io import write
        write(filename, atoms)
    
    def _cell_to_string(self, cell) -> str:
        """Convert cell to i-PI format string."""
        cell = np.array(cell)
        if cell.shape == (3,):
            return f"[{cell[0]}, 0, 0, 0, {cell[1]}, 0, 0, 0, {cell[2]}]"
        elif cell.shape == (3, 3):
            flat = cell.flatten()
            return f"[{', '.join(map(str, flat))}]"
        else:
            raise ValueError(f"Invalid cell shape: {cell.shape}")
    
    def _get_integrator(self) -> str:
        """Get integrator string for mode."""
        integrators = {
            IPIMode.PIMD: "nvt",
            IPIMode.RPMD: "nvt",
            IPIMode.CMD: "nvt",
            IPIMode.TRPMD: "nvt",
        }
        return integrators.get(self.config.mode, "nvt")
    
    def _prettify_xml(self, xml_str: str) -> str:
        """Add formatting to XML string."""
        # Simple formatting - add newlines after tags
        xml_str = xml_str.replace("><", ">\n<")
        lines = xml_str.split("\n")
        indent = 0
        formatted = []
        for line in lines:
            if line.strip():
                if line.startswith("</"):
                    indent -= 2
                formatted.append(" " * indent + line)
                if line.startswith("<") and not line.startswith("</") and "/>" not in line:
                    indent += 2
        return "<?xml version=\"1.0\"?>\n" + "\n".join(formatted)
    
    def run_ipi(self, wait: bool = True) -> subprocess.Popen:
        """Run i-PI simulation.
        
        Args:
            wait: Whether to wait for completion
            
        Returns:
            subprocess.Popen object
        """
        if self.input_file is None:
            raise ValueError("Input file not generated. Call generate_input_xml first.")
        
        cmd = ["i-pi", self.input_file]
        
        logger.info(f"Starting i-PI: {' '.join(cmd)}")
        
        with open(os.path.join(self.working_dir, "ipi.out"), "w") as out:
            with open(os.path.join(self.working_dir, "ipi.err"), "w") as err:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=out,
                    stderr=err,
                    cwd=self.working_dir
                )
        
        if wait:
            self.process.wait()
            logger.info(f"i-PI finished with return code {self.process.returncode}")
        
        return self.process
    
    def run_with_driver(self, driver_cmd: List[str], n_clients: int = 1):
        """Run i-PI with external driver.
        
        Args:
            driver_cmd: Command to run driver client
            n_clients: Number of client processes
        """
        # Start i-PI
        ipi_proc = self.run_ipi(wait=False)
        
        # Wait a bit for i-PI to start
        time.sleep(2)
        
        # Start drivers
        drivers = []
        for i in range(n_clients):
            logger.info(f"Starting driver {i+1}/{n_clients}")
            driver_proc = subprocess.Popen(
                driver_cmd,
                cwd=self.working_dir
            )
            drivers.append(driver_proc)
        
        # Wait for all
        ipi_proc.wait()
        for d in drivers:
            d.wait()
    
    def parse_output(self) -> PIMDResults:
        """Parse i-PI output files.
        
        Returns:
            PIMDResults object
        """
        results = PIMDResults()
        
        # Parse properties file
        props_file = os.path.join(self.working_dir, self.config.properties_file)
        if os.path.exists(props_file):
            props = self._parse_properties(props_file)
            results.potential_energy = props.get("potential")
            results.kinetic_energy = props.get("kinetic_md")
            results.total_energy = props.get("total_energy")
            results.temperature = props.get("temperature")
            results.pressure = props.get("pressure")
        
        # Parse trajectory
        traj_file = os.path.join(self.working_dir, self.config.trajectory_file)
        if os.path.exists(traj_file):
            results.centroid_positions = self._parse_trajectory(traj_file)
        
        results.output_files = {
            "properties": props_file if os.path.exists(props_file) else None,
            "trajectory": traj_file if os.path.exists(traj_file) else None,
            "working_dir": self.working_dir,
        }
        
        return results
    
    def _parse_properties(self, filename: str) -> Dict[str, np.ndarray]:
        """Parse i-PI properties output file."""
        data = {}
        with open(filename, "r") as f:
            # Read header
            header = f.readline().strip().split()
            
            # Read data
            rows = []
            for line in f:
                if line.strip() and not line.startswith("#"):
                    rows.append([float(x) for x in line.split()])
        
        if rows:
            arr = np.array(rows)
            for i, col in enumerate(header):
                data[col] = arr[:, i]
        
        return data
    
    def _parse_trajectory(self, filename: str) -> np.ndarray:
        """Parse XYZ trajectory file."""
        from ase.io import read
        
        frames = []
        try:
            atoms_list = read(filename, index=":")
            for atoms in atoms_list:
                frames.append(atoms.get_positions())
        except Exception as e:
            logger.warning(f"Failed to parse trajectory: {e}")
            return None
        
        return np.array(frames)


class PIMDSimulation(IPIInterface):
    """Path Integral Molecular Dynamics simulation.
    
    PIMD is the finite-temperature method of choice for computing
    exact quantum statistical mechanical properties of many-body systems.
    
    Key features:
    -------------
    - Exact quantum Boltzmann statistics via path integrals
    - Canonical sampling at finite temperature
    - Multiple energy estimators (primitive, virial, centroid virial)
    - Nuclear quantum effects (zero-point energy, tunneling)
    
    References:
    -----------
    - Tuckerman, M. E. (2010). Statistical Mechanics, Chapter 12
    - Parrinello & Rahman (1984). J. Chem. Phys. 80, 860
    - Marx & Parrinello (1996). J. Chem. Phys. 104, 4077
    """
    
    def __init__(self, config: IPIConfig, working_dir: Optional[str] = None):
        """Initialize PIMD simulation.
        
        Args:
            config: Configuration (mode will be set to PIMD)
            working_dir: Working directory
        """
        config.mode = IPIMode.PIMD
        super().__init__(config, working_dir)
    
    def run(self, atoms: Optional[Any] = None, 
            driver_cmd: Optional[List[str]] = None) -> PIMDResults:
        """Run PIMD simulation.
        
        Args:
            atoms: Initial atomic configuration
            driver_cmd: Driver command for forces (optional)
            
        Returns:
            PIMDResults
        """
        # Generate input
        self.generate_input_xml(atoms)
        
        # Run simulation
        if driver_cmd:
            self.run_with_driver(driver_cmd)
        else:
            self.run_ipi(wait=True)
        
        # Parse and return results
        return self.parse_output()
    
    def check_convergence(self, bead_numbers: List[int], atoms: Any,
                          driver_cmd: List[str]) -> Dict:
        """Check convergence with respect to number of beads.
        
        The Trotter theorem requires P -> infinity for exact results.
        In practice, convergence is achieved when:
        P * T >= (hbar * omega_max / k_B) for highest frequency mode
        
        Args:
            bead_numbers: List of bead numbers to test
            atoms: Atomic configuration
            driver_cmd: Driver command
            
        Returns:
            Convergence data
        """
        convergence = {
            "n_beads": [],
            "energies": [],
            "errors": [],
            "kinetic_energies": [],
        }
        
        for n_beads in bead_numbers:
            logger.info(f"Testing convergence with P={n_beads} beads")
            
            self.config.n_beads = n_beads
            results = self.run(atoms, driver_cmd)
            
            try:
                E, err = results.get_average_energy()
                KE, KE_err = results.get_quantum_kinetic_energy()
                
                convergence["n_beads"].append(n_beads)
                convergence["energies"].append(E)
                convergence["errors"].append(err)
                convergence["kinetic_energies"].append(KE)
            except ValueError as e:
                logger.warning(f"Failed to get energy for P={n_beads}: {e}")
        
        # Check if converged
        if len(convergence["energies"]) >= 2:
            de = np.abs(convergence["energies"][-1] - convergence["energies"][-2])
            convergence["converged"] = de < 1e-3  # 1 meV threshold
            convergence["energy_diff"] = de
        
        return convergence


class RPMDSimulation(IPIInterface):
    """Ring Polymer Molecular Dynamics simulation.
    
    RPMD is an approximate quantum dynamics method based on the
    isomorphism between quantum statistics and classical ring polymers.
    
    Key features:
    -------------
    - Real-time quantum dynamics approximation
    - Conserves quantum Boltzmann distribution
    - Approximates quantum time correlation functions
    - Nuclei move classically on effective potential from ring polymer
    
    Applications:
    -------------
    - Reaction rates (RPMD rate theory)
    - Quantum diffusion
    - Vibrational spectra
    - Proton transfer
    
    References:
    -----------
    - Craig & Manolopoulos (2004). J. Chem. Phys. 121, 3368
    - Habershon et al. (2013). Annu. Rev. Phys. Chem. 64, 387
    """
    
    def __init__(self, config: IPIConfig, working_dir: Optional[str] = None):
        """Initialize RPMD simulation.
        
        Args:
            config: Configuration (mode will be set to RPMD)
            working_dir: Working directory
        """
        config.mode = IPIMode.RPMD
        # For RPMD, centroid should have no thermostat (or very weak)
        config.centroid_friction = 0.0
        super().__init__(config, working_dir)
    
    def run(self, atoms: Optional[Any] = None,
            driver_cmd: Optional[List[str]] = None) -> PIMDResults:
        """Run RPMD simulation.
        
        Args:
            atoms: Initial atomic configuration
            driver_cmd: Driver command for forces (optional)
            
        Returns:
            PIMDResults
        """
        return super().run(atoms, driver_cmd)
    
    def calculate_velocity_autocorrelation(self, results: PIMDResults,
                                           max_lag: Optional[int] = None) -> np.ndarray:
        """Calculate velocity autocorrelation function.
        
        This is used for computing vibrational spectra via Fourier transform.
        
        Args:
            results: PIMD results
            max_lag: Maximum time lag for correlation
            
        Returns:
            VACF array
        """
        if results.velocities is None:
            raise ValueError("Velocities not available")
        
        v = results.velocities
        n_frames = v.shape[0]
        
        if max_lag is None:
            max_lag = n_frames // 2
        
        # Compute VACF for centroid velocities
        centroid_v = np.mean(v, axis=1)  # [n_frames, n_atoms, 3]
        
        vacf = np.zeros(max_lag)
        for dt in range(max_lag):
            for t in range(n_frames - dt):
                vacf[dt] += np.sum(centroid_v[t] * centroid_v[t + dt])
        
        # Normalize
        vacf /= vacf[0]
        
        return vacf
    
    def compute_ir_spectrum(self, results: PIMDResults, 
                            dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute IR spectrum from dipole autocorrelation.
        
        Note: Requires dipole moments to be stored in results.
        
        Args:
            results: PIMD results
            dt: Time step in femtoseconds
            
        Returns:
            (frequencies_cm1, intensities)
        """
        # This is a placeholder - actual implementation requires dipole data
        # from the simulation
        raise NotImplementedError("IR spectrum calculation requires dipole data")


class TRPMDSimulation(IPIInterface):
    """Thermostatted Ring Polymer MD.
    
    TRPMD is a modification of RPMD that includes thermostatting
    of internal modes to improve convergence while maintaining
    the correct quantum statistics.
    
    Advantages over RPMD:
    --------------------
    - Better sampling of configuration space
    - Reduced ring polymer fluctuation artifacts
    - More stable for long simulations
    - Better zero-point energy conservation
    
    References:
    -----------
    - Rossi et al. (2014). J. Chem. Phys. 141, 181101
    """
    
    def __init__(self, config: IPIConfig, working_dir: Optional[str] = None):
        """Initialize TRPMD simulation."""
        config.mode = IPIMode.TRPMD
        config.thermostat = ThermostatType.PILE_L
        super().__init__(config, working_dir)


class PIConvergenceChecker:
    """Check convergence of path integral calculations.
    
    Provides utilities to determine adequate number of beads and
    verify convergence of thermodynamic properties.
    """
    
    def __init__(self, max_beads: int = 128, min_beads: int = 4,
                 threshold: float = 1e-4):
        """Initialize convergence checker.
        
        Args:
            max_beads: Maximum number of beads to test
            min_beads: Minimum number of beads
            threshold: Convergence threshold for relative energy change
        """
        self.max_beads = max_beads
        self.min_beads = min_beads
        self.threshold = threshold
    
    def estimate_beads(self, temperature: float, 
                       highest_frequency_cm1: float) -> int:
        """Estimate required number of beads.
        
        Based on the criterion: P >= hbar * omega_max / (k_B * T)
        
        Args:
            temperature: Temperature in Kelvin
            highest_frequency_cm1: Highest vibrational frequency in cm^-1
            
        Returns:
            Estimated number of beads
        """
        # Constants
        hbar = 1.054571817e-34  # J*s
        k_B = 1.380649e-23      # J/K
        c = 2.99792458e10       # cm/s
        
        omega = 2 * np.pi * c * highest_frequency_cm1  # rad/s
        P_min = hbar * omega / (k_B * temperature)
        
        # Round up to power of 2 for efficiency
        P = int(2**np.ceil(np.log2(P_min)))
        P = max(P, self.min_beads)
        P = min(P, self.max_beads)
        
        return P
    
    def check_convergence(self, bead_numbers: List[int],
                          energies: List[float],
                          errors: Optional[List[float]] = None) -> Dict:
        """Check convergence from bead number study.
        
        Args:
            bead_numbers: List of bead numbers tested
            energies: Corresponding energies
            errors: Statistical errors (optional)
            
        Returns:
            Convergence analysis
        """
        bead_numbers = np.array(bead_numbers)
        energies = np.array(energies)
        
        # Check monotonic convergence (typically decreasing)
        diffs = np.diff(energies)
        
        analysis = {
            "n_beads": bead_numbers.tolist(),
            "energies": energies.tolist(),
            "energy_differences": diffs.tolist(),
            "converged": False,
            "recommended_beads": None,
            "extrapolated_energy": None,
        }
        
        # Check if converged
        if len(energies) >= 2:
            rel_diff = np.abs(diffs[-1] / energies[-1])
            analysis["converged"] = rel_diff < self.threshold
            analysis["recommended_beads"] = int(bead_numbers[-1])
            
            # Richardson extrapolation for infinite P limit
            if len(energies) >= 3:
                # E(P) ≈ E_inf + a/P^2
                # Use last two points to estimate E_inf
                P1, P2 = bead_numbers[-2], bead_numbers[-1]
                E1, E2 = energies[-2], energies[-1]
                
                if P1 != P2:
                    a = (E1 - E2) / (1.0/P1**2 - 1.0/P2**2)
                    E_inf = E2 - a / P2**2
                    analysis["extrapolated_energy"] = float(E_inf)
        
        return analysis
    
    def richardson_extrapolation(self, bead_numbers: np.ndarray,
                                  energies: np.ndarray) -> float:
        """Perform Richardson extrapolation to P -> infinity limit.
        
        The convergence is expected to be O(1/P^2) for primitive estimator.
        
        Args:
            bead_numbers: Array of bead numbers
            energies: Corresponding energies
            
        Returns:
            Extrapolated energy at infinite beads
        """
        # Fit E(P) = E_inf + a/P^2
        x = 1.0 / bead_numbers**2
        
        # Linear regression
        A = np.vstack([x, np.ones(len(x))]).T
        a, E_inf = np.linalg.lstsq(A, energies, rcond=None)[0]
        
        return E_inf


def generate_input_xml(config: IPIConfig, atoms: Optional[Any] = None) -> str:
    """Generate i-PI input XML string.
    
    Args:
        config: IPIConfig object
        atoms: ASE atoms object (optional)
        
    Returns:
        XML string
    """
    interface = IPIInterface(config)
    xml_file = interface.generate_input_xml(atoms)
    
    with open(xml_file, "r") as f:
        return f.read()


def run_pimd(n_beads: int = 32, temperature: float = 300.0,
             timestep: float = 0.5, n_steps: int = 10000,
             atoms: Optional[Any] = None,
             driver_cmd: Optional[List[str]] = None,
             working_dir: Optional[str] = None) -> PIMDResults:
    """Run a PIMD simulation with simplified interface.
    
    Args:
        n_beads: Number of path integral beads
        temperature: Temperature in Kelvin
        timestep: Time step in femtoseconds
        n_steps: Number of steps
        atoms: Initial atomic configuration
        driver_cmd: Driver command for external code
        working_dir: Working directory
        
    Returns:
        PIMDResults object
    """
    config = IPIConfig(
        n_beads=n_beads,
        temperature=temperature,
        timestep=timestep,
        n_steps=n_steps,
        mode=IPIMode.PIMD
    )
    
    sim = PIMDSimulation(config, working_dir)
    return sim.run(atoms, driver_cmd)


def run_rpmd(n_beads: int = 32, temperature: float = 300.0,
             timestep: float = 0.5, n_steps: int = 10000,
             atoms: Optional[Any] = None,
             driver_cmd: Optional[List[str]] = None,
             working_dir: Optional[str] = None) -> PIMDResults:
    """Run an RPMD simulation with simplified interface.
    
    Args:
        n_beads: Number of path integral beads
        temperature: Temperature in Kelvin
        timestep: Time step in femtoseconds
        n_steps: Number of steps
        atoms: Initial atomic configuration
        driver_cmd: Driver command for external code
        working_dir: Working directory
        
    Returns:
        PIMDResults object
    """
    config = IPIConfig(
        n_beads=n_beads,
        temperature=temperature,
        timestep=timestep,
        n_steps=n_steps,
        mode=IPIMode.RPMD
    )
    
    sim = RPMDSimulation(config, working_dir)
    return sim.run(atoms, driver_cmd)


def estimate_required_beads(temperature: float, 
                            highest_frequency_cm1: float,
                            safety_factor: float = 2.0) -> int:
    """Estimate required number of beads for convergence.
    
    Args:
        temperature: Temperature in Kelvin
        highest_frequency_cm1: Highest vibrational frequency in cm^-1
        safety_factor: Safety factor for bead number
        
    Returns:
        Estimated number of beads
    """
    checker = PIConvergenceChecker()
    P = checker.estimate_beads(temperature, highest_frequency_cm1)
    return int(P * safety_factor)
