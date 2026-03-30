"""
QM/MM Interface Module

Provides interfaces for Quantum Mechanics / Molecular Mechanics coupling.
Supports VASP for QM and LAMMPS for MM.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import subprocess
import os


class QMEngine(ABC):
    """Abstract base class for QM engines."""
    
    @abstractmethod
    def calculate(self, positions: np.ndarray, 
                  elements: List[str],
                  charge: int = 0,
                  multiplicity: int = 1) -> Dict:
        """
        Perform QM calculation.
        
        Args:
            positions: (N, 3) atomic positions in Angstrom
            elements: List of element symbols
            charge: Total charge
            multiplicity: Spin multiplicity
            
        Returns:
            Dictionary with 'energy', 'forces', 'charges'
        """
        pass
    
    @abstractmethod
    def set_options(self, **kwargs):
        """Set calculation options."""
        pass


class MMEngine(ABC):
    """Abstract base class for MM engines."""
    
    @abstractmethod
    def calculate(self, positions: np.ndarray,
                  atom_types: List[str]) -> Dict:
        """
        Perform MM calculation.
        
        Args:
            positions: (N, 3) atomic positions in Angstrom
            atom_types: List of atom types
            
        Returns:
            Dictionary with 'energy', 'forces'
        """
        pass
    
    @abstractmethod
    def set_force_field(self, force_field_file: str):
        """Load force field parameters."""
        pass


class VASPEngine(QMEngine):
    """VASP QM engine wrapper."""
    
    def __init__(self, vasp_cmd: str = 'vasp', 
                 work_dir: str = './vasp_run'):
        """
        Initialize VASP engine.
        
        Args:
            vasp_cmd: VASP executable command
            work_dir: Working directory for VASP calculations
        """
        self.vasp_cmd = vasp_cmd
        self.work_dir = work_dir
        self.options = {
            'encut': 400,
            'kpts': [1, 1, 1],
            'xc': 'PBE',
            'isym': 0,
            'ispin': 1
        }
        os.makedirs(work_dir, exist_ok=True)
        
    def set_options(self, **kwargs):
        """Set VASP calculation options."""
        self.options.update(kwargs)
        
    def _write_poscar(self, positions: np.ndarray, elements: List[str]):
        """Write POSCAR file."""
        unique_elements = []
        element_counts = []
        for elem in sorted(set(elements)):
            unique_elements.append(elem)
            element_counts.append(elements.count(elem))
        
        with open(os.path.join(self.work_dir, 'POSCAR'), 'w') as f:
            f.write('QM Region\n')
            f.write('1.0\n')
            # Cell - use large enough box
            f.write('20.0 0.0 0.0\n')
            f.write('0.0 20.0 0.0\n')
            f.write('0.0 0.0 20.0\n')
            f.write(' '.join(unique_elements) + '\n')
            f.write(' '.join(map(str, element_counts)) + '\n')
            f.write('Cartesian\n')
            for pos in positions:
                f.write(f'{pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}\n')
    
    def _write_incar(self, charge: int, multiplicity: int):
        """Write INCAR file."""
        with open(os.path.join(self.work_dir, 'INCAR'), 'w') as f:
            f.write(f'ENCUT = {self.options["encut"]}\n')
            f.write(f'ISYM = {self.options["isym"]}\n')
            f.write(f'ISPIN = {self.options["ispin"]}\n')
            f.write('EDIFF = 1E-6\n')
            f.write('NELM = 100\n')
            f.write('IBRION = -1\n')  # Single point
            f.write('NSW = 0\n')
            f.write(f'NELECT = {self._get_electron_count() - charge}\n')
            
    def _get_electron_count(self) -> int:
        """Get default electron count (should be overridden)."""
        return 0  # Placeholder
        
    def _write_kpoints(self):
        """Write KPOINTS file."""
        kpts = self.options['kpts']
        with open(os.path.join(self.work_dir, 'KPOINTS'), 'w') as f:
            f.write('Automatic mesh\n')
            f.write('0\n')
            f.write('Gamma\n')
            f.write(f'{kpts[0]} {kpts[1]} {kpts[2]}\n')
            f.write('0 0 0\n')
    
    def _read_vasp_out(self) -> Dict:
        """Read VASP output."""
        outcar_path = os.path.join(self.work_dir, 'OUTCAR')
        vasprun_path = os.path.join(self.work_dir, 'vasprun.xml')
        
        energy = None
        forces = None
        
        # Try to read from OUTCAR
        if os.path.exists(outcar_path):
            with open(outcar_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if 'free  energy   TOTEN' in line:
                        energy = float(line.split()[-2])
                    if 'TOTAL-FORCE' in line:
                        forces = []
                        for j in range(i+2, len(lines)):
                            parts = lines[j].split()
                            if len(parts) < 6:
                                break
                            try:
                                forces.append([float(parts[3]), 
                                              float(parts[4]), 
                                              float(parts[5])])
                            except ValueError:
                                break
        
        return {
            'energy': energy,
            'forces': np.array(forces) if forces else None,
            'charges': None  # Would need to parse CHGCAR
        }
    
    def calculate(self, positions: np.ndarray,
                  elements: List[str],
                  charge: int = 0,
                  multiplicity: int = 1) -> Dict:
        """Run VASP calculation."""
        self._write_poscar(positions, elements)
        self._write_incar(charge, multiplicity)
        self._write_kpoints()
        
        # Run VASP
        try:
            result = subprocess.run(
                self.vasp_cmd,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=3600
            )
        except subprocess.TimeoutExpired:
            return {'energy': None, 'forces': None, 'charges': None}
        
        return self._read_vasp_out()


class LAMMPSEngine(MMEngine):
    """LAMMPS MM engine wrapper."""
    
    def __init__(self, lammps_cmd: str = 'lammps',
                 work_dir: str = './lammps_run'):
        """
        Initialize LAMMPS engine.
        
        Args:
            lammps_cmd: LAMMPS executable command
            work_dir: Working directory for LAMMPS calculations
        """
        self.lammps_cmd = lammps_cmd
        self.work_dir = work_dir
        self.force_field_file = None
        self.input_file = None
        os.makedirs(work_dir, exist_ok=True)
        
    def set_force_field(self, force_field_file: str):
        """Load force field file."""
        self.force_field_file = force_field_file
        
    def set_input_file(self, input_file: str):
        """Set LAMMPS input file."""
        self.input_file = input_file
        
    def _write_data_file(self, positions: np.ndarray, atom_types: List[str]):
        """Write LAMMPS data file."""
        unique_types = sorted(set(atom_types))
        type_map = {t: i+1 for i, t in enumerate(unique_types)}
        
        with open(os.path.join(self.work_dir, 'data.qmmm'), 'w') as f:
            f.write('LAMMPS data file for QM/MM\n\n')
            f.write(f'{len(positions)} atoms\n')
            f.write(f'{len(unique_types)} atom types\n\n')
            f.write('-50.0 50.0 xlo xhi\n')
            f.write('-50.0 50.0 ylo yhi\n')
            f.write('-50.0 50.0 zlo zhi\n\n')
            f.write('Masses\n\n')
            for i, t in enumerate(unique_types, 1):
                mass = self._get_mass(t)
                f.write(f'{i} {mass}\n')
            f.write('\nAtoms\n\n')
            for i, (pos, atype) in enumerate(zip(positions, atom_types), 1):
                f.write(f'{i} 1 {type_map[atype]} {pos[0]} {pos[1]} {pos[2]}\n')
    
    def _get_mass(self, atom_type: str) -> float:
        """Get atomic mass for atom type."""
        masses = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'P': 30.974, 'S': 32.065, 'Fe': 55.845
        }
        return masses.get(atom_type, 12.0)
        
    def _write_input_file(self):
        """Write LAMMPS input file."""
        with open(os.path.join(self.work_dir, 'in.qmmm'), 'w') as f:
            f.write('# LAMMPS input for QM/MM\n')
            f.write('units real\n')
            f.write('atom_style charge\n')
            f.write('read_data data.qmmm\n')
            if self.force_field_file:
                f.write(f'include {self.force_field_file}\n')
            f.write('run 0\n')
    
    def calculate(self, positions: np.ndarray,
                  atom_types: List[str]) -> Dict:
        """Run LAMMPS calculation."""
        self._write_data_file(positions, atom_types)
        self._write_input_file()
        
        try:
            result = subprocess.run(
                [self.lammps_cmd, '-in', 'in.qmmm'],
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
        except subprocess.TimeoutExpired:
            return {'energy': None, 'forces': None}
        
        # Parse output
        return self._parse_output(result.stdout)
    
    def _parse_output(self, output: str) -> Dict:
        """Parse LAMMPS output."""
        energy = None
        for line in output.split('\n'):
            if 'TotEng' in line or 'Total energy' in line:
                try:
                    energy = float(line.split()[-1])
                except (ValueError, IndexError):
                    pass
        return {'energy': energy, 'forces': None}


class QMMMInterface:
    """
    Main QM/MM interface class.
    
    Implements additive QM/MM coupling scheme:
    E_total = E_QM(QM) + E_MM(MM) + E_QMMM_coupling
    """
    
    def __init__(self, 
                 qm_engine: QMEngine,
                 mm_engine: MMEngine,
                 embedding: str = 'electrostatic'):
        """
        Initialize QM/MM interface.
        
        Args:
            qm_engine: QM engine instance
            mm_engine: MM engine instance
            embedding: Embedding scheme ('mechanical' or 'electrostatic')
        """
        self.qm_engine = qm_engine
        self.mm_engine = mm_engine
        self.embedding = embedding
        self.qm_atoms = None
        self.mm_atoms = None
        self.link_atoms = None
        
    def set_regions(self, qm_mask: np.ndarray, mm_mask: np.ndarray):
        """
        Define QM and MM regions.
        
        Args:
            qm_mask: Boolean mask for QM atoms
            mm_mask: Boolean mask for MM atoms
        """
        self.qm_atoms = qm_mask
        self.mm_atoms = mm_mask
        
    def calculate(self, 
                  positions: np.ndarray,
                  qm_elements: List[str],
                  mm_types: List[str],
                  mm_charges: Optional[np.ndarray] = None) -> Dict:
        """
        Perform QM/MM calculation.
        
        Args:
            positions: (N, 3) atomic positions
            qm_elements: Element symbols for all atoms (QM atoms use these)
            mm_types: Atom types for MM region
            mm_charges: Charges on MM atoms for electrostatic embedding
            
        Returns:
            Dictionary with total energy and forces
        """
        # Extract QM positions
        qm_positions = positions[self.qm_atoms]
        qm_elems = [qm_elements[i] for i in range(len(qm_elements)) 
                    if self.qm_atoms[i]]
        
        # QM calculation
        qm_result = self.qm_engine.calculate(qm_positions, qm_elems)
        
        # MM calculation
        mm_positions = positions[self.mm_atoms]
        mm_atom_types = [mm_types[i] for i in range(len(mm_types))
                        if self.mm_atoms[i]]
        mm_result = self.mm_engine.calculate(mm_positions, mm_atom_types)
        
        # QM/MM coupling
        coupling_energy = self._calculate_coupling(
            positions, qm_elems, mm_charges
        )
        
        # Total energy
        total_energy = (qm_result.get('energy', 0) + 
                       mm_result.get('energy', 0) +
                       coupling_energy)
        
        return {
            'total_energy': total_energy,
            'qm_energy': qm_result.get('energy'),
            'mm_energy': mm_result.get('energy'),
            'coupling_energy': coupling_energy,
            'qm_forces': qm_result.get('forces'),
            'mm_forces': mm_result.get('forces')
        }
    
    def _calculate_coupling(self, 
                           positions: np.ndarray,
                           qm_elements: List[str],
                           mm_charges: Optional[np.ndarray]) -> float:
        """
        Calculate QM/MM coupling energy.
        
        Args:
            positions: Atomic positions
            qm_elements: QM element types
            mm_charges: MM atom charges
            
        Returns:
            Coupling energy
        """
        if self.embedding == 'mechanical' or mm_charges is None:
            return 0.0
        
        # Electrostatic embedding
        qm_positions = positions[self.qm_atoms]
        mm_positions = positions[self.mm_atoms]
        
        # Simple point charge - QM charge interaction
        coupling = 0.0
        for i, qm_pos in enumerate(qm_positions):
            # Get QM charge (simplified - should come from QM calculation)
            qm_charge = self._get_qm_charge(qm_elements[i])
            for j, mm_pos in enumerate(mm_positions):
                r = np.linalg.norm(qm_pos - mm_pos)
                if r > 0.1:  # Avoid division by zero
                    coupling += qm_charge * mm_charges[j] / r
        
        return coupling * 14.3996  # Convert to appropriate units
    
    def _get_qm_charge(self, element: str) -> float:
        """Get approximate charge for element."""
        charges = {'H': 0.4, 'C': 0.0, 'N': -0.3, 'O': -0.5}
        return charges.get(element, 0.0)


class VASPLAMMPSCoupling(QMMMInterface):
    """Convenience class for VASP-LAMMPS coupling."""
    
    def __init__(self, 
                 vasp_cmd: str = 'vasp',
                 lammps_cmd: str = 'lammps',
                 work_dir: str = './qmmm_run',
                 embedding: str = 'electrostatic'):
        """
        Initialize VASP-LAMMPS coupling.
        
        Args:
            vasp_cmd: VASP command
            lammps_cmd: LAMMPS command
            work_dir: Working directory
            embedding: Embedding scheme
        """
        qm_engine = VASPEngine(vasp_cmd, os.path.join(work_dir, 'qm'))
        mm_engine = LAMMPSEngine(lammps_cmd, os.path.join(work_dir, 'mm'))
        
        super().__init__(qm_engine, mm_engine, embedding)
