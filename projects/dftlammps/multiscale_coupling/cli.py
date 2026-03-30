#!/usr/bin/env python3
"""
Command-line interface for multiscale coupling module.
"""
import argparse
import sys
import json
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dftlammps.multiscale_coupling import (
    VASPLAMMPSCoupling, CrossScaleValidator
)
from dftlammps.multiscale_coupling.utils import load_config


def run_qmmm(args):
    """Run QM/MM calculation."""
    print("Running QM/MM calculation...")
    
    config = load_config(args.config) if args.config else {}
    
    qmmm = VASPLAMMPSCoupling(
        vasp_cmd=args.vasp or 'vasp_std',
        lammps_cmd=args.lammps or 'lmp',
        work_dir=args.work_dir or './qmmm_run'
    )
    
    print(f"Configuration loaded from: {args.config}")
    print(f"Working directory: {args.work_dir}")
    print("\nNote: This is a demonstration. Full implementation requires")
    print("proper structure files and simulation setup.")


def run_validate(args):
    """Run validation checks."""
    print("Running validation checks...")
    
    validator = CrossScaleValidator(tolerance=args.tolerance)
    
    # Load data
    print(f"Loading data from: {args.data}")
    print("\nValidation checks would be performed here.")


def run_cg(args):
    """Run coarse-graining."""
    print("Running coarse-graining...")
    
    from dftlammps.multiscale_coupling import CoarseGrainer
    
    print(f"Input trajectory: {args.trajectory}")
    print(f"Number of beads: {args.n_beads}")
    print("\nCoarse-graining would be performed here.")


def main():
    parser = argparse.ArgumentParser(
        description='DFTLAMMPS Multiscale Coupling CLI'
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # QM/MM command
    qmmm_parser = subparsers.add_parser('qmmm', help='Run QM/MM calculation')
    qmmm_parser.add_argument('-c', '--config', help='Configuration file')
    qmmm_parser.add_argument('--vasp', help='VASP command')
    qmmm_parser.add_argument('--lammps', help='LAMMPS command')
    qmmm_parser.add_argument('-w', '--work-dir', help='Working directory')
    qmmm_parser.add_argument('-s', '--structure', help='Structure file')
    
    # Validation command
    val_parser = subparsers.add_parser('validate', help='Run validation')
    val_parser.add_argument('-d', '--data', required=True, help='Data file')
    val_parser.add_argument('-t', '--tolerance', type=float, default=1e-3,
                           help='Validation tolerance')
    
    # Coarse-graining command
    cg_parser = subparsers.add_parser('cg', help='Run coarse-graining')
    cg_parser.add_argument('-t', '--trajectory', required=True,
                          help='Input trajectory')
    cg_parser.add_argument('-n', '--n-beads', type=int, required=True,
                          help='Number of CG beads')
    cg_parser.add_argument('-o', '--output', help='Output file')
    
    args = parser.parse_args()
    
    if args.command == 'qmmm':
        run_qmmm(args)
    elif args.command == 'validate':
        run_validate(args)
    elif args.command == 'cg':
        run_cg(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
