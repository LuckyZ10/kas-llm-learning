#!/usr/bin/env python3
"""
Example: Batch Screening Workflow

Demonstrates a high-throughput screening workflow using the API.
"""

import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sdks.python.dft_lammps import Client


def generate_candidate_structures(count: int = 10) -> List[Dict[str, Any]]:
    """Generate candidate structures for screening"""
    structures = []
    
    # Example: Generate Li-S structures with varying compositions
    for i in range(count):
        # Simple cubic structures with varying lattice constants
        lattice_const = 5.0 + (i * 0.1)
        
        structure = {
            "name": f"LiS_candidate_{i}",
            "species": ["Li", "Li", "S"],
            "positions": [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.25, 0.25, 0.25]
            ],
            "cell": [
                [lattice_const, 0.0, 0.0],
                [0.0, lattice_const, 0.0],
                [0.0, 0.0, lattice_const]
            ],
            "metadata": {
                "candidate_id": i,
                "lattice_constant": lattice_const
            }
        }
        structures.append(structure)
    
    return structures


def screen_by_formation_energy(results: List[Dict], threshold: float = -1.0) -> List[Dict]:
    """Filter structures by formation energy"""
    screened = []
    for result in results:
        if result.get("status") == "completed":
            formation_energy = result.get("results", {}).get("formation_energy_per_atom")
            if formation_energy is not None and formation_energy < threshold:
                screened.append(result)
    return screened


def main():
    # Initialize client
    api_key = os.getenv("DFT_LAMMPS_API_KEY", "demo-api-key")
    client = Client(api_key=api_key, base_url="http://localhost:8080")
    
    print("=" * 60)
    print("High-Throughput Screening Workflow Example")
    print("=" * 60)
    
    # 1. Create screening project
    print("\n1. Creating screening project...")
    project = client.projects.create(
        name="Li-S Battery Candidates Screening",
        description="High-throughput screening of Li-S structures",
        project_type="battery_screening",
        target_properties={
            "band_gap": {"min": 0.1, "max": 3.0},
            "formation_energy": {"max": -0.5}
        },
        material_system="Li-S",
        tags=["screening", "high-throughput", "battery"]
    )
    print(f"   Project: {project.id}")
    
    # 2. Generate candidate structures
    print("\n2. Generating candidate structures...")
    candidates = generate_candidate_structures(count=20)
    print(f"   Generated {len(candidates)} candidates")
    
    # 3. Prepare batch submissions
    print("\n3. Preparing batch submissions...")
    calculations = []
    for candidate in candidates:
        calc = {
            "structure": {
                "species": candidate["species"],
                "positions": candidate["positions"],
                "cell": candidate["cell"]
            },
            "calculation_type": "dft",
            "parameters": {
                "ecut": 500,
                "kpoints": "4 4 4",
                "functional": "PBE",
                "properties": ["energy", "forces", "stress", "band_gap"]
            },
            "priority": 5
        }
        calculations.append(calc)
    
    # 4. Submit in batches
    print("\n4. Submitting calculations in batches...")
    batch_size = 10
    all_calculation_ids = []
    
    for i in range(0, len(calculations), batch_size):
        batch = calculations[i:i + batch_size]
        print(f"   Submitting batch {i//batch_size + 1} ({len(batch)} calculations)...")
        
        result = client.calculations.submit_batch(
            project_id=project.id,
            calculations=batch
        )
        
        all_calculation_ids.extend(result.get("calculation_ids", []))
        print(f"   Batch submitted: {result.get('batch_id')}")
    
    print(f"\n   Total calculations submitted: {len(all_calculation_ids)}")
    
    # 5. Set up webhook for completion notifications
    print("\n5. Setting up webhook for notifications...")
    try:
        webhook = client.webhooks.subscribe(
            url="https://example.com/webhooks/batch-complete",
            events=["batch.completed", "calculation.completed"],
            metadata={
                "project_id": project.id,
                "workflow": "screening"
            }
        )
        print(f"   Webhook created: {webhook.webhook_id}")
    except Exception as e:
        print(f"   Note: Webhook setup failed: {e}")
    
    # 6. Poll for results (in production, use webhooks)
    print("\n6. Monitoring calculations (demo - would use webhooks in production)...")
    print("   This would typically wait for webhook notifications")
    
    # Simulate getting results
    print("   Simulating result retrieval...")
    results = []
    for calc_id in all_calculation_ids[:5]:  # Just check first 5
        calc = client.calculations.get(calc_id)
        results.append({
            "id": calc_id,
            "status": calc.status,
            "results": calc.results
        })
    
    # 7. Apply screening criteria
    print("\n7. Applying screening criteria...")
    
    # Simulate some completed results
    simulated_results = [
        {"id": "calc_1", "status": "completed", "results": {"formation_energy_per_atom": -1.5, "band_gap": 1.2}},
        {"id": "calc_2", "status": "completed", "results": {"formation_energy_per_atom": -0.8, "band_gap": 0.5}},
        {"id": "calc_3", "status": "completed", "results": {"formation_energy_per_atom": -0.3, "band_gap": 2.1}},  # Rejected
        {"id": "calc_4", "status": "completed", "results": {"formation_energy_per_atom": -2.0, "band_gap": 1.8}},
        {"id": "calc_5", "status": "failed", "results": None},
    ]
    
    passed = screen_by_formation_energy(simulated_results, threshold=-0.5)
    
    print(f"   Total completed: {len([r for r in simulated_results if r['status'] == 'completed'])}")
    print(f"   Passed screening: {len(passed)}")
    print(f"   Failed: {len([r for r in simulated_results if r['status'] == 'failed'])}")
    
    # 8. Export results
    print("\n8. Exporting screening results...")
    print("   Top candidates:")
    for candidate in sorted(passed, key=lambda x: x["results"].get("formation_energy_per_atom", 0))[:3]:
        print(f"   - {candidate['id']}: E_form = {candidate['results']['formation_energy_per_atom']:.3f} eV/atom")
    
    # 9. Update project status
    print("\n9. Updating project status...")
    client.projects.update(
        project.id,
        status="screening_complete",
        target_properties={
            **project.target_properties,
            "screening_summary": {
                "total_candidates": len(candidates),
                "passed": len(passed),
                "failed": len(simulated_results) - len(passed)
            }
        }
    )
    
    # 10. Generate report
    print("\n10. Generating screening report...")
    report = {
        "project_id": project.id,
        "screening_date": "2024-01-15T10:00:00Z",
        "total_candidates": len(candidates),
        "completed_calculations": len(simulated_results),
        "passed_screening": len(passed),
        "top_candidates": [
            {
                "id": c["id"],
                "formation_energy": c["results"]["formation_energy_per_atom"],
                "band_gap": c["results"]["band_gap"]
            }
            for c in sorted(passed, key=lambda x: x["results"].get("formation_energy_per_atom", 0))[:5]
        ]
    }
    
    print("\n" + "=" * 60)
    print("Screening Report")
    print("=" * 60)
    print(json.dumps(report, indent=2))
    
    print("\n" + "=" * 60)
    print("Screening workflow completed!")
    print("=" * 60)
    print(f"\nProject ID: {project.id}")
    print(f"View results at: http://localhost:8080/portal")


if __name__ == "__main__":
    main()
