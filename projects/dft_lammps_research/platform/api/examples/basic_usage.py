#!/usr/bin/env python3
"""
Example: Basic API Usage

Demonstrates basic API operations using the Python SDK.
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path for SDK import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sdks.python.dft_lammps import Client


def main():
    # Initialize client
    api_key = os.getenv("DFT_LAMMPS_API_KEY", "demo-api-key")
    client = Client(
        api_key=api_key,
        base_url="http://localhost:8080"
    )
    
    print("=" * 60)
    print("DFT+LAMMPS API Platform - Basic Usage Example")
    print("=" * 60)
    
    # 1. Check API health
    print("\n1. Checking API health...")
    health = client.health()
    print(f"   Status: {health.get('status', 'unknown')}")
    print(f"   Version: {health.get('version', 'unknown')}")
    
    # 2. Get usage stats
    print("\n2. Getting usage statistics...")
    usage = client.usage()
    print(f"   Tier: {usage.get('tier', 'unknown')}")
    print(f"   Requests today: {usage.get('requests', {}).get('total', 0)}")
    
    # 3. Create a project
    print("\n3. Creating a new project...")
    project = client.projects.create(
        name=f"Example Project {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        description="Demo project created via API example",
        project_type="battery_screening",
        target_properties={
            "band_gap": {"min": 0.5, "max": 2.0},
            "ionic_conductivity": {"min": 1e-4}
        },
        material_system="Li-S",
        tags=["demo", "example", "api"]
    )
    print(f"   Created project: {project.id}")
    print(f"   Name: {project.name}")
    print(f"   Type: {project.project_type}")
    
    # 4. List projects
    print("\n4. Listing projects...")
    projects = client.projects.list(page=1, page_size=5)
    print(f"   Total projects: {projects.get('total', 0)}")
    for p in projects.get('items', []):
        print(f"   - {p.id}: {p.name} ({p.status})")
    
    # 5. Update project
    print("\n5. Updating project...")
    updated = client.projects.update(
        project.id,
        description="Updated description via API"
    )
    print(f"   Updated: {updated.updated_at}")
    
    # 6. Submit a calculation
    print("\n6. Submitting a calculation...")
    
    # Example structure data (Li2S)
    structure = {
        "species": ["Li", "Li", "S"],
        "positions": [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25]
        ],
        "cell": [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ]
    }
    
    calculation = client.calculations.submit(
        project_id=project.id,
        structure=structure,
        calculation_type="dft",
        parameters={
            "ecut": 500,
            "kpoints": "4 4 4",
            "functional": "PBE"
        },
        priority=8
    )
    print(f"   Calculation ID: {calculation.id}")
    print(f"   Status: {calculation.status}")
    print(f"   Type: {calculation.calculation_type}")
    
    # 7. Get calculation status
    print("\n7. Checking calculation status...")
    calc_status = client.calculations.get(calculation.id)
    print(f"   Current status: {calc_status.status}")
    
    # 8. Submit batch calculation
    print("\n8. Submitting batch calculations...")
    batch_calcs = [
        {
            "structure": structure,
            "calculation_type": "dft",
            "parameters": {"ecut": 400},
            "priority": 5
        },
        {
            "structure": {
                "species": ["Li", "P", "S"],
                "positions": [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]],
                "cell": [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
            },
            "calculation_type": "dft",
            "parameters": {"ecut": 500},
            "priority": 5
        }
    ]
    
    batch = client.calculations.submit_batch(
        project_id=project.id,
        calculations=batch_calcs
    )
    print(f"   Batch ID: {batch.get('batch_id')}")
    print(f"   Total submitted: {batch.get('total')}")
    print(f"   Calculation IDs: {batch.get('calculation_ids', [])}")
    
    # 9. Create webhook subscription
    print("\n9. Creating webhook subscription...")
    try:
        webhook = client.webhooks.subscribe(
            url="https://example.com/webhooks/dft-lammps",
            events=["calculation.completed", "calculation.failed"],
            metadata={"source": "example_script"}
        )
        print(f"   Webhook ID: {webhook.webhook_id}")
        print(f"   Secret: {webhook.secret}")
        print(f"   Events: {webhook.events}")
    except Exception as e:
        print(f"   Note: Webhook creation failed (expected in demo): {e}")
    
    # 10. List webhooks
    print("\n10. Listing webhook subscriptions...")
    webhooks = client.webhooks.list()
    print(f"   Total webhooks: {len(webhooks)}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print(f"\nProject ID for reference: {project.id}")
    print(f"Calculation ID for reference: {calculation.id}")


if __name__ == "__main__":
    main()
