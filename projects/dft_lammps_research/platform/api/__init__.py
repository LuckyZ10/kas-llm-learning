"""
DFT+LAMMPS API Platform

A production-grade API platform for materials research and molecular dynamics simulations.

Modules:
    gateway: API Gateway with authentication, rate limiting, and request routing
    auth: OAuth2 and API Key authentication
    webhooks: Event subscription and notification system
    portal: Developer portal with interactive documentation
    sdks: Client SDKs for Python, JavaScript, and Go
    integrations: Third-party integrations (Jupyter, VS Code, databases, workflows)

Quick Start:
    # Start the API Gateway
    from api_platform.gateway.main import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

    # Use the Python SDK
    from dft_lammps import Client
    client = Client(api_key="your-api-key")
    project = client.projects.create(name="My Project")

Documentation:
    - API Docs: http://localhost:8080/docs
    - Developer Portal: http://localhost:8080/portal
"""

__version__ = "1.0.0"
__author__ = "DFT+LAMMPS Team"

# Package exports
from api_platform.gateway.main import gateway, app
from api_platform.auth.permissions import Permission, Role
from api_platform.webhooks.manager import WebhookEventType, webhook_manager

__all__ = [
    "gateway",
    "app",
    "Permission",
    "Role",
    "WebhookEventType",
    "webhook_manager",
]
