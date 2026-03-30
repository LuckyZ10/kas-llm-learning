"""
Developer Portal

Interactive API documentation and developer management
"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from api_platform.gateway.main import gateway
from api_platform.auth.api_key import api_key_manager

router = APIRouter()


class DeveloperProfile(BaseModel):
    """Developer profile"""
    client_id: str
    name: str
    email: str
    tier: str
    created_at: str
    permissions: List[str]
    quota: dict


class APIKeyCreate(BaseModel):
    """Create API key request"""
    name: str = Field(..., description="Name for the API key")
    expires_days: Optional[int] = Field(365, description="Days until expiration")
    permissions: List[str] = Field(default_factory=list, description="Key-specific permissions")


class APIKeyResponse(BaseModel):
    """API key response"""
    key_id: str
    name: str
    key: str  # Only shown on creation
    tier: str
    permissions: List[str]
    created_at: str
    expires_at: Optional[str]


class APIKeyListItem(BaseModel):
    """API key list item"""
    key_id: str
    name: str
    tier: str
    permissions: List[str]
    created_at: str
    expires_at: Optional[str]
    last_used_at: Optional[str]
    usage_count: int
    active: bool


class UsageStats(BaseModel):
    """Usage statistics"""
    period: str
    requests: dict
    calculations: dict
    storage: dict
    limits: dict


# HTML template for developer portal
PORTAL_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DFT+LAMMPS API Developer Portal</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.10.0/swagger-ui.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            opacity: 0.9;
        }
        
        .nav {
            background: white;
            border-bottom: 1px solid #ddd;
            padding: 0 2rem;
            display: flex;
            gap: 2rem;
        }
        
        .nav a {
            padding: 1rem 0;
            text-decoration: none;
            color: #666;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        
        .nav a:hover, .nav a.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .card h3 {
            color: #333;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }
        
        .stat {
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 0;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .stat:last-child {
            border-bottom: none;
        }
        
        .stat-label {
            color: #666;
        }
        
        .stat-value {
            font-weight: 600;
            color: #333;
        }
        
        .btn {
            display: inline-block;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 500;
            transition: all 0.2s;
            border: none;
            cursor: pointer;
        }
        
        .btn-primary {
            background: #667eea;
            color: white;
        }
        
        .btn-primary:hover {
            background: #5a6fd6;
        }
        
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }
        
        .btn-secondary:hover {
            background: #e0e0e0;
        }
        
        .swagger-container {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            overflow: hidden;
        }
        
        .quickstart {
            background: #1a1a2e;
            color: #fff;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .quickstart h3 {
            margin-bottom: 1rem;
            color: #fff;
        }
        
        .code-block {
            background: #16213e;
            border-radius: 6px;
            padding: 1rem;
            overflow-x: auto;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        
        .code-block .comment {
            color: #6272a4;
        }
        
        .code-block .string {
            color: #f1fa8c;
        }
        
        .code-block .keyword {
            color: #ff79c6;
        }
        
        .code-block .function {
            color: #50fa7b;
        }
        
        .tier-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .tier-free {
            background: #e3f2fd;
            color: #1976d2;
        }
        
        .tier-pro {
            background: #f3e5f5;
            color: #7b1fa2;
        }
        
        .tier-enterprise {
            background: #fff3e0;
            color: #e65100;
        }
        
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        .feature {
            text-align: center;
            padding: 2rem;
        }
        
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .feature h4 {
            margin-bottom: 0.5rem;
            color: #333;
        }
        
        .feature p {
            color: #666;
            font-size: 0.95rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 DFT+LAMMPS API Developer Portal</h1>
        <p>Production-grade API for materials research and molecular dynamics simulations</p>
    </div>
    
    <div class="nav">
        <a href="#overview" class="active">Overview</a>
        <a href="#api-docs">API Documentation</a>
        <a href="#quickstart">Quick Start</a>
        <a href="#sdks">SDKs</a>
        <a href="#webhooks">Webhooks</a>
    </div>
    
    <div class="container">
        <div id="overview">
            <div class="grid">
                <div class="card">
                    <h3>📊 Your Usage</h3>
                    <div class="stat">
                        <span class="stat-label">API Requests (Today)</span>
                        <span class="stat-value">1,234 / 10,000</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Calculations</span>
                        <span class="stat-value">45 / 100</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Storage Used</span>
                        <span class="stat-value">0.5 GB / 1 GB</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Current Tier</span>
                        <span class="stat-value"><span class="tier-badge tier-free">Free</span></span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🔑 API Keys</h3>
                    <div class="stat">
                        <span class="stat-label">Active Keys</span>
                        <span class="stat-value">2</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Webhooks</span>
                        <span class="stat-value">1 Active</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Projects</span>
                        <span class="stat-value">3</span>
                    </div>
                    <div style="margin-top: 1rem;">
                        <a href="#" class="btn btn-primary">Manage API Keys</a>
                    </div>
                </div>
            </div>
            
            <div class="features">
                <div class="feature card">
                    <div class="feature-icon">⚡</div>
                    <h4>High Performance</h4>
                    <p>Sub-second response times with global edge caching</p>
                </div>
                <div class="feature card">
                    <div class="feature-icon">🔒</div>
                    <h4>Enterprise Security</h4>
                    <p>OAuth2 and API Key authentication with end-to-end encryption</p>
                </div>
                <div class="feature card">
                    <div class="feature-icon">📈</div>
                    <h4>Real-time Events</h4>
                    <p>Webhook notifications for calculation completions</p>
                </div>
                <div class="feature card">
                    <div class="feature-icon">🚀</div>
                    <h4>HPC Integration</h4>
                    <p>Seamless integration with SLURM and PBS schedulers</p>
                </div>
            </div>
        </div>
        
        <div id="quickstart" class="quickstart">
            <h3>🚀 Quick Start</h3>
            <div class="code-block">
<span class="comment"># Install the Python SDK</span>
$ pip install dft-lammps-client

<span class="comment"># Submit your first calculation</span>
<span class="keyword">from</span> dft_lammps <span class="keyword">import</span> Client

client = Client(api_key=<span class="string">"your-api-key"</span>)

project = client.projects.create(
    name=<span class="string">"Li-S Battery Study"</span>,
    project_type=<span class="string">"battery_screening"</span>
)

calculation = client.calculations.submit(
    project_id=project.id,
    structure=<span class="string">"Li2S.cif"</span>,
    calculation_type=<span class="string">"dft"</span>
)

<span class="comment"># Wait for completion via webhook or poll</span>
result = client.calculations.wait(calculation.id)
<span class="function">print</span>(result.energy)
            </div>
        </div>
        
        <div id="api-docs">
            <h2 style="margin-bottom: 1rem;">📚 API Documentation</h2>
            <div class="swagger-container">
                <div id="swagger-ui"></div>
            </div>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.10.0/swagger-ui-bundle.js"></script>
    <script>
        window.onload = function() {
            SwaggerUIBundle({
                url: '/openapi.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.presets.standalone
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "BaseLayout",
                validatorUrl: null,
                tryItOutEnabled: true,
                supportedSubmitMethods: ['get', 'post', 'put', 'delete', 'patch'],
                onComplete: function() {
                    console.log('Swagger UI loaded');
                }
            });
        };
    </script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def developer_portal():
    """Developer portal main page"""
    return HTMLResponse(content=PORTAL_HTML)


@router.get(
    "/profile",
    response_model=DeveloperProfile,
    summary="Get profile",
    description="Get current developer profile"
)
async def get_profile(
    auth: dict = Depends(gateway.authenticate_request)
):
    """Get developer profile information"""
    return DeveloperProfile(
        client_id=auth["client_id"],
        name=auth.get("client_name", "Unknown"),
        email="developer@example.com",  # In production: from database
        tier=auth["tier"],
        created_at="2024-01-01T00:00:00Z",  # In production: from database
        permissions=auth["permissions"],
        quota={
            "requests_per_minute": 60 if auth["tier"] == "free" else 300,
            "requests_per_day": 10000 if auth["tier"] == "free" else 100000,
            "max_projects": 5 if auth["tier"] == "free" else 50,
        }
    )


@router.post(
    "/api-keys",
    response_model=APIKeyResponse,
    summary="Create API key",
    description="Create a new API key"
)
async def create_api_key(
    key_data: APIKeyCreate,
    auth: dict = Depends(gateway.authenticate_request)
):
    """
    Create a new API key for programmatic access.
    
    The key will only be shown once. Store it securely.
    """
    api_key, key_id = await api_key_manager.generate_api_key(
        client_id=auth["client_id"],
        name=key_data.name,
        tier=auth["tier"],
        permissions=key_data.permissions or auth["permissions"],
        expires_days=key_data.expires_days
    )
    
    return APIKeyResponse(
        key_id=key_id,
        name=key_data.name,
        key=api_key,
        tier=auth["tier"],
        permissions=key_data.permissions or auth["permissions"],
        created_at=datetime.utcnow().isoformat(),
        expires_at=(datetime.utcnow() + datetime.timedelta(days=key_data.expires_days)).isoformat() if key_data.expires_days else None
    )


@router.get(
    "/api-keys",
    response_model=List[APIKeyListItem],
    summary="List API keys",
    description="List all API keys"
)
async def list_api_keys(
    auth: dict = Depends(gateway.authenticate_request)
):
    """List all API keys (without the actual key values)"""
    keys = await api_key_manager.get_client_keys(auth["client_id"])
    return [APIKeyListItem(**key) for key in keys]


@router.delete(
    "/api-keys/{key_id}",
    status_code=204,
    summary="Revoke API key",
    description="Revoke an API key"
)
async def revoke_api_key(
    key_id: str,
    auth: dict = Depends(gateway.authenticate_request)
):
    """Revoke an API key"""
    success = await api_key_manager.revoke_key(key_id, auth["client_id"])
    
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")
    
    return None


@router.get(
    "/usage",
    response_model=UsageStats,
    summary="Get usage stats",
    description="Get detailed usage statistics"
)
async def get_usage_stats(
    period: str = "30d",
    auth: dict = Depends(gateway.authenticate_request)
):
    """
    Get API usage statistics.
    
    - **period**: Time period (24h, 7d, 30d, 90d)
    """
    return UsageStats(
        period=period,
        requests={
            "total": 1234,
            "successful": 1200,
            "failed": 34,
            "by_endpoint": {
                "GET /projects": 500,
                "POST /calculations": 200,
                "GET /calculations": 534,
            }
        },
        calculations={
            "total": 45,
            "completed": 42,
            "failed": 3,
            "by_type": {
                "dft": 20,
                "lammps": 15,
                "ml": 10,
            }
        },
        storage={
            "used_gb": 0.5,
            "total_gb": 1 if auth["tier"] == "free" else 50,
        },
        limits={
            "requests_per_minute": 60 if auth["tier"] == "free" else 300,
            "requests_per_day": 10000 if auth["tier"] == "free" else 100000,
            "calculations_per_day": 100 if auth["tier"] == "free" else 10000,
        }
    )
