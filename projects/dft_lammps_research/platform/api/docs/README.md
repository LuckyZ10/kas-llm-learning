# DFT+LAMMPS API Platform Documentation

## Overview

The DFT+LAMMPS API Platform is a production-grade system that enables external systems and third-party developers to access the platform's core capabilities through a RESTful API.

## Features

### 1. RESTful API
- Standard REST interface following industry best practices
- Resource-oriented URL design
- Full CRUD operations for all resources
- Batch operation support for high-throughput workflows

### 2. API Gateway
- Unified entry point for all API requests
- OAuth2 and API Key authentication
- Rate limiting and quota management
- Request/response transformation
- API versioning (v1 stable, v2 beta)

### 3. Developer Portal
- Interactive API documentation (Swagger/OpenAPI)
- API key management
- Usage statistics and monitoring
- Code examples

### 4. Webhook System
- Event-driven notifications
- Asynchronous delivery with retries
- Idempotency and signature verification
- Delivery tracking and redelivery

### 5. SDKs
- **Python**: `dft-lammps-client`
- **JavaScript/TypeScript**: `dft-lammps-client`
- **Go**: `github.com/dft-lammps/go-client`

### 6. Integrations
- Jupyter Notebook extension
- VS Code extension
- Materials Project / AFLOW database connectors
- Airflow / Prefect workflow orchestration

## Quick Start

### 1. Get API Key

Visit the Developer Portal at `http://localhost:8080/portal` to create an API key.

### 2. Install SDK

**Python:**
```bash
pip install dft-lammps-client
```

**JavaScript:**
```bash
npm install dft-lammps-client
```

**Go:**
```bash
go get github.com/dft-lammps/go-client
```

### 3. Make Your First Request

**Python:**
```python
from dft_lammps import Client

client = Client(api_key="your-api-key")
project = client.projects.create(
    name="Battery Screening Study",
    project_type="battery_screening"
)
print(f"Created project: {project.id}")
```

**JavaScript:**
```javascript
const { Client } = require('dft-lammps-client');

const client = new Client({ apiKey: 'your-api-key' });
const project = await client.projects.create({
    name: 'Battery Screening Study',
    projectType: 'battery_screening'
});
console.log(`Created project: ${project.id}`);
```

**Go:**
```go
import "github.com/dft-lammps/go-client"

client, _ := dftlammps.NewClient("your-api-key")
project, _ := client.Projects.Create(ctx, &dftlammps.CreateProjectRequest{
    Name: "Battery Screening Study",
    ProjectType: "battery_screening",
})
```

**cURL:**
```bash
curl -X POST https://api.dft-lammps.org/api/v1/projects \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Battery Screening Study",
    "project_type": "battery_screening"
  }'
```

## API Reference

### Authentication

The API supports two authentication methods:

1. **API Key** (recommended for server-to-server):
   ```
   X-API-Key: your-api-key
   ```

2. **OAuth2 Bearer Token** (for user-facing applications):
   ```
   Authorization: Bearer access-token
   ```

### Rate Limits

| Tier | Requests/min | Requests/day | Calculations/day |
|------|--------------|--------------|------------------|
| Free | 60 | 10,000 | 100 |
| Pro | 300 | 100,000 | 10,000 |
| Enterprise | 1,000 | 1,000,000 | Unlimited |

### Projects API

**List Projects:**
```
GET /api/v1/projects?page=1&page_size=20
```

**Create Project:**
```
POST /api/v1/projects
{
  "name": "Project Name",
  "description": "Description",
  "project_type": "battery_screening",
  "target_properties": {"band_gap": {"min": 0.5, "max": 2.0}}
}
```

**Get Project:**
```
GET /api/v1/projects/{project_id}
```

**Update Project:**
```
PATCH /api/v1/projects/{project_id}
{
  "name": "New Name",
  "status": "active"
}
```

**Delete Project:**
```
DELETE /api/v1/projects/{project_id}
```

### Calculations API

**Submit Calculation:**
```
POST /api/v1/projects/{project_id}/calculations
{
  "structure": {
    "species": ["Li", "S"],
    "positions": [[0, 0, 0], [0.5, 0.5, 0.5]],
    "cell": [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
  },
  "calculation_type": "dft",
  "parameters": {"ecut": 500, "kpoints": "4 4 4"},
  "priority": 5
}
```

**Get Calculation:**
```
GET /api/v1/calculations/{calculation_id}
```

**Submit Batch:**
```
POST /api/v1/projects/{project_id}/calculations/batch
{
  "calculations": [
    {"structure": {...}, "calculation_type": "dft"},
    {"structure": {...}, "calculation_type": "dft"}
  ]
}
```

### Webhooks API

**Subscribe to Events:**
```
POST /api/v1/webhooks/subscribe
{
  "url": "https://your-server.com/webhook",
  "events": ["calculation.completed", "calculation.failed"],
  "metadata": {"team": "materials-research"}
}
```

**List Subscriptions:**
```
GET /api/v1/webhooks
```

**Delete Subscription:**
```
DELETE /api/v1/webhooks/{webhook_id}
```

## Webhook Events

### Event Types

| Event | Description |
|-------|-------------|
| `project.completed` | All calculations in project completed |
| `project.failed` | Project failed |
| `calculation.submitted` | Calculation submitted |
| `calculation.started` | Calculation started running |
| `calculation.completed` | Calculation completed successfully |
| `calculation.failed` | Calculation failed |
| `batch.completed` | Batch submission completed |
| `batch.failed` | Batch submission failed |
| `system.rate_limit_warning` | Approaching rate limit |
| `system.quota_exceeded` | Quota exceeded |

### Webhook Payload

```json
{
  "event_id": "evt_abc123",
  "event_type": "calculation.completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "calculation_id": "calc_def456",
    "project_id": "proj_ghi789",
    "calculation_type": "dft",
    "results": {
      "energy": -123.456,
      "forces": [...],
      "stress": [...]
    },
    "duration_seconds": 3600
  }
}
```

### Signature Verification

Webhooks are signed with HMAC-SHA256. Verify the signature:

**Python:**
```python
import hmac
import hashlib

signature = hmac.new(
    secret.encode(),
    payload.encode(),
    hashlib.sha256
).hexdigest()

expected = f"sha256={signature}"
assert hmac.compare_digest(expected, request.headers["X-Webhook-Signature"])
```

**JavaScript:**
```javascript
const crypto = require('crypto');

const signature = crypto
  .createHmac('sha256', secret)
  .update(payload)
  .digest('hex');

const expected = `sha256=${signature}`;
const isValid = crypto.timingSafeEqual(
  Buffer.from(expected),
  Buffer.from(request.headers['x-webhook-signature'])
);
```

## Jupyter Integration

```python
# Load extension
%load_ext dft_lammps_jupyter

# Set API key
%dftlammps_api_key your-api-key

# Check status
%dftlammps_status

# Submit calculation
%%dftlammps_calc project_id=proj_123 type=dft
structure = load_structure("Li2S.cif")
result = submit_calculation(structure, params={"ecut": 500})
```

## VS Code Extension

1. Install from VS Code marketplace
2. Click DFT+LAMMPS icon in sidebar
3. Sign in with API key
4. Browse projects and submit calculations directly

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY api_platform/ ./api_platform/
COPY examples/ ./examples/

EXPOSE 8080

CMD ["uvicorn", "api_platform.gateway.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Environment Variables

```bash
# Required
JWT_SECRET_KEY=your-secret-key

# Optional
REDIS_HOST=localhost
REDIS_PORT=6379
DATABASE_URL=postgresql://user:pass@localhost/api
LOG_LEVEL=INFO
CORS_ORIGINS=["*"]
```

## Support

- Documentation: https://docs.dft-lammps.org
- Developer Portal: https://api.dft-lammps.org/portal
- Support Email: support@dft-lammps.org
- GitHub Issues: https://github.com/dft-lammps/api-platform/issues
