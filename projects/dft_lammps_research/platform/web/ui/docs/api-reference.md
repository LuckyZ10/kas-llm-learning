# DFT+LAMMPS Research Platform - API Reference

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

All API requests require authentication via Bearer token:

```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/projects
```

### Login

```http
POST /api/v1/auth/login
Content-Type: application/x-www-form-urlencoded

username=researcher&password=researcher
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": "uuid",
    "email": "researcher@example.com",
    "username": "researcher",
    "role": "researcher"
  }
}
```

---

## Projects

### List Projects

```http
GET /api/v1/projects?page=1&page_size=20&status=active
```

**Query Parameters:**
- `status` (optional): Filter by status (draft, active, completed, archived)
- `search` (optional): Search in project name
- `page` (optional): Page number (default: 1)
- `page_size` (optional): Items per page (default: 20, max: 100)

**Response:**
```json
{
  "items": [
    {
      "id": "uuid",
      "name": "Battery Screening Project",
      "status": "active",
      "project_type": "battery_screening",
      "total_structures": 1000,
      "completed_calculations": 750,
      "created_at": "2024-01-15T10:00:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20
}
```

### Create Project

```http
POST /api/v1/projects
Content-Type: application/json

{
  "name": "New Project",
  "description": "Project description",
  "project_type": "battery_screening",
  "material_system": "Li-Mn-O",
  "work_directory": "./workdir/new_project",
  "target_properties": {
    "ionic_conductivity": { "min": 0.001, "max": 1.0 }
  }
}
```

### Get Project

```http
GET /api/v1/projects/{project_id}
```

### Update Project

```http
PATCH /api/v1/projects/{project_id}
Content-Type: application/json

{
  "name": "Updated Name",
  "status": "active"
}
```

### Delete Project

```http
DELETE /api/v1/projects/{project_id}
```

---

## Workflows

### List Workflows

```http
GET /api/v1/workflows?project_id=uuid&status=running
```

**Query Parameters:**
- `project_id` (optional): Filter by project
- `workflow_type` (optional): Filter by type
- `status` (optional): Filter by status

### Create Workflow

```http
POST /api/v1/workflows
Content-Type: application/json

{
  "name": "Active Learning Workflow",
  "description": "Automated training workflow",
  "workflow_type": "active_learning",
  "project_id": "project-uuid",
  "definition": {
    "nodes": [
      {
        "id": "node-1",
        "type": "structure_input",
        "position": { "x": 100, "y": 100 },
        "data": {
          "label": "Input Structures",
          "node_type": "input",
          "config": {}
        }
      }
    ],
    "edges": []
  }
}
```

### Execute Workflow

```http
POST /api/v1/workflows/{workflow_id}/execute
Content-Type: application/json

{
  "initial_context": {
    "temperature": 300,
    "pressure": 1.0
  }
}
```

### Control Workflow

```http
POST /api/v1/workflows/{workflow_id}/pause
POST /api/v1/workflows/{workflow_id}/cancel
```

---

## Tasks

### List Tasks

```http
GET /api/v1/tasks?workflow_id=uuid&status=running
```

### Get Task

```http
GET /api/v1/tasks/{task_id}
```

### Get Task Logs

```http
GET /api/v1/tasks/{task_id}/logs?lines=100
```

---

## Screening Results

### List Results

```http
GET /api/v1/screening?project_id=uuid&min_ionic_conductivity=0.001
```

**Query Parameters:**
- `project_id` (optional): Filter by project
- `formula` (optional): Search formula
- `min_ionic_conductivity` (optional): Minimum conductivity
- `max_formation_energy` (optional): Maximum formation energy

### Get Result

```http
GET /api/v1/screening/{result_id}
```

### Filter Results

```http
POST /api/v1/screening/filter
Content-Type: application/json

{
  "project_id": "uuid",
  "formula_contains": "LiMn",
  "property_ranges": {
    "formation_energy": { "min": -5.0, "max": 0.0 },
    "band_gap": { "min": 0.5, "max": 5.0 }
  },
  "ml_only": false,
  "dft_only": false
}
```

### Compare Structures

```http
POST /api/v1/screening/compare
Content-Type: application/json

{
  "structure_ids": ["id1", "id2", "id3"],
  "properties": ["formation_energy", "band_gap", "ionic_conductivity"]
}
```

---

## Monitoring

### System Stats

```http
GET /api/v1/monitoring/stats
```

**Response:**
```json
{
  "workflows": {
    "running": 5,
    "completed": 100,
    "failed": 2
  },
  "tasks": {
    "running": 20,
    "completed": 500,
    "failed": 5
  },
  "resources": {
    "total_cpu_hours": 1250.5,
    "total_memory_gb": 2048.0
  }
}
```

### Training Metrics

```http
GET /api/v1/monitoring/training?model_id=optional
```

**Response:**
```json
{
  "current": {
    "loss": 0.00123,
    "force_rmse": 0.045,
    "energy_rmse": 0.0008,
    "lr": 0.0001
  },
  "history": {
    "batch": [0, 100, 200, ...],
    "loss": [0.5, 0.1, 0.05, ...],
    "force_rmse": [0.5, 0.2, 0.1, ...]
  }
}
```

### MD Metrics

```http
GET /api/v1/monitoring/md/{trajectory_id}?metric=temperature
```

**Metrics:** temperature, energy, pressure, volume

### Active Learning Progress

```http
GET /api/v1/monitoring/al/progress?project_id=optional
```

---

## Files

### List Files

```http
GET /api/v1/files/list?path=.&pattern=*.out
```

### Download File

```http
GET /api/v1/files/download/{file_path}
```

### Upload File

```http
POST /api/v1/files/upload?path=uploads
Content-Type: multipart/form-data

file: (binary)
```

---

## Reports

### Generate Project Report

```http
POST /api/v1/reports/project/{project_id}?format=pdf&include_charts=true
```

**Formats:** pdf, html, markdown

### Generate Workflow Report

```http
POST /api/v1/reports/workflow/{workflow_id}?format=pdf
```

### Generate Screening Report

```http
POST /api/v1/reports/screening?project_id=uuid&top_n=50&format=csv
```

---

## WebSocket

Real-time updates via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/global')

ws.onmessage = (event) => {
  const message = JSON.parse(event.data)
  console.log(message)
}
```

### Message Types

**Task Update:**
```json
{
  "type": "task_update",
  "task_id": "uuid",
  "status": "completed",
  "data": {},
  "timestamp": 1234567890
}
```

**Workflow Update:**
```json
{
  "type": "workflow_update",
  "workflow_id": "uuid",
  "status": "running",
  "progress": 45.5,
  "timestamp": 1234567890
}
```

**System Stats:**
```json
{
  "type": "system_stats",
  "stats": {},
  "timestamp": 1234567890
}
```

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Invalid request parameters"
}
```

### 401 Unauthorized

```json
{
  "detail": "Not authenticated"
}
```

### 403 Forbidden

```json
{
  "detail": "Not authorized to access this resource"
}
```

### 404 Not Found

```json
{
  "detail": "Resource not found"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Internal server error"
}
```

---

## Rate Limiting

API requests are rate-limited:
- 100 requests per minute for authenticated users
- 20 requests per minute for anonymous users

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```
