# DFT-LAMMPS Web Platform

A comprehensive web interface and RESTful API for the DFT-LAMMPS materials simulation platform, integrating DFT calculations, Machine Learning potentials, and Molecular Dynamics simulations.

## Features

- **Modern Web Interface**: React-based SPA with Material-UI design
- **RESTful API**: FastAPI-powered backend with automatic OpenAPI documentation
- **Asynchronous Task Queue**: Redis + Celery for distributed task processing
- **Authentication & Authorization**: JWT-based auth with role-based access control
- **Structure Visualization**: 3D molecular structure viewer (integrates with 3DMol.js/three.js)
- **Result Analysis**: Interactive dashboards for simulation results
- **Multi-database Support**: PostgreSQL for structured data, MongoDB for flexible documents

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Nginx (80/443)                       │
│                    (Reverse Proxy + Static)                 │
└─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
    ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
    │  React SPA  │    │  FastAPI    │    │   Uploads   │
    │  (Frontend) │    │   (API)     │    │   /Results  │
    └─────────────┘    └──────┬──────┘    └─────────────┘
                              │
               ┌──────────────┼──────────────┐
               │              │              │
        ┌──────▼──────┐┌──────▼──────┐┌──────▼──────┐
        │   Redis     ││  PostgreSQL ││   MongoDB   │
        │  (Queue)    ││  (Primary)  ││  (Documents)│
        └──────┬──────┘└─────────────┘└─────────────┘
               │
        ┌──────▼──────┐
        │    Celery   │
        │   Workers   │
        └─────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 4GB RAM available
- Ports 80, 8000, 5432, 6379, 27017 available

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-org/dftlammps.git
cd dftlammps
```

2. **Set environment variables** (optional but recommended):
```bash
export POSTGRES_PASSWORD=your_secure_password
export MONGODB_PASSWORD=your_mongodb_password
export JWT_SECRET_KEY=your_jwt_secret
export DEFAULT_ADMIN_PASSWORD=your_admin_password
```

3. **Start the services**:
```bash
docker-compose -f docker-compose.web.yml up -d
```

4. **Access the application**:
- Web Interface: http://localhost
- API Documentation: http://localhost/api/docs
- Flower (Celery Monitor): http://localhost:5555

### Default Credentials

- **Username**: `admin`
- **Password**: `admin123` (or your `DEFAULT_ADMIN_PASSWORD`)

**⚠️ Important**: Change the default password immediately after first login!

## API Documentation

The API is automatically documented using OpenAPI (Swagger) and ReDoc:

- **Swagger UI**: http://localhost/api/docs
- **ReDoc**: http://localhost/api/redoc
- **OpenAPI JSON**: http://localhost/api/openapi.json

### Authentication

All API endpoints require Bearer token authentication:

```bash
# Login to get token
curl -X POST http://localhost/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Use token in subsequent requests
curl http://localhost/api/v1/tasks \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Example: Submit a Task

```bash
curl -X POST http://localhost/api/v1/tasks \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "full_workflow",
    "name": "Li3PS4 Simulation",
    "description": "Full workflow test",
    "material_id": "mp-1234",
    "dft_config": {
      "code": "vasp",
      "functional": "PBE",
      "encut": 520
    },
    "ml_config": {
      "framework": "deepmd",
      "preset": "fast"
    }
  }'
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_PASSWORD` | PostgreSQL password | `dftlammps_secure_pass` |
| `MONGODB_PASSWORD` | MongoDB password | `mongodb_secure_pass` |
| `JWT_SECRET_KEY` | JWT signing key | Auto-generated |
| `DEFAULT_ADMIN_PASSWORD` | Initial admin password | `admin123` |
| `API_HOST` | API bind address | `0.0.0.0` |
| `API_PORT` | API port | `8000` |

### Volume Mounts

- `postgres_data`: PostgreSQL database files
- `mongodb_data`: MongoDB database files
- `redis_data`: Redis persistence files
- `uploads_data`: Uploaded structure files
- `results_data`: Simulation results

## Services

| Service | Port | Description |
|---------|------|-------------|
| nginx | 80/443 | Reverse proxy and static files |
| api | 8000 | FastAPI backend |
| postgres | 5432 | PostgreSQL database |
| mongodb | 27017 | MongoDB database |
| redis | 6379 | Redis cache and queue |
| flower | 5555 | Celery monitoring UI |

## Development

### Frontend Development

```bash
cd web/frontend
npm install
npm start  # Runs on http://localhost:3000
```

### API Development

```bash
# Install dependencies
pip install -r web/requirements.web.txt

# Run development server
uvicorn dftlammps.web.api_server:app --reload
```

### Running Tests

```bash
# Backend tests
pytest dftlammps/web/tests/

# Frontend tests
cd web/frontend
npm test
```

## Troubleshooting

### Check service logs

```bash
docker-compose -f docker-compose.web.yml logs -f [service-name]
```

### Reset all data

```bash
docker-compose -f docker-compose.web.yml down -v
docker-compose -f docker-compose.web.yml up -d
```

### API not responding

```bash
# Check API health
curl http://localhost/health

# Check if database is connected
curl http://localhost/api/v1/health
```

## Production Deployment

### SSL/TLS

1. Obtain SSL certificates
2. Mount certificates to nginx container
3. Update nginx.conf to enable HTTPS

### Resource Limits

Add resource limits to docker-compose.yml:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

### Backup Strategy

```bash
# Backup PostgreSQL
docker exec dftlammps-postgres pg_dump -U dftlammps dftlammps > backup.sql

# Backup MongoDB
docker exec dftlammps-mongodb mongodump --out /backup

# Backup uploads and results
tar -czvf dftlammps-backup.tar.gz /path/to/volumes
```

## License

MIT License - see LICENSE file for details.

## Support

For issues and feature requests, please use GitHub Issues.

For questions and discussions, use GitHub Discussions.

## Acknowledgments

- FastAPI for the excellent web framework
- Celery for distributed task processing
- Materials Project for structure data
- LAMMPS and VASP communities
