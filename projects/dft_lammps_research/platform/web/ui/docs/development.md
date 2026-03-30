# Development Guide

## Development Environment Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional)

### Backend Development

1. **Create virtual environment:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Setup environment variables:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Setup database:**
```bash
# Create database
createdb dft_lammps

# Run migrations (if using Alembic)
alembic upgrade head
```

5. **Start development server:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Frontend Development

1. **Install dependencies:**
```bash
cd frontend
npm install
```

2. **Start development server:**
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Full Stack Development

Use Docker Compose for a complete development environment:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down
```

## Project Structure

```
webui_v2/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── endpoints/      # API route handlers
│   │   │   └── router.py       # API router configuration
│   │   ├── core/               # Core configuration
│   │   ├── db/                 # Database setup
│   │   ├── models/             # SQLAlchemy models
│   │   ├── schemas/            # Pydantic schemas
│   │   ├── services/           # Business logic
│   │   ├── tasks/              # Celery background tasks
│   │   ├── websocket/          # WebSocket handlers
│   │   ├── celery.py           # Celery configuration
│   │   └── main.py             # FastAPI application
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── api/                # API client
│   │   ├── components/         # React components
│   │   ├── contexts/           # React contexts
│   │   ├── pages/              # Page components
│   │   ├── store/              # Zustand stores
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
├── docs/
├── docker-compose.yml
└── README.md
```

## API Development

### Adding a New Endpoint

1. **Create endpoint file** in `backend/app/api/endpoints/`
2. **Add router** in `backend/app/api/router.py`
3. **Create schemas** in `backend/app/schemas/`

Example:

```python
# backend/app/api/endpoints/my_feature.py
from fastapi import APIRouter
from app.schemas.my_feature import MyFeatureCreate, MyFeatureResponse

router = APIRouter()

@router.get("", response_model=List[MyFeatureResponse])
async def list_items():
    return []

@router.post("", response_model=MyFeatureResponse)
async def create_item(data: MyFeatureCreate):
    return {}
```

### Database Models

1. **Create model** in `backend/app/models/`
2. **Import in database.py** to register
3. **Create migration** (if using Alembic)

## Frontend Development

### Adding a New Page

1. **Create page component** in `frontend/src/pages/`
2. **Add route** in `frontend/src/App.tsx`
3. **Add navigation link** in `Layout.tsx`

### State Management

Use Zustand for global state:

```typescript
// frontend/src/store/myStore.ts
import { create } from 'zustand'

interface MyStore {
  value: string
  setValue: (value: string) => void
}

export const useMyStore = create<MyStore>((set) => ({
  value: '',
  setValue: (value) => set({ value }),
}))
```

### API Integration

Add new API methods to the client:

```typescript
// frontend/src/api/client.ts
export const myFeatureApi = {
  getAll: () => api.get('/my-feature'),
  create: (data: any) => api.post('/my-feature', data),
}
```

Use React Query for data fetching:

```typescript
const { data, isLoading } = useQuery({
  queryKey: ['my-feature'],
  queryFn: () => myFeatureApi.getAll().then(res => res.data),
})
```

## Testing

### Backend Tests

```bash
cd backend
pytest

# With coverage
pytest --cov=app
```

### Frontend Tests

```bash
cd frontend
npm test

# With coverage
npm run test:coverage
```

## Code Style

### Python

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters

```bash
# Format code
black app/

# Lint
flake8 app/

# Type check
mypy app/
```

### TypeScript

- Use strict mode
- Explicit return types on functions
- Prefer interfaces over types

```bash
# Lint
npm run lint

# Type check
npm run type-check
```

## Deployment

### Production Build

```bash
# Build frontend
cd frontend
npm run build

# Build backend (if needed)
cd ../backend
pip install -r requirements.txt
```

### Docker Production

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Run production stack
docker-compose -f docker-compose.prod.yml up -d
```

## Debugging

### Backend Debugging

```python
import structlog

logger = structlog.get_logger()

# Use structured logging
logger.info("Processing", task_id=task_id, status="running")
logger.error("Failed", error=str(e))
```

### Frontend Debugging

```typescript
// React DevTools
// Redux DevTools (for Zustand)

// Console logging with context
console.log('[WorkflowEditor]', 'Node added:', node)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests
5. Submit pull request

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Tailwind CSS Documentation](https://tailwindcss.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
