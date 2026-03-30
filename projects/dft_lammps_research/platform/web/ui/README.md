# DFT+LAMMPS Research Platform - Web UI V2

A modern, production-grade web interface for DFT+LAMMPS materials research workflow management.

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
cd webui_v2
docker-compose up -d
```

Access the application:
- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **WebSocket**: ws://localhost:8000/ws/global

### Manual Setup

#### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 📁 Project Structure

```
webui_v2/
├── backend/           # FastAPI + WebSocket + Redis
│   ├── app/
│   │   ├── api/       # REST API endpoints
│   │   ├── models/    # SQLAlchemy database models
│   │   ├── schemas/   # Pydantic validation schemas
│   │   ├── services/  # Business logic
│   │   ├── websocket/ # Real-time communication
│   │   └── tasks/     # Celery background tasks
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/          # React + TypeScript + Tailwind CSS
│   ├── src/
│   │   ├── api/       # API client
│   │   ├── components/# React components
│   │   ├── pages/     # Page components
│   │   ├── contexts/  # React contexts
│   │   └── store/     # Zustand stores
│   ├── Dockerfile
│   └── package.json
├── docker/            # Docker configurations
├── docs/              # Documentation
└── docker-compose.yml
```

## ✨ Features

### Project Management
- Create and manage research projects
- Track calculation progress
- Organize by material systems

### Workflow Designer
- Visual node-based workflow editor
- Drag-and-drop interface
- Support for: DFT, MD, ML training, analysis
- Real-time execution monitoring

### Real-time Monitoring
- Live task status updates via WebSocket
- ML training progress with live charts
- MD simulation metrics
- Active learning progress tracking

### Structure Visualization
- Interactive 3D structure viewer (Three.js)
- Ball-and-stick and space-filling modes
- Unit cell display
- Support for CIF, POSCAR, XYZ formats

### Screening Results
- Advanced filtering and search
- Property scatter plots
- Multi-structure comparison
- Export to CSV/PDF

### Report Generation
- Project reports (PDF/HTML/Markdown)
- Screening reports with top candidates
- Workflow execution reports
- Customizable templates

## 🏗️ Architecture

### Frontend
- **Framework**: React 18 + TypeScript 5
- **Styling**: Tailwind CSS + Headless UI
- **3D Visualization**: React Three Fiber + Three.js
- **Charts**: Plotly.js
- **State Management**: Zustand + React Query
- **Routing**: React Router v6
- **Real-time**: Native WebSocket API

### Backend
- **Framework**: FastAPI (Python)
- **Database**: PostgreSQL + SQLAlchemy
- **Cache/PubSub**: Redis
- **Task Queue**: Celery + Redis
- **Authentication**: JWT tokens
- **WebSocket**: Native FastAPI WebSocket

## 📚 Documentation

- [User Guide](./docs/user-guide.md) - Complete user manual
- [API Reference](./docs/api-reference.md) - REST API documentation
- [Development Guide](./docs/development.md) - Developer setup and guidelines

## 🛠️ Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Code Style

```bash
# Python formatting
black app/

# TypeScript linting
cd frontend
npm run lint
```

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/dft_lammps

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key
```

## 📝 License

MIT License - See [LICENSE](../LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 🐛 Troubleshooting

### Common Issues

**WebSocket Connection Failed**:
- Check if backend is running
- Verify Redis is accessible
- Check browser console for errors

**Database Connection Error**:
- Verify PostgreSQL is running
- Check DATABASE_URL configuration
- Ensure database exists

**Frontend Build Errors**:
- Clear node_modules: `rm -rf node_modules && npm install`
- Check Node.js version (requires 18+)

## 📞 Support

For issues and questions:
- Check the [User Guide](./docs/user-guide.md)
- Review [API Documentation](http://localhost:8000/docs)
- Open an issue on GitHub

## 📊 Project Statistics

- **Total Files**: 85+
- **Backend Code**: FastAPI with 20+ API endpoints
- **Frontend Code**: React + TypeScript with 15+ pages
- **Database Models**: 6 SQLAlchemy models
- **Documentation**: 4 comprehensive guides

## 🎯 Key Capabilities

| Feature | Technology | Status |
|---------|------------|--------|
| REST API | FastAPI | ✅ Complete |
| WebSocket | Native WS | ✅ Complete |
| Database | PostgreSQL + SQLAlchemy | ✅ Complete |
| Cache/Queue | Redis + Celery | ✅ Complete |
| Frontend Framework | React 18 + TS | ✅ Complete |
| Styling | Tailwind CSS | ✅ Complete |
| 3D Visualization | Three.js + R3F | ✅ Complete |
| Charts | Plotly.js | ✅ Complete |
| Auth | JWT Tokens | ✅ Complete |
| Report Generation | PDF/HTML/CSV | ✅ Complete |
| Docker Deployment | Compose | ✅ Complete |
