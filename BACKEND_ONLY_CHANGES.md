# HMS EEG Classification System - Backend Only Configuration

## Summary of Changes

This document summarizes the changes made to configure the HMS EEG Classification System as a backend-only API service, with the frontend to be developed separately.

## Changes Made

### 1. Docker Compose (`docker-compose.yml`)
- **Commented out** the `frontend` service (lines 238-253)
- **Commented out** the `nginx` service (lines 255-272) - will be enabled when frontend is ready
- Added comments indicating these services will be configured when frontend is ready

### 2. Dockerfile
- **Commented out** Stage 7: Web interface builder (node/frontend build)
- **Commented out** Stage 8: Nginx web server
- Added comments indicating these stages will be added when frontend is ready

### 3. Makefile
- Updated success messages to indicate "Backend System" instead of full system
- Changed main URL from `http://localhost:3000` to `http://localhost:8000/docs`
- Removed references to "Main App" on port 3000

### 4. Python Scripts

#### `run_docker.py`:
- Removed frontend service from health checks
- Changed browser opening from port 3000 to API docs at port 8000
- Updated URL display to show API endpoints instead of frontend
- Removed `webapp/frontend` from directory creation

#### `docker-run.sh`:
- No changes needed (uses Makefile commands)

### 5. Documentation

#### `README.md`:
- Added "Backend API" to the title
- Clarified this is a backend system with frontend to be developed separately
- Updated service endpoints to reflect backend-only services
- Added section on "Frontend Integration (Planned)"

#### `DOCKER_QUICKSTART.md`:
- Updated title to include "Backend API"
- Added clarification about frontend being developed separately
- Added API testing examples (curl and Python)
- Added section on future frontend integration

### 6. Additional Files
- Created `webapp/nginx.conf.example` as a template for future nginx configuration
- Created `test_docker_build.sh` to verify Docker builds work correctly
- Created this file (`BACKEND_ONLY_CHANGES.md`) to document changes

## How to Run

### Quick Start
```bash
# Build and run with pre-built models
make quick

# OR full setup with training
make all
```

### Access Points
- **API Documentation**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health
- **MLflow**: http://localhost:5000
- **Grafana**: http://localhost:3001
- **Visualization Dashboard**: http://localhost:8050

## Frontend Integration (Future)

When the frontend is ready:

1. Uncomment the `frontend` service in `docker-compose.yml`
2. Uncomment the `nginx` service in `docker-compose.yml`
3. Uncomment the web-builder and web-server stages in `Dockerfile`
4. Update `nginx.conf` with proper routing
5. Update documentation to include frontend URLs

The backend API is already configured with:
- CORS support for cross-origin requests
- WebSocket support for real-time features
- RESTful endpoints
- JWT authentication ready

## Testing the API

### Health Check
```bash
curl http://localhost:8000/health
```

### Example Prediction Request
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "eeg_data": [...],
        "spectrogram_data": [...]
    }
)
print(response.json())
```

## Troubleshooting

If you encounter build errors:
1. Run `docker-compose config` to validate the configuration
2. Ensure all required directories exist
3. Check that Python dependencies are compatible with Python 3.9 (used in Docker) 