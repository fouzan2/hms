# HMS EEG Classification System - Quick Start Guide

## 🚀 Run Everything with One Command

```bash
make run
```

That's it! This single command will:
1. ✅ Check system requirements
2. ✅ Set up the environment
3. ✅ Download the dataset from Kaggle (folder by folder)
4. ✅ Prepare and preprocess data
5. ✅ Train the ML models
6. ✅ Build and start all services with Docker
7. ✅ Launch the web interface
8. ✅ Open your browser to the dashboard

## 📋 Prerequisites

Before running, ensure you have:

- **Docker** and **Docker Compose** installed
- **Python 3.8+** installed
- **Node.js 18+** and npm installed
- **Kaggle API credentials** (for dataset download)
- **NVIDIA GPU** (optional, but recommended for faster training)

## 🔧 Initial Setup (One-time only)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd hms
   ```

2. **Set up Kaggle credentials**:
   ```bash
   # Create .kaggle directory in your home folder
   mkdir -p ~/.kaggle
   
   # Download your kaggle.json from Kaggle Account Settings
   # Place it in ~/.kaggle/
   cp ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Configure environment**:
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env and add your Kaggle credentials:
   # KAGGLE_USERNAME=your_username
   # KAGGLE_KEY=your_api_key
   ```

## 🏃 Running the Project

### Option 1: Fully Automated (Recommended)
```bash
make run
```

### Option 2: Using Python Script
```bash
python run_project.py
```

### Option 3: With Options
```bash
# Skip dataset download (if already downloaded)
python run_project.py --skip-download

# Skip model training (use pre-trained models)
python run_project.py --skip-train

# Both options
python run_project.py --skip-download --skip-train
```

## 🌐 Accessing the System

Once running, access the system at:

- **Main Dashboard**: http://localhost
- **API Documentation**: http://localhost/api/docs
- **MLflow UI**: http://localhost/mlflow
- **Grafana Monitoring**: http://localhost/grafana (admin/admin)
- **Visualization Dashboard**: http://localhost/dashboard

## 📊 What You'll See

### Main Dashboard
- System overview with key metrics
- Quick actions for common tasks
- Real-time activity feed
- System health status

### Upload Page
- Drag-and-drop EEG file upload
- Support for EDF, BDF, and CSV formats
- Progress tracking for uploads
- Batch upload capabilities

### Monitoring Page
- Real-time analysis progress
- System resource utilization
- Processing queue status
- Live performance metrics

## 🛠️ Common Commands

```bash
# View logs
make logs

# Stop all services
make stop

# Restart services
make restart

# View specific service logs
make logs-api
make logs-frontend

# Access service shells
make shell-api
make shell-frontend

# Run tests
make test

# Clean everything
make clean
```

## 🔍 Troubleshooting

### Docker Issues
```bash
# Check Docker is running
docker info

# Clean Docker resources
docker system prune -a
```

### Port Conflicts
If you get port already in use errors:
```bash
# Find process using port (e.g., 3000)
lsof -i :3000

# Kill the process
kill -9 <PID>
```

### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Install NVIDIA Docker support
make install-nvidia-docker
```

### Dataset Download Failed
1. Verify Kaggle credentials in `.env`
2. Check internet connection
3. Try downloading specific folders:
   ```bash
   docker-compose run --rm data-downloader python scripts/download_dataset.py --folders train_eegs
   ```

## 📁 Project Structure

```
hms/
├── webapp/frontend/      # Next.js web interface
├── src/                  # Python source code
├── models/              # Trained ML models
├── data/                # Dataset storage
├── docker-compose.yml   # Service orchestration
├── Makefile            # Command shortcuts
└── run_project.py      # One-command runner
```

## 🚦 Development Workflow

### Frontend Development
```bash
cd webapp/frontend
npm run dev
```

### Backend Development
```bash
make dev-api
```

### Jupyter Notebooks
```bash
make jupyter
# Access at http://localhost:8888 (token: jupyter_token)
```

## 🔐 Security Notes

- Change default passwords in production
- Update `NEXTAUTH_SECRET` in `.env`
- Configure proper SSL certificates
- Implement proper authentication

## 💡 Tips

1. **First Run**: The initial run will take 30-60 minutes due to dataset download and model training
2. **Subsequent Runs**: Use `--skip-download --skip-train` flags for faster startup
3. **GPU Usage**: Training is 10x faster with GPU support
4. **Memory**: Ensure at least 16GB RAM available
5. **Storage**: Reserve 50GB+ for dataset and models

## 🆘 Getting Help

- Check logs: `make logs`
- View documentation: See `/docs` folder
- Common issues: See `TROUBLESHOOTING.md`

## 🎉 Success!

When everything is running correctly, you'll see:
```
✅ HMS EEG Classification System is fully deployed!
📊 Access points:
   - Web Interface: http://localhost
   - API: http://localhost:8000/docs
   - MLflow: http://localhost:5000
   - Grafana: http://localhost:3001
```

Press `Ctrl+C` to stop all services when done. 