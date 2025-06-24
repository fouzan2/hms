"""
FastAPI deployment for HMS brain activity classification system.
Provides endpoints for real-time and batch EEG analysis.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import io
import time
import asyncio
from datetime import datetime
import logging
import yaml
import mne
import h5py
import tempfile
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import redis.asyncio as redis
from collections import deque
import uuid
import json

from ..models.ensemble_model import HMSEnsembleModel
from ..preprocessing.eeg_preprocessor import EEGPreprocessor
from ..preprocessing.spectrogram_generator import SpectrogramGenerator
from ..utils.interpretability import ModelInterpreter, UncertaintyQuantification
from ..evaluation.evaluator import ClinicalMetrics
from .streaming_api import StreamProcessor, WebSocketManager, AlertManager, KafkaStreamProcessor
from .model_manager import ModelSerializer, ModelValidator

# Import Phase 8 modules
from .model_optimization import OptimizedModelWrapper, OptimizationConfig
from .performance_monitoring import (
    MonitoringConfig, create_monitoring_system, 
    PerformanceMonitor, ModelDriftDetector, ResourceMonitor
)
from .distributed_training import DistributedModelServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HMS Brain Activity Classification API",
    description="Medical-grade EEG classification for harmful brain activities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
prediction_counter = Counter('eeg_predictions_total', 'Total number of predictions')
prediction_histogram = Histogram('eeg_prediction_duration_seconds', 'Prediction duration')
active_connections = Gauge('eeg_active_connections', 'Number of active connections')
seizure_detections = Counter('seizure_detections_total', 'Total number of seizure detections')
model_updates = Counter('model_updates_total', 'Total number of model updates')


# Request/Response models
class EEGPredictionRequest(BaseModel):
    """Request model for EEG prediction."""
    eeg_data: List[List[float]] = Field(..., description="EEG data (channels x time)")
    sampling_rate: int = Field(200, description="Sampling rate in Hz")
    channel_names: Optional[List[str]] = Field(None, description="EEG channel names")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    include_uncertainty: bool = Field(True, description="Include uncertainty estimates")
    include_interpretation: bool = Field(False, description="Include model interpretation")


class EEGPredictionResponse(BaseModel):
    """Response model for EEG prediction."""
    prediction_id: str
    timestamp: str
    predicted_class: str
    class_probabilities: Dict[str, float]
    confidence: float
    uncertainty: Optional[Dict[str, float]]
    clinical_metrics: Dict[str, float]
    interpretation: Optional[Dict[str, Any]]
    processing_time_ms: float
    warnings: List[str]


class StreamingRequest(BaseModel):
    """Request model for streaming EEG analysis."""
    patient_id: str
    sampling_rate: int = 200
    window_size: int = 50  # seconds
    overlap: float = 0.5  # 50% overlap


class ModelUpdateRequest(BaseModel):
    """Request model for model updates."""
    model_id: str
    update_type: str = Field("hot_reload", description="Type of update: hot_reload, gradual, canary")
    validation_required: bool = Field(True, description="Validate before deployment")


# Global model and preprocessor instances
class ModelService:
    """Service for model management and prediction."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model management
        self.model_serializer = ModelSerializer()
        self.model_validator = ModelValidator(config_path)
        self.current_model_id = None
        
        # Initialize monitoring system
        monitoring_config = MonitoringConfig(
            enable_metrics=True,
            enable_alerts=True,
            metrics_port=self.config.get('deployment', {}).get('monitoring', {}).get('metrics_port', 8001),
            latency_threshold_ms=self.config.get('deployment', {}).get('monitoring', {}).get('alert_thresholds', {}).get('latency_ms', 1000),
            drift_threshold=0.15
        )
        self.monitoring_system = create_monitoring_system(monitoring_config)
        self.performance_monitor = self.monitoring_system['performance_monitor']
        self.drift_detector = self.monitoring_system['drift_detector']
        self.resource_monitor = self.monitoring_system['resource_monitor']
        
        # Load optimized model if available
        self.model = self._load_optimized_model()
        
        # Initialize preprocessors
        self.eeg_preprocessor = EEGPreprocessor(self.config)
        self.spectrogram_generator = SpectrogramGenerator(self.config)
        
        # Initialize interpreters
        self.interpreter = ModelInterpreter(self.model, self.config, self.device)
        self.uncertainty_estimator = UncertaintyQuantification(self.model, self.config, self.device)
        
        # Clinical metrics
        self.clinical_metrics = ClinicalMetrics(self.config)
        
        # Cache for recent predictions
        self.prediction_cache = deque(maxlen=1000)
        
        # Distributed model server for scaling
        self.distributed_server = None
        if self.config.get('deployment', {}).get('distributed_serving', {}).get('enabled', False):
            self._setup_distributed_serving()
        
        # Redis for distributed caching (optional)
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_enabled = True
        except:
            logger.warning("Redis not available, using local cache only")
            self.redis_client = None
            self.redis_enabled = False
            
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_optimized_model(self):
        """Load optimized model if available, otherwise load regular model."""
        # Check for optimized models
        optimized_dir = Path("models/optimized")
        
        if optimized_dir.exists():
            # Look for ONNX or quantized models
            onnx_models = list(optimized_dir.rglob("*.onnx"))
            if onnx_models:
                logger.info(f"Loading optimized ONNX model: {onnx_models[0]}")
                opt_config = OptimizationConfig(optimization_level=1)
                return OptimizedModelWrapper(onnx_models[0], opt_config)
            
            # Look for quantized PyTorch models
            quantized_models = list(optimized_dir.rglob("*quantized*.pth"))
            if quantized_models:
                logger.info(f"Loading quantized model: {quantized_models[0]}")
                model = torch.load(quantized_models[0], map_location=self.device)
                model.eval()
                return model
        
        # Fallback to regular model loading
        return self._load_model()
            
    def _load_model(self) -> HMSEnsembleModel:
        """Load trained model."""
        # Try to load latest model from registry
        latest_model = self.model_serializer.registry.get_latest_version("HMSEnsembleModel")
        
        if latest_model:
            model, metadata = self.model_serializer.load_model(
                latest_model.model_id, 
                device=self.device
            )
            self.current_model_id = latest_model.model_id
            logger.info(f"Loaded model from registry: {latest_model.model_id}")
            return model
        
        # Fallback to checkpoint
        model = HMSEnsembleModel(self.config)
        
        checkpoint_path = Path("checkpoints/best_model.pth")
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded model from checkpoint")
        else:
            logger.warning("No checkpoint found, using untrained model")
            
        model.to(self.device)
        model.eval()
        
        return model
    
    def _setup_distributed_serving(self):
        """Setup distributed model serving for high throughput."""
        from .distributed_training import DistributedConfig, DistributedModelServer
        
        dist_config = DistributedConfig()
        model_path = Path("models/optimized/ensemble/model.onnx")
        
        if model_path.exists():
            self.distributed_server = DistributedModelServer(model_path, dist_config)
            asyncio.create_task(self.distributed_server.setup_serving(num_replicas=4))
            logger.info("Distributed model serving initialized")
        
    async def update_model(self, model_id: str, validate: bool = True) -> Dict[str, Any]:
        """Update the deployed model."""
        if validate:
            validation_results = self.model_validator.validate_model(model_id)
            if not validation_results['passed']:
                raise ValueError(f"Model validation failed: {validation_results}")
                
        # Load new model
        new_model, metadata = self.model_serializer.load_model(model_id, device=self.device)
        
        # Update model
        self.model = new_model
        self.current_model_id = model_id
        
        # Reinitialize interpreters
        self.interpreter = ModelInterpreter(self.model, self.config, self.device)
        self.uncertainty_estimator = UncertaintyQuantification(self.model, self.config, self.device)
        
        # Update metrics
        model_updates.inc()
        
        logger.info(f"Updated model to: {model_id}")
        
        return {
            'status': 'success',
            'model_id': model_id,
            'validation': validation_results if validate else None
        }
        
    @prediction_histogram.time()
    def predict(self, eeg_data: np.ndarray, 
               channel_names: List[str],
               include_uncertainty: bool = True,
               include_interpretation: bool = False) -> Dict:
        """Make prediction on EEG data."""
        start_time = time.time()
        
        # Preprocess EEG
        preprocessed_eeg = self.eeg_preprocessor.preprocess_eeg(
            eeg_data, 
            channel_names,
            apply_ica=False  # Faster for real-time
        )
        
        # Generate spectrogram
        spectrogram = self.spectrogram_generator.generate_multichannel_spectrogram(
            preprocessed_eeg,
            method='stft'
        )
        
        # Create 3D representation
        spectrogram_3d = self.spectrogram_generator.create_3d_spectrogram_representation(
            spectrogram
        )
        
        # Convert to tensors
        eeg_tensor = torch.FloatTensor(preprocessed_eeg).unsqueeze(0).to(self.device)
        spec_tensor = torch.FloatTensor(spectrogram_3d).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        
        # Use optimized model if available
        if hasattr(self.model, 'predict'):
            # OptimizedModelWrapper interface
            outputs = self.model.predict(eeg_tensor)
            probs = torch.softmax(torch.from_numpy(outputs), dim=1).numpy()[0]
        else:
            # Regular PyTorch model
            with torch.no_grad():
                outputs = self.model(eeg_tensor, spec_tensor)
                
            # Extract results
            probs = torch.softmax(outputs['logits'], dim=1).cpu().numpy()[0]
            
        pred_class = np.argmax(probs)
        confidence = float(probs[pred_class])
        
        # Get class probabilities
        class_probs = {
            self.config['classes'][i]: float(probs[i])
            for i in range(len(self.config['classes']))
        }
        
        # Uncertainty estimation
        uncertainty = None
        if include_uncertainty and not hasattr(self.model, 'predict'):
            uncertainty_results = self.uncertainty_estimator.monte_carlo_dropout(
                eeg_tensor, spec_tensor
            )
            uncertainty = {
                'epistemic': float(uncertainty_results['epistemic_uncertainty'][0]),
                'aleatoric': float(uncertainty_results['aleatoric_uncertainty'][0]),
                'total': float(uncertainty_results['total_uncertainty'][0])
            }
            
        # Interpretation
        interpretation = None
        if include_interpretation and not hasattr(self.model, 'predict'):
            # Get attention weights
            attention = self.interpreter.compute_attention_weights(eeg_tensor, spec_tensor)
            
            # Get integrated gradients for predicted class
            ig_eeg = self.interpreter.compute_integrated_gradients(
                eeg_tensor, pred_class, model_type='resnet'
            )
            
            interpretation = {
                'attention_weights': attention['temporal_attention'].tolist() if 'temporal_attention' in attention else None,
                'feature_importance': {
                    'channels': np.mean(np.abs(ig_eeg[0]), axis=1).tolist()
                }
            }
            
        # Clinical metrics
        is_seizure = 1 if pred_class == self.config['classes'].index('Seizure') else 0
        clinical = {
            'seizure_probability': class_probs.get('Seizure', 0.0),
            'requires_urgent_attention': is_seizure or confidence < 0.7,
            'recommended_action': self._get_recommended_action(pred_class, confidence)
        }
        
        # Update metrics
        prediction_counter.inc()
        if is_seizure:
            seizure_detections.inc()
            
        processing_time = (time.time() - start_time) * 1000
        
        # Record in monitoring system
        self.performance_monitor.record_inference(
            model_name=self.current_model_id or "ensemble",
            input_data=eeg_data,
            output=probs
        )
        
        # Check for drift
        if hasattr(self, '_reference_data'):
            drift_scores = self.drift_detector.check_drift(
                model_name=self.current_model_id or "ensemble",
                current_data=eeg_data,
                current_predictions=probs
            )
            if drift_scores.get('overall_drift', 0) > 0.15:
                logger.warning(f"Model drift detected: {drift_scores}")
        
        return {
            'predicted_class': self.config['classes'][pred_class],
            'class_probabilities': class_probs,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'clinical_metrics': clinical,
            'interpretation': interpretation,
            'processing_time_ms': processing_time,
            'model_id': self.current_model_id
        }
        
    def _get_recommended_action(self, pred_class: int, confidence: float) -> str:
        """Get recommended clinical action based on prediction."""
        class_name = self.config['classes'][pred_class]
        
        if class_name == 'Seizure':
            return "URGENT: Seizure detected. Immediate clinical intervention required."
        elif class_name in ['LPD', 'GPD']:
            return "Periodic discharges detected. Clinical review recommended within 1 hour."
        elif class_name in ['LRDA', 'GRDA']:
            return "Rhythmic activity detected. Monitor closely and review within 2 hours."
        elif confidence < 0.7:
            return "Low confidence prediction. Manual expert review recommended."
        else:
            return "No harmful activity detected. Continue routine monitoring."


# Initialize services
model_service = ModelService()

# Initialize streaming components
stream_processor = StreamProcessor(model_service, model_service.redis_client)
websocket_manager = WebSocketManager(stream_processor)
alert_manager = AlertManager()

# Initialize Kafka processor if configured
kafka_processor = None
if model_service.config.get('kafka', {}).get('enabled', False):
    kafka_processor = KafkaStreamProcessor(
        model_service,
        kafka_bootstrap_servers=model_service.config['kafka']['bootstrap_servers']
    )


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # Start Kafka processor if enabled
    if kafka_processor:
        await kafka_processor.start()
        asyncio.create_task(kafka_processor.process_messages())
        
    # Start alert processor
    asyncio.create_task(alert_manager.process_alerts())
    
    logger.info("API services started")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    if kafka_processor:
        await kafka_processor.stop()
        
    logger.info("API services stopped")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "HMS Brain Activity Classification API",
        "version": "1.0.0",
        "status": "operational",
        "model_id": model_service.current_model_id
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model_service.model is not None,
        "device": str(model_service.device),
        "services": {
            "redis": model_service.redis_enabled,
            "kafka": kafka_processor is not None,
            "streaming": len(stream_processor.sessions)
        }
    }
    
    # Check Redis connection
    if model_service.redis_client:
        try:
            await model_service.redis_client.ping()
            health_status["services"]["redis_connected"] = True
        except:
            health_status["services"]["redis_connected"] = False
            health_status["status"] = "degraded"
            
    return health_status


@app.post("/predict", response_model=EEGPredictionResponse)
async def predict_eeg(request: EEGPredictionRequest):
    """Predict brain activity from EEG data."""
    try:
        # Validate input
        eeg_array = np.array(request.eeg_data)
        if eeg_array.ndim != 2:
            raise HTTPException(400, "EEG data must be 2D array (channels x time)")
            
        # Use default channel names if not provided
        channel_names = request.channel_names or model_service.config['eeg']['channels'][:eeg_array.shape[0]]
        
        # Make prediction
        result = model_service.predict(
            eeg_array,
            channel_names,
            include_uncertainty=request.include_uncertainty,
            include_interpretation=request.include_interpretation
        )
        
        # Generate prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Cache prediction
        cache_entry = {
            'prediction_id': prediction_id,
            'timestamp': datetime.utcnow().isoformat(),
            'patient_id': request.patient_id,
            'result': result
        }
        model_service.prediction_cache.append(cache_entry)
        
        # Cache in Redis
        if model_service.redis_client:
            await model_service.redis_client.setex(
                f"prediction:{prediction_id}",
                3600,
                json.dumps(cache_entry, default=str)
            )
        
        # Check for alerts
        await alert_manager.check_prediction(result, None)
        
        # Build response
        warnings = []
        if eeg_array.shape[1] < model_service.config['dataset']['eeg_sampling_rate'] * 10:
            warnings.append("EEG segment shorter than recommended 10 seconds")
            
        response = EEGPredictionResponse(
            prediction_id=prediction_id,
            timestamp=datetime.utcnow().isoformat(),
            predicted_class=result['predicted_class'],
            class_probabilities=result['class_probabilities'],
            confidence=result['confidence'],
            uncertainty=result.get('uncertainty'),
            clinical_metrics=result['clinical_metrics'],
            interpretation=result.get('interpretation'),
            processing_time_ms=result['processing_time_ms'],
            warnings=warnings
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.post("/predict/file")
async def predict_from_file(
    file: UploadFile = File(...),
    patient_id: Optional[str] = None,
    include_uncertainty: bool = True
):
    """Predict from uploaded EEG file."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
            
        # Load EEG data based on file type
        if file.filename.endswith('.edf'):
            raw = mne.io.read_raw_edf(tmp_path, preload=True)
            eeg_data = raw.get_data()
            channel_names = raw.ch_names
            sampling_rate = raw.info['sfreq']
            
        elif file.filename.endswith('.h5'):
            with h5py.File(tmp_path, 'r') as f:
                eeg_data = f['eeg'][:]
                channel_names = [ch.decode() for ch in f['channels'][:]]
                sampling_rate = f.attrs.get('sampling_rate', 200)
                
        else:
            raise HTTPException(400, "Unsupported file format. Use .edf or .h5")
            
        # Make prediction
        result = model_service.predict(
            eeg_data,
            channel_names,
            include_uncertainty=include_uncertainty
        )
        
        # Clean up
        Path(tmp_path).unlink()
        
        return {
            'filename': file.filename,
            'patient_id': patient_id,
            'sampling_rate': sampling_rate,
            'duration_seconds': eeg_data.shape[1] / sampling_rate,
            'prediction': result
        }
        
    except Exception as e:
        logger.error(f"File prediction error: {str(e)}")
        raise HTTPException(500, f"File processing failed: {str(e)}")


@app.websocket("/ws/stream/{patient_id}")
async def websocket_stream(websocket: WebSocket, patient_id: str):
    """WebSocket endpoint for streaming EEG analysis."""
    client_id = str(uuid.uuid4())
    await websocket_manager.connect(websocket, client_id)
    
    try:
        await websocket_manager.handle_streaming(websocket, client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket_manager.disconnect(client_id)


@app.get("/predictions/recent")
async def get_recent_predictions(limit: int = 10, patient_id: Optional[str] = None):
    """Get recent predictions."""
    predictions = list(model_service.prediction_cache)
    
    if patient_id:
        predictions = [p for p in predictions if p.get('patient_id') == patient_id]
        
    # Sort by timestamp and limit
    predictions.sort(key=lambda x: x['timestamp'], reverse=True)
    predictions = predictions[:limit]
    
    return {
        'count': len(predictions),
        'predictions': predictions
    }


@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics."""
    return StreamingResponse(
        io.BytesIO(generate_latest()),
        media_type="text/plain"
    )


@app.get("/model/info")
async def get_model_info():
    """Get model information."""
    return {
        'model_type': 'Ensemble (ResNet1D-GRU + EfficientNet)',
        'classes': model_service.config['classes'],
        'input_requirements': {
            'eeg_channels': len(model_service.config['eeg']['channels']),
            'sampling_rate': model_service.config['dataset']['eeg_sampling_rate'],
            'minimum_duration_seconds': 10
        },
        'ensemble_config': model_service.config['models']['ensemble']
    }


@app.post("/batch/predict")
async def batch_predict(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Batch prediction on multiple files."""
    batch_id = str(uuid.uuid4())
    
    # Process in background
    background_tasks.add_task(
        process_batch,
        batch_id,
        files
    )
    
    return {
        'batch_id': batch_id,
        'num_files': len(files),
        'status': 'processing',
        'message': f'Batch job started. Check status at /batch/status/{batch_id}'
    }


async def process_batch(batch_id: str, files: List[UploadFile]):
    """Process batch of files in background."""
    results = []
    
    for file in files:
        try:
            # Process each file
            result = await predict_from_file(file)
            results.append({
                'filename': file.filename,
                'status': 'success',
                'result': result
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'status': 'error',
                'error': str(e)
            })
            
    # Store results (in Redis if available)
    if model_service.redis_enabled:
        model_service.redis_client.setex(
            f"batch:{batch_id}",
            3600,  # 1 hour TTL
            json.dumps(results)
        )


@app.get("/batch/status/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get batch processing status."""
    if model_service.redis_enabled:
        results = model_service.redis_client.get(f"batch:{batch_id}")
        if results:
            return {
                'batch_id': batch_id,
                'status': 'completed',
                'results': json.loads(results)
            }
            
    return {
        'batch_id': batch_id,
        'status': 'not_found',
        'message': 'Batch results not found or expired'
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 