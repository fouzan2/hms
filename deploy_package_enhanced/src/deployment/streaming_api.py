"""
Real-time streaming API for HMS brain activity classification.
Handles streaming EEG data with low-latency processing.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import numpy as np
import torch
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import redis.asyncio as redis
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import msgpack
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingStatus(Enum):
    """Status of streaming connection."""
    CONNECTED = "connected"
    PROCESSING = "processing"
    PAUSED = "paused"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class StreamingSession:
    """Streaming session information."""
    session_id: str
    patient_id: str
    start_time: datetime
    status: StreamingStatus
    processed_samples: int
    predictions_made: int
    last_prediction_time: Optional[datetime]
    buffer_size: int
    window_size: int
    overlap: float
    sampling_rate: int
    channels: List[str]
    

class StreamingBuffer:
    """Efficient circular buffer for streaming EEG data."""
    
    def __init__(self, max_size: int, n_channels: int):
        self.max_size = max_size
        self.n_channels = n_channels
        self.buffer = np.zeros((n_channels, max_size), dtype=np.float32)
        self.write_pos = 0
        self.size = 0
        
    def add_samples(self, samples: np.ndarray):
        """Add new samples to buffer."""
        n_samples = samples.shape[1]
        
        if n_samples > self.max_size:
            # If samples exceed buffer size, only keep latest
            samples = samples[:, -self.max_size:]
            n_samples = self.max_size
            
        # Calculate positions
        end_pos = self.write_pos + n_samples
        
        if end_pos <= self.max_size:
            # Simple case: no wrap around
            self.buffer[:, self.write_pos:end_pos] = samples
        else:
            # Wrap around case
            first_part = self.max_size - self.write_pos
            self.buffer[:, self.write_pos:] = samples[:, :first_part]
            self.buffer[:, :end_pos - self.max_size] = samples[:, first_part:]
            
        self.write_pos = end_pos % self.max_size
        self.size = min(self.size + n_samples, self.max_size)
        
    def get_window(self, window_size: int) -> Optional[np.ndarray]:
        """Get the latest window of data."""
        if self.size < window_size:
            return None
            
        if self.write_pos >= window_size:
            # No wrap around
            return self.buffer[:, self.write_pos - window_size:self.write_pos].copy()
        else:
            # Wrap around
            first_part = window_size - self.write_pos
            window = np.zeros((self.n_channels, window_size), dtype=np.float32)
            window[:, :first_part] = self.buffer[:, -(first_part):]
            window[:, first_part:] = self.buffer[:, :self.write_pos]
            return window
            
    def clear(self):
        """Clear the buffer."""
        self.buffer.fill(0)
        self.write_pos = 0
        self.size = 0


class StreamProcessor:
    """Process streaming EEG data in real-time."""
    
    def __init__(self, model_service, redis_client: Optional[redis.Redis] = None):
        self.model_service = model_service
        self.redis_client = redis_client
        self.sessions: Dict[str, StreamingSession] = {}
        self.buffers: Dict[str, StreamingBuffer] = {}
        self.processing_locks: Dict[str, asyncio.Lock] = {}
        
    async def create_session(self, 
                           patient_id: str,
                           sampling_rate: int = 200,
                           window_size: int = 50,
                           overlap: float = 0.5,
                           channels: Optional[List[str]] = None) -> StreamingSession:
        """Create a new streaming session."""
        
        session_id = str(uuid.uuid4())
        
        # Use default channels if not provided
        if channels is None:
            channels = self.model_service.config['eeg']['channels']
            
        session = StreamingSession(
            session_id=session_id,
            patient_id=patient_id,
            start_time=datetime.utcnow(),
            status=StreamingStatus.CONNECTED,
            processed_samples=0,
            predictions_made=0,
            last_prediction_time=None,
            buffer_size=window_size * sampling_rate * 2,  # 2x window for buffering
            window_size=window_size * sampling_rate,
            overlap=overlap,
            sampling_rate=sampling_rate,
            channels=channels
        )
        
        # Initialize buffer
        self.buffers[session_id] = StreamingBuffer(
            max_size=session.buffer_size,
            n_channels=len(channels)
        )
        
        # Create processing lock
        self.processing_locks[session_id] = asyncio.Lock()
        
        # Store session
        self.sessions[session_id] = session
        
        # Cache in Redis if available
        if self.redis_client:
            await self.redis_client.setex(
                f"session:{session_id}",
                3600,  # 1 hour TTL
                json.dumps(asdict(session), default=str)
            )
            
        logger.info(f"Created streaming session: {session_id}")
        return session
        
    async def process_chunk(self, 
                          session_id: str, 
                          samples: np.ndarray,
                          timestamp: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Process a chunk of streaming data."""
        
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
            
        buffer = self.buffers.get(session_id)
        if not buffer:
            raise ValueError(f"Buffer for session {session_id} not found")
            
        # Update session status
        session.status = StreamingStatus.PROCESSING
        session.processed_samples += samples.shape[1]
        
        # Add samples to buffer
        buffer.add_samples(samples)
        
        # Check if we have enough data for prediction
        window = buffer.get_window(session.window_size)
        if window is None:
            return None
            
        # Process with lock to prevent concurrent processing
        async with self.processing_locks[session_id]:
            # Make prediction
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.model_service.predict,
                window,
                session.channels,
                True,  # include_uncertainty
                False  # include_interpretation (faster without)
            )
            
            # Update session
            session.predictions_made += 1
            session.last_prediction_time = datetime.utcnow()
            
            # Add metadata
            result['session_id'] = session_id
            result['timestamp'] = timestamp or time.time()
            result['sample_position'] = session.processed_samples
            
            # Calculate sliding window position for next prediction
            slide_samples = int(session.window_size * (1 - session.overlap))
            
            # Store result in Redis if available
            if self.redis_client:
                await self._cache_prediction(session_id, result)
                
            return result
            
    async def _cache_prediction(self, session_id: str, result: Dict[str, Any]):
        """Cache prediction result in Redis."""
        key = f"predictions:{session_id}"
        
        # Store as time series
        await self.redis_client.zadd(
            key,
            {json.dumps(result, default=str): result['timestamp']}
        )
        
        # Keep only last hour of predictions
        one_hour_ago = time.time() - 3600
        await self.redis_client.zremrangebyscore(key, 0, one_hour_ago)
        
        # Set expiry
        await self.redis_client.expire(key, 3600)
        
    async def get_session_predictions(self, 
                                    session_id: str,
                                    start_time: Optional[float] = None,
                                    end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get historical predictions for a session."""
        
        if not self.redis_client:
            return []
            
        key = f"predictions:{session_id}"
        
        # Get predictions in time range
        start = start_time or 0
        end = end_time or time.time()
        
        results = await self.redis_client.zrangebyscore(
            key, start, end, withscores=True
        )
        
        predictions = []
        for data, score in results:
            pred = json.loads(data)
            pred['timestamp'] = score
            predictions.append(pred)
            
        return predictions
        
    async def close_session(self, session_id: str):
        """Close a streaming session."""
        
        session = self.sessions.get(session_id)
        if session:
            session.status = StreamingStatus.DISCONNECTED
            
            # Clean up resources
            if session_id in self.buffers:
                del self.buffers[session_id]
            if session_id in self.processing_locks:
                del self.processing_locks[session_id]
                
            # Update Redis if available
            if self.redis_client:
                await self.redis_client.delete(f"session:{session_id}")
                
            del self.sessions[session_id]
            
        logger.info(f"Closed streaming session: {session_id}")


class KafkaStreamProcessor:
    """Process EEG streams from Kafka for scalable deployment."""
    
    def __init__(self, 
                model_service,
                kafka_bootstrap_servers: str = "localhost:9092",
                input_topic: str = "eeg-raw",
                output_topic: str = "eeg-predictions"):
        
        self.model_service = model_service
        self.kafka_servers = kafka_bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.consumer = None
        self.producer = None
        self.processing = False
        
    async def start(self):
        """Start Kafka consumer and producer."""
        
        self.consumer = AIOKafkaConsumer(
            self.input_topic,
            bootstrap_servers=self.kafka_servers,
            value_deserializer=lambda m: msgpack.unpackb(m, raw=False),
            group_id="eeg-processor-group",
            enable_auto_commit=True,
            auto_commit_interval_ms=1000
        )
        
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda v: msgpack.packb(v, use_bin_type=True)
        )
        
        await self.consumer.start()
        await self.producer.start()
        
        self.processing = True
        logger.info("Kafka stream processor started")
        
    async def stop(self):
        """Stop Kafka consumer and producer."""
        
        self.processing = False
        
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()
            
        logger.info("Kafka stream processor stopped")
        
    async def process_messages(self):
        """Process messages from Kafka."""
        
        try:
            async for msg in self.consumer:
                if not self.processing:
                    break
                    
                try:
                    # Extract message data
                    data = msg.value
                    patient_id = data.get('patient_id')
                    samples = np.array(data.get('samples'))
                    channels = data.get('channels')
                    timestamp = data.get('timestamp')
                    
                    # Process EEG data
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.model_service.predict,
                        samples,
                        channels,
                        True,  # include_uncertainty
                        False  # include_interpretation
                    )
                    
                    # Prepare output message
                    output_msg = {
                        'patient_id': patient_id,
                        'timestamp': timestamp,
                        'prediction': result['predicted_class'],
                        'probabilities': result['class_probabilities'],
                        'confidence': result['confidence'],
                        'uncertainty': result.get('uncertainty'),
                        'clinical_metrics': result['clinical_metrics'],
                        'processing_time_ms': result['processing_time_ms']
                    }
                    
                    # Send to output topic
                    await self.producer.send_and_wait(
                        self.output_topic,
                        value=output_msg,
                        key=patient_id.encode() if patient_id else None
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing Kafka message: {e}")
                    
        except Exception as e:
            logger.error(f"Kafka processing error: {e}")


class WebSocketManager:
    """Manage WebSocket connections for real-time streaming."""
    
    def __init__(self, stream_processor: StreamProcessor):
        self.stream_processor = stream_processor
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_sessions: Dict[str, str] = {}  # websocket_id -> session_id
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")
        
    async def disconnect(self, client_id: str):
        """Handle WebSocket disconnection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
        # Close associated session
        if client_id in self.connection_sessions:
            session_id = self.connection_sessions[client_id]
            await self.stream_processor.close_session(session_id)
            del self.connection_sessions[client_id]
            
        logger.info(f"WebSocket disconnected: {client_id}")
        
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client."""
        websocket = self.active_connections.get(client_id)
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                await self.disconnect(client_id)
                
    async def broadcast(self, message: Dict[str, Any], exclude: Optional[List[str]] = None):
        """Broadcast message to all connected clients."""
        exclude = exclude or []
        disconnected = []
        
        for client_id, websocket in self.active_connections.items():
            if client_id not in exclude:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {e}")
                    disconnected.append(client_id)
                    
        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect(client_id)
            
    async def handle_streaming(self, websocket: WebSocket, client_id: str):
        """Handle streaming EEG data from WebSocket."""
        
        try:
            # Wait for initialization message
            init_msg = await websocket.receive_json()
            
            if init_msg.get('type') != 'init':
                await websocket.send_json({
                    'type': 'error',
                    'message': 'Expected init message'
                })
                return
                
            # Create streaming session
            session = await self.stream_processor.create_session(
                patient_id=init_msg.get('patient_id', 'unknown'),
                sampling_rate=init_msg.get('sampling_rate', 200),
                window_size=init_msg.get('window_size', 50),
                overlap=init_msg.get('overlap', 0.5),
                channels=init_msg.get('channels')
            )
            
            self.connection_sessions[client_id] = session.session_id
            
            # Send session info
            await websocket.send_json({
                'type': 'session_created',
                'session_id': session.session_id,
                'config': {
                    'sampling_rate': session.sampling_rate,
                    'window_size': session.window_size // session.sampling_rate,
                    'overlap': session.overlap,
                    'channels': session.channels
                }
            })
            
            # Process streaming data
            while True:
                try:
                    # Receive data with timeout
                    data = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=30.0  # 30 second timeout
                    )
                    
                    if data.get('type') == 'eeg_data':
                        # Process EEG chunk
                        samples = np.array(data['samples'])
                        timestamp = data.get('timestamp')
                        
                        result = await self.stream_processor.process_chunk(
                            session.session_id,
                            samples,
                            timestamp
                        )
                        
                        if result:
                            # Send prediction result
                            await websocket.send_json({
                                'type': 'prediction',
                                'data': result
                            })
                            
                    elif data.get('type') == 'pause':
                        session.status = StreamingStatus.PAUSED
                        await websocket.send_json({
                            'type': 'status',
                            'status': 'paused'
                        })
                        
                    elif data.get('type') == 'resume':
                        session.status = StreamingStatus.PROCESSING
                        await websocket.send_json({
                            'type': 'status',
                            'status': 'resumed'
                        })
                        
                    elif data.get('type') == 'close':
                        break
                        
                except asyncio.TimeoutError:
                    # Send heartbeat
                    await websocket.send_json({
                        'type': 'heartbeat',
                        'timestamp': time.time()
                    })
                    
                except WebSocketDisconnect:
                    break
                    
                except Exception as e:
                    logger.error(f"Error in streaming handler: {e}")
                    await websocket.send_json({
                        'type': 'error',
                        'message': str(e)
                    })
                    
        finally:
            await self.disconnect(client_id)


class AlertManager:
    """Manage real-time alerts for critical findings."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.alert_queue = asyncio.Queue()
        self.alert_history = deque(maxlen=1000)
        
    async def check_prediction(self, prediction: Dict[str, Any], session: StreamingSession):
        """Check prediction for alert conditions."""
        
        alert_triggered = False
        alert_reasons = []
        
        # Check for seizure
        if prediction.get('predicted_class') == 'Seizure':
            alert_triggered = True
            alert_reasons.append("Seizure detected")
            
        # Check for high uncertainty
        uncertainty = prediction.get('uncertainty', {})
        if uncertainty.get('total', 0) > 0.5:
            alert_triggered = True
            alert_reasons.append("High uncertainty in prediction")
            
        # Check for low confidence
        if prediction.get('confidence', 1.0) < 0.6:
            alert_triggered = True
            alert_reasons.append("Low confidence prediction")
            
        if alert_triggered:
            alert = {
                'alert_id': str(uuid.uuid4()),
                'timestamp': datetime.utcnow().isoformat(),
                'patient_id': session.patient_id,
                'session_id': session.session_id,
                'prediction': prediction,
                'reasons': alert_reasons,
                'severity': 'critical' if 'Seizure' in alert_reasons else 'warning'
            }
            
            await self.alert_queue.put(alert)
            self.alert_history.append(alert)
            
            # Send webhook notification if configured
            if self.webhook_url:
                await self._send_webhook(alert)
                
    async def _send_webhook(self, alert: Dict[str, Any]):
        """Send alert via webhook."""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=alert,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Webhook failed: {response.status}")
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            
    async def process_alerts(self):
        """Process alert queue."""
        while True:
            try:
                alert = await self.alert_queue.get()
                logger.warning(f"ALERT: {alert['reasons']} for patient {alert['patient_id']}")
                # Additional alert processing can be added here
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
                await asyncio.sleep(1) 