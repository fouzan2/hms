"""
Metrics Server for HMS EEG Classification System

This server exposes Prometheus metrics for monitoring the system.
"""

import os
import time
import logging
from flask import Flask, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import asyncio
from pathlib import Path

# Import monitoring components
from .performance_monitoring import MonitoringConfig, MetricsCollector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Initialize metrics collector
metrics_config = MonitoringConfig(
    enable_metrics=True,
    metrics_port=int(os.environ.get('METRICS_PORT', 8001)),
    log_interval=60
)
metrics_collector = MetricsCollector(metrics_config)


@app.route('/metrics')
def metrics():
    """Expose metrics for Prometheus scraping."""
    # Update system metrics
    metrics_collector.update_memory_usage()
    metrics_collector.update_gpu_utilization()
    
    # Generate latest metrics
    return Response(
        generate_latest(metrics_collector.registry),
        mimetype=CONTENT_TYPE_LATEST
    )


@app.route('/health')
def health():
    """Health check endpoint."""
    return {'status': 'healthy', 'timestamp': time.time()}


@app.route('/')
def index():
    """Index page with links to available endpoints."""
    return '''
    <html>
    <head><title>HMS EEG Metrics Server</title></head>
    <body>
        <h1>HMS EEG Classification Metrics Server</h1>
        <p>Available endpoints:</p>
        <ul>
            <li><a href="/metrics">/metrics</a> - Prometheus metrics</li>
            <li><a href="/health">/health</a> - Health check</li>
        </ul>
    </body>
    </html>
    '''


def run_server():
    """Run the metrics server."""
    port = int(os.environ.get('METRICS_PORT', 8001))
    logger.info(f"Starting metrics server on port {port}")
    app.run(host='0.0.0.0', port=port, threaded=True)


if __name__ == '__main__':
    run_server() 