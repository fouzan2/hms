#!/usr/bin/env python3
"""
Run the HMS EEG Visualization Dashboard

This script launches the interactive web-based dashboard for monitoring
the HMS EEG Classification System.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.visualization.dashboard.app import DashboardApp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for dashboard server."""
    parser = argparse.ArgumentParser(description='HMS EEG Visualization Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8050, help='Port to bind to')
    parser.add_argument('--redis-url', default=os.getenv('REDIS_URL', 'redis://localhost:6379'),
                       help='Redis URL for real-time data')
    parser.add_argument('--data-dir', default='logs', help='Directory containing log files')
    parser.add_argument('--update-interval', type=int, default=5000,
                       help='Update interval in milliseconds')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    logger.info("Starting HMS EEG Visualization Dashboard...")
    logger.info(f"Configuration:")
    logger.info(f"  Host: {args.host}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Redis URL: {args.redis_url}")
    logger.info(f"  Data Directory: {args.data_dir}")
    logger.info(f"  Update Interval: {args.update_interval}ms")
    
    # Create dashboard app
    dashboard = DashboardApp(
        redis_url=args.redis_url,
        update_interval=args.update_interval,
        data_dir=args.data_dir
    )
    
    # Run the dashboard
    try:
        dashboard.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except KeyboardInterrupt:
        logger.info("Dashboard server stopped by user")
    except Exception as e:
        logger.error(f"Dashboard server error: {e}")
        raise


if __name__ == '__main__':
    main() 