"""
HMS EEG Classification System - Interactive Dashboard Application

This module provides a comprehensive web-based dashboard for:
- Real-time model performance monitoring
- Training progress visualization
- System resource monitoring
- Clinical alerts and notifications
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import redis
from pathlib import Path
import logging
from typing import Dict, List, Optional
import threading
import time

logger = logging.getLogger(__name__)


class DashboardApp:
    """Interactive dashboard for HMS EEG Classification System."""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 update_interval: int = 5000,  # milliseconds
                 data_dir: str = "logs"):
        """
        Initialize dashboard application.
        
        Args:
            redis_url: Redis connection URL for real-time data
            update_interval: Update interval in milliseconds
            data_dir: Directory containing log files
        """
        self.redis_url = redis_url
        self.update_interval = update_interval
        self.data_dir = Path(data_dir)
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_enabled = True
        except:
            logger.warning("Redis connection failed. Real-time features disabled.")
            self.redis_enabled = False
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                "https://use.fontawesome.com/releases/v5.15.4/css/all.css"
            ],
            suppress_callback_exceptions=True
        )
        
        # Setup layout
        self._setup_layout()
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1([
                        html.I(className="fas fa-brain mr-2"),
                        "HMS EEG Classification Dashboard"
                    ], className="text-center text-primary mb-4"),
                    html.P("Real-time monitoring and analysis system", 
                          className="text-center text-muted")
                ])
            ]),
            
            # Alert section
            dbc.Row([
                dbc.Col([
                    html.Div(id="alerts-container")
                ])
            ], className="mb-3"),
            
            # Navigation tabs
            dbc.Tabs([
                dbc.Tab(label="Overview", tab_id="overview"),
                dbc.Tab(label="Training Monitor", tab_id="training"),
                dbc.Tab(label="Model Performance", tab_id="performance"),
                dbc.Tab(label="Clinical Analysis", tab_id="clinical"),
                dbc.Tab(label="System Resources", tab_id="resources"),
                dbc.Tab(label="Predictions", tab_id="predictions")
            ], id="tabs", active_tab="overview"),
            
            # Tab content
            html.Div(id="tab-content", className="mt-4"),
            
            # Auto-refresh interval
            dcc.Interval(
                id="interval-component",
                interval=self.update_interval,
                n_intervals=0
            ),
            
            # Hidden div for storing data
            html.Div(id="hidden-data", style={"display": "none"})
            
        ], fluid=True)
    
    def _create_overview_tab(self):
        """Create overview tab content."""
        return dbc.Container([
            # Key metrics cards
            dbc.Row([
                dbc.Col([
                    self._create_metric_card(
                        "Total Predictions",
                        "0",
                        "fas fa-chart-line",
                        "primary",
                        "total-predictions"
                    )
                ], lg=3),
                dbc.Col([
                    self._create_metric_card(
                        "Model Accuracy",
                        "0%",
                        "fas fa-bullseye",
                        "success",
                        "model-accuracy"
                    )
                ], lg=3),
                dbc.Col([
                    self._create_metric_card(
                        "Active Alerts",
                        "0",
                        "fas fa-exclamation-triangle",
                        "warning",
                        "active-alerts"
                    )
                ], lg=3),
                dbc.Col([
                    self._create_metric_card(
                        "System Status",
                        "Healthy",
                        "fas fa-heartbeat",
                        "info",
                        "system-status"
                    )
                ], lg=3)
            ], className="mb-4"),
            
            # Charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Predictions Timeline"),
                        dbc.CardBody([
                            dcc.Graph(id="predictions-timeline")
                        ])
                    ])
                ], lg=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Class Distribution"),
                        dbc.CardBody([
                            dcc.Graph(id="class-distribution")
                        ])
                    ])
                ], lg=4)
            ], className="mb-4"),
            
            # Recent activity
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Activity"),
                        dbc.CardBody([
                            html.Div(id="recent-activity")
                        ])
                    ])
                ])
            ])
        ])
    
    def _create_training_tab(self):
        """Create training monitoring tab."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Training Progress"),
                        dbc.CardBody([
                            dcc.Graph(id="training-progress")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Loss Curves"),
                        dbc.CardBody([
                            dcc.Graph(id="loss-curves")
                        ])
                    ])
                ], lg=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Learning Rate Schedule"),
                        dbc.CardBody([
                            dcc.Graph(id="learning-rate")
                        ])
                    ])
                ], lg=6)
            ])
        ])
    
    def _create_performance_tab(self):
        """Create model performance tab."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Confusion Matrix"),
                        dbc.CardBody([
                            dcc.Graph(id="confusion-matrix")
                        ])
                    ])
                ], lg=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("ROC Curves"),
                        dbc.CardBody([
                            dcc.Graph(id="roc-curves")
                        ])
                    ])
                ], lg=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Per-Class Metrics"),
                        dbc.CardBody([
                            html.Div(id="class-metrics-table")
                        ])
                    ])
                ])
            ])
        ])
    
    def _create_clinical_tab(self):
        """Create clinical analysis tab."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Clinical Alerts Timeline"),
                        dbc.CardBody([
                            dcc.Graph(id="clinical-alerts-timeline")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Seizure Detection Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="seizure-analysis")
                        ])
                    ])
                ], lg=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Critical Findings"),
                        dbc.CardBody([
                            html.Div(id="critical-findings")
                        ])
                    ])
                ], lg=6)
            ])
        ])
    
    def _create_resources_tab(self):
        """Create system resources tab."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("CPU Usage"),
                        dbc.CardBody([
                            dcc.Graph(id="cpu-usage")
                        ])
                    ])
                ], lg=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("GPU Usage"),
                        dbc.CardBody([
                            dcc.Graph(id="gpu-usage")
                        ])
                    ])
                ], lg=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Memory Usage"),
                        dbc.CardBody([
                            dcc.Graph(id="memory-usage")
                        ])
                    ])
                ], lg=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("API Response Times"),
                        dbc.CardBody([
                            dcc.Graph(id="api-latency")
                        ])
                    ])
                ], lg=6)
            ])
        ])
    
    def _create_predictions_tab(self):
        """Create predictions analysis tab."""
        return dbc.Container([
            # Filters
            dbc.Row([
                dbc.Col([
                    html.Label("Date Range"),
                    dcc.DatePickerRange(
                        id="date-range",
                        start_date=datetime.now() - timedelta(days=7),
                        end_date=datetime.now(),
                        display_format="YYYY-MM-DD"
                    )
                ], lg=4),
                dbc.Col([
                    html.Label("Class Filter"),
                    dcc.Dropdown(
                        id="class-filter",
                        options=[
                            {"label": "All Classes", "value": "all"},
                            {"label": "Seizure", "value": "seizure"},
                            {"label": "LPD", "value": "lpd"},
                            {"label": "GPD", "value": "gpd"},
                            {"label": "LRDA", "value": "lrda"},
                            {"label": "GRDA", "value": "grda"},
                            {"label": "Other", "value": "other"}
                        ],
                        value="all"
                    )
                ], lg=4),
                dbc.Col([
                    html.Label("Confidence Threshold"),
                    dcc.Slider(
                        id="confidence-threshold",
                        min=0,
                        max=1,
                        step=0.05,
                        value=0.5,
                        marks={i/10: f"{i/10:.1f}" for i in range(11)}
                    )
                ], lg=4)
            ], className="mb-4"),
            
            # Predictions table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Predictions"),
                        dbc.CardBody([
                            html.Div(id="predictions-table")
                        ])
                    ])
                ])
            ])
        ])
    
    def _create_metric_card(self, title, value, icon, color, id_suffix):
        """Create a metric card component."""
        return dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className=f"{icon} fa-3x mb-3", 
                          style={"color": f"var(--bs-{color})"}),
                    html.H4(value, id=f"{id_suffix}-value", className="mb-0"),
                    html.P(title, className="text-muted mb-0")
                ], className="text-center")
            ])
        ], className="shadow-sm")
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            Output("tab-content", "children"),
            Input("tabs", "active_tab")
        )
        def render_tab_content(active_tab):
            """Render content based on active tab."""
            if active_tab == "overview":
                return self._create_overview_tab()
            elif active_tab == "training":
                return self._create_training_tab()
            elif active_tab == "performance":
                return self._create_performance_tab()
            elif active_tab == "clinical":
                return self._create_clinical_tab()
            elif active_tab == "resources":
                return self._create_resources_tab()
            elif active_tab == "predictions":
                return self._create_predictions_tab()
        
        @self.app.callback(
            [Output("alerts-container", "children"),
             Output("active-alerts-value", "children")],
            Input("interval-component", "n_intervals")
        )
        def update_alerts(n):
            """Update alerts section."""
            alerts = self._get_active_alerts()
            
            alert_components = []
            for alert in alerts[:5]:  # Show max 5 alerts
                severity_map = {
                    "critical": "danger",
                    "warning": "warning",
                    "info": "info"
                }
                color = severity_map.get(alert["severity"], "secondary")
                
                alert_components.append(
                    dbc.Alert([
                        html.I(className="fas fa-exclamation-circle mr-2"),
                        html.Strong(f"{alert['title']}: "),
                        alert["message"],
                        html.Small(f" - {alert['timestamp']}", className="float-right")
                    ], color=color, dismissable=True)
                )
            
            return alert_components, str(len(alerts))
        
        @self.app.callback(
            [Output("predictions-timeline", "figure"),
             Output("class-distribution", "figure")],
            Input("interval-component", "n_intervals")
        )
        def update_overview_charts(n):
            """Update overview charts."""
            # Get recent predictions data
            predictions_data = self._get_recent_predictions()
            
            # Predictions timeline
            timeline_fig = go.Figure()
            
            if predictions_data:
                df = pd.DataFrame(predictions_data)
                
                # Group by time and class
                timeline_data = df.groupby([
                    pd.Grouper(key='timestamp', freq='1H'),
                    'predicted_class'
                ]).size().reset_index(name='count')
                
                for class_name in df['predicted_class'].unique():
                    class_data = timeline_data[timeline_data['predicted_class'] == class_name]
                    timeline_fig.add_trace(go.Scatter(
                        x=class_data['timestamp'],
                        y=class_data['count'],
                        name=class_name,
                        mode='lines+markers'
                    ))
            
            timeline_fig.update_layout(
                title="Predictions Over Time",
                xaxis_title="Time",
                yaxis_title="Count",
                template="plotly_white",
                height=350
            )
            
            # Class distribution pie chart
            distribution_fig = go.Figure()
            
            if predictions_data:
                class_counts = df['predicted_class'].value_counts()
                distribution_fig.add_trace(go.Pie(
                    labels=class_counts.index,
                    values=class_counts.values,
                    hole=0.3
                ))
            
            distribution_fig.update_layout(
                title="Prediction Distribution",
                template="plotly_white",
                height=350
            )
            
            return timeline_fig, distribution_fig
        
        @self.app.callback(
            Output("confusion-matrix", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_confusion_matrix(n):
            """Update confusion matrix visualization."""
            cm_data = self._get_confusion_matrix_data()
            
            if cm_data is None:
                return go.Figure()
            
            classes = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
            
            fig = go.Figure(data=go.Heatmap(
                z=cm_data,
                x=classes,
                y=classes,
                colorscale='Blues',
                text=cm_data,
                texttemplate='%{text}',
                textfont={"size": 12}
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                template="plotly_white",
                height=400
            )
            
            return fig
        
        @self.app.callback(
            [Output("cpu-usage", "figure"),
             Output("gpu-usage", "figure"),
             Output("memory-usage", "figure"),
             Output("api-latency", "figure")],
            Input("interval-component", "n_intervals")
        )
        def update_resource_charts(n):
            """Update resource monitoring charts."""
            resource_data = self._get_resource_metrics()
            
            # CPU usage
            cpu_fig = go.Figure()
            cpu_fig.add_trace(go.Scatter(
                x=resource_data['timestamps'],
                y=resource_data['cpu_usage'],
                mode='lines',
                fill='tozeroy',
                name='CPU %'
            ))
            cpu_fig.update_layout(
                title="CPU Usage",
                yaxis_title="Usage %",
                template="plotly_white",
                height=300
            )
            
            # GPU usage
            gpu_fig = go.Figure()
            gpu_fig.add_trace(go.Scatter(
                x=resource_data['timestamps'],
                y=resource_data['gpu_memory'],
                mode='lines',
                fill='tozeroy',
                name='GPU Memory'
            ))
            gpu_fig.update_layout(
                title="GPU Memory Usage",
                yaxis_title="Memory (MB)",
                template="plotly_white",
                height=300
            )
            
            # Memory usage
            mem_fig = go.Figure()
            mem_fig.add_trace(go.Scatter(
                x=resource_data['timestamps'],
                y=resource_data['memory_usage'],
                mode='lines',
                fill='tozeroy',
                name='RAM'
            ))
            mem_fig.update_layout(
                title="Memory Usage",
                yaxis_title="Memory (GB)",
                template="plotly_white",
                height=300
            )
            
            # API latency
            latency_fig = go.Figure()
            latency_fig.add_trace(go.Box(
                y=resource_data['api_latencies'],
                name='Response Time',
                boxpoints='outliers'
            ))
            latency_fig.update_layout(
                title="API Response Times",
                yaxis_title="Latency (ms)",
                template="plotly_white",
                height=300
            )
            
            return cpu_fig, gpu_fig, mem_fig, latency_fig
    
    def _get_active_alerts(self) -> List[Dict]:
        """Get active alerts from Redis or file."""
        alerts = []
        
        if self.redis_enabled:
            try:
                # Get alerts from Redis
                alert_keys = self.redis_client.keys("alert:*")
                for key in alert_keys[-10:]:  # Get last 10 alerts
                    alert_data = self.redis_client.get(key)
                    if alert_data:
                        alerts.append(json.loads(alert_data))
            except Exception as e:
                logger.error(f"Error fetching alerts from Redis: {e}")
        
        # Fallback to file-based alerts
        if not alerts:
            alerts_file = self.data_dir / "alerts.json"
            if alerts_file.exists():
                with open(alerts_file, 'r') as f:
                    alerts = json.load(f)
        
        return sorted(alerts, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def _get_recent_predictions(self) -> List[Dict]:
        """Get recent predictions data."""
        predictions = []
        
        if self.redis_enabled:
            try:
                # Get from Redis
                pred_keys = self.redis_client.keys("prediction:*")
                for key in pred_keys[-100:]:  # Get last 100 predictions
                    pred_data = self.redis_client.get(key)
                    if pred_data:
                        predictions.append(json.loads(pred_data))
            except Exception as e:
                logger.error(f"Error fetching predictions from Redis: {e}")
        
        # Fallback to file
        if not predictions:
            pred_file = self.data_dir / "predictions.json"
            if pred_file.exists():
                with open(pred_file, 'r') as f:
                    predictions = json.load(f)
        
        return predictions
    
    def _get_confusion_matrix_data(self) -> Optional[np.ndarray]:
        """Get confusion matrix data."""
        cm_file = self.data_dir / "confusion_matrix.npy"
        if cm_file.exists():
            return np.load(cm_file)
        
        # Generate dummy data for demo
        return np.random.randint(0, 100, size=(6, 6))
    
    def _get_resource_metrics(self) -> Dict:
        """Get system resource metrics."""
        # Generate sample data for demo
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=50,
            freq='1min'
        )
        
        return {
            'timestamps': timestamps,
            'cpu_usage': np.random.uniform(20, 80, 50),
            'gpu_memory': np.random.uniform(1000, 4000, 50),
            'memory_usage': np.random.uniform(4, 16, 50),
            'api_latencies': np.random.exponential(50, 100)
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8050, debug: bool = False):
        """Run the dashboard application."""
        logger.info(f"Starting dashboard on {host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug) 