"""
Configuration settings for the clustering and migration analysis system.
Loads database credentials from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
LOGS_DIR = BASE_DIR / 'logs'
NOTEBOOKS_DIR = BASE_DIR / 'notebooks'

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# Database configuration
DB_CONFIG = {
    'user': os.getenv('DB_USER', 'OCDS_00062050'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST', 'MSBSPSS'),
    'port': int(os.getenv('DB_PORT', 1521)),
    'service_name': os.getenv('DB_SERVICE_NAME', 'MSBSPSS')
}

# Analysis configuration
ANALYSIS_CONFIG = {
    'start_date': '2024-12-31',
    'new_client_days': 90,
    'outlier_percentile_lower': 0.05,
    'outlier_percentile_upper': 0.95,
    'cluster_quantiles': [0.25, 0.5, 0.75]
}

# Cluster mapping
CLUSTER_NAMES = {
    0: 'VIP',
    1: 'Лояльные',
    2: 'Обычные',
    3: 'Спящие'
}

# Segment types
SEGMENT_TYPES = ['MB_IP', 'MB_TOO', 'SB', 'NEW_CLIENTS', 'SOC', 'CASH']

# Visualization settings
VIZ_CONFIG = {
    'style': 'seaborn-v0_8-darkgrid',
    'palette': 'husl',
    'dpi': 300,
    'figsize_dynamics': (16, 14),
    'figsize_migration': (16, 12)
}

# ML Configuration
ML_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'churn_definition': 'downgrade',  # 'downgrade' or 'exit'
    'feature_window_months': 3,  # Number of months for rolling features
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'filename': str(LOGS_DIR / 'clustering_analysis.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console', 'file']
    }
}
