"""
Lab Automation Module Configuration
"""

# Equipment default settings
EQUIPMENT_DEFAULTS = {
    'robot': {
        'default_speed': 50.0,  # %
        'default_acceleration': 1.0,
        'connection_timeout': 30.0,
    },
    'furnace': {
        'max_ramp_rate': 10.0,  # °C/min
        'safety_margin': 50.0,  # °C below max temp
    },
    'xrd': {
        'default_wavelength': 1.5406,  # Cu K-alpha
        'default_scan_rate': 2.0,  # °/min
    }
}

# Synthesis default parameters
SYNTHESIS_DEFAULTS = {
    'solid_state': {
        'calcination_temp': 800,
        'calcination_time': 4,
        'heating_rate': 5,
        'num_regrinds': 2,
    },
    'sol_gel': {
        'aging_time': 24,
        'drying_temp': 80,
        'calcination_temp': 450,
    },
    'thin_film': {
        'annealing_temp': 400,
        'annealing_time': 60,
    }
}

# Control system defaults
CONTROL_DEFAULTS = {
    'pid': {
        'kp': 1.0,
        'ki': 0.0,
        'kd': 0.0,
        'sample_time': 1.0,
        'output_limits': (0, 100),
    },
    'mpc': {
        'horizon': 10,
        'control_horizon': 5,
    }
}

# LIMS configuration
LIMS_CONFIG = {
    'connection_timeout': 30.0,
    'retry_attempts': 3,
    'batch_size': 100,
    'audit_enabled': True,
}

# ROS2 configuration
ROS2_CONFIG = {
    'namespace': '/lab_automation',
    'qos_depth': 10,
    'heartbeat_interval': 1.0,
}

# Data directories
DATA_DIRECTORIES = {
    'raw_data': './data/raw',
    'processed_data': './data/processed',
    'recipes': './data/recipes',
    'logs': './logs',
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': './logs/lab_automation.log',
            'formatter': 'standard',
            'level': 'DEBUG',
        },
    },
    'loggers': {
        'lab_automation': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}
