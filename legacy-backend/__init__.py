"""
Brain# Core modules
from . import data

# Optional imports that may have dependencies
try:
    from . import models
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

try:
    from . import training
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

try:
    from . import inference
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False

try:
    from . import visualization
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    from . import reports
    REPORTS_AVAILABLE = True
except ImportError:
    REPORTS_AVAILABLE = False

__all__ = ['data']

# Add available modules to exports
if MODELS_AVAILABLE:
    __all__.append('models')
if TRAINING_AVAILABLE:
    __all__.append('training')
if INFERENCE_AVAILABLE:
    __all__.append('inference')
if VISUALIZATION_AVAILABLE:
    __all__.append('visualization')
if REPORTS_AVAILABLE:
    __all__.append('reports')tor

A comprehensive AI-powered system for brain tumor detection and analysis.
"""

__version__ = "1.0.0"
__author__ = "Brain Tumor Detection Team"
__email__ = "team@braintumordetector.com"

# Core modules
from . import data
from . import models  
from . import training
from . import inference
from . import visualization
from . import reports

__all__ = [
    'data',
    'models', 
    'training',
    'inference',
    'visualization',
    'reports'
]