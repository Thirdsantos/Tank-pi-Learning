"""Aqua Learning - Aquarium Parameter Prediction System."""

__version__ = "1.0.0"
__author__ = "Aqua Learning Team"

# Make key modules easily importable
from . import config
from . import data  
from . import models

__all__ = ['config', 'data', 'models']

