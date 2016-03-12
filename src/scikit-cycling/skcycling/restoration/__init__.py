"""
The :mod:`skcycling.restoration` module include denoising methods.
"""

from .denoise import outliers_rejection
from .denoise import moving_average

__all__ = [
    'outliers_rejection',
    'moving_average',
]
