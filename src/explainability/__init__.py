"""Explainability package initialization"""

# Imported lazily: shap and lime are heavy optional dependencies, and importing
# one should not require the other to be installed.

__all__ = [
    'SHAPAnalyzer',
    'LIMEAnalyzer',
]


def __getattr__(name):
    if name == 'SHAPAnalyzer':
        from .shap_analysis import SHAPAnalyzer
        return SHAPAnalyzer
    if name == 'LIMEAnalyzer':
        from .lime_analysis import LIMEAnalyzer
        return LIMEAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
