"""Sequence models including H3, Hyena, and other state space models."""

# Import all sequence models
try:
    from .hyena import *
except ImportError:
    pass

try:
    from .h3 import *
except ImportError:
    pass

try:
    from .long_conv import *
except ImportError:
    pass

try:
    from .ssm import *
except ImportError:
    pass

try:
    from .mamba import *
except ImportError:
    pass

# Import any other modules in this directory
import os
import glob

__all__ = []

# Auto-discover and import all Python modules in this directory
_current_dir = os.path.dirname(__file__)
for _module_path in glob.glob(os.path.join(_current_dir, "*.py")):
    _module_name = os.path.basename(_module_path)[:-3]
    if _module_name not in ["__init__", "__pycache__"]:
        try:
            exec(f"from .{_module_name} import *")
            __all__.append(_module_name)
        except ImportError:
            pass
