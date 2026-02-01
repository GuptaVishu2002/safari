# Import common utilities
try:
    from .config import *
except ImportError:
    pass

try:
    from .train import *
except ImportError:
    pass

try:
    from .optim import *
except ImportError:
    pass

try:
    from .registry import *
except ImportError:
    pass

# Auto-discover modules
import os
import glob

__all__ = []

_current_dir = os.path.dirname(__file__)
for _module_path in glob.glob(os.path.join(_current_dir, "*.py")):
    _module_name = os.path.basename(_module_path)[:-3]
    if _module_name not in ["__init__", "__pycache__"]:
        try:
            exec(f"from .{_module_name} import *")
            __all__.append(_module_name)
        except ImportError:
            pass

# Also check for subdirectories
for _subdir in glob.glob(os.path.join(_current_dir, "*/")):
    _subdir_name = os.path.basename(_subdir.rstrip('/'))
    if _subdir_name not in ["__pycache__"]:
        try:
            exec(f"from . import {_subdir_name}")
            __all__.append(_subdir_name)
        except ImportError:
            pass
