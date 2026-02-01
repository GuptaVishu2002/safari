# Import common ops
try:
    from .fft_conv import *
except ImportError:
    pass

try:
    from .flash_attention import *
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
