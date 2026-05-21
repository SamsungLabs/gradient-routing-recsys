from pathlib import Path

from .datasets import __init__
from .models import __init__


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def import_class(class_path: str):
    components = class_path.split(".")
    mod = __import__(".".join(components[:-1]), fromlist=[""])
    return getattr(mod, components[-1])
