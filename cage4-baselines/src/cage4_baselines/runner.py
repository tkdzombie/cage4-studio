"""Compatibility import for v0.3.0 callers; use the documented package path."""
from .experiments.runner import main, run, save_checkpoint

if __name__ == "__main__":
    main()
