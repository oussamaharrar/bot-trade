"""Gateway exports."""
from .paper import PaperGateway
from .ccxt_adapter import CCXTAdapter

__all__ = ["PaperGateway", "CCXTAdapter"]
