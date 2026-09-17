"""Face and Hair Parsing subsystem."""
from .base import ParserBackend, ParsedFaceMasks
from .bisenet import BiSeNetParser

__all__ = ["ParserBackend", "ParsedFaceMasks", "BiSeNetParser"]
