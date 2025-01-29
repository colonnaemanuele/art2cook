"""Definitions for the objects used by our resource endpoints."""
from pydantic import BaseModel


class DataPayload(BaseModel):
    '''Data payload'''
    digitized_artwork_path: str