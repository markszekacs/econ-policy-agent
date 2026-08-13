"""Shared Pydantic schemas for API layer."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
