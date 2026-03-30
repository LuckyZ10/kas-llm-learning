"""
Pydantic Schemas for Screening Results
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ScreeningResultBase(BaseModel):
    structure_id: str
    formula: str
    space_group: Optional[str] = None
    lattice_parameters: Dict[str, float] = Field(default_factory=dict)


class ScreeningResultResponse(ScreeningResultBase):
    id: str
    structure_file: Optional[str]
    formation_energy: Optional[float]
    band_gap: Optional[float]
    ionic_conductivity: Optional[float]
    bulk_modulus: Optional[float]
    shear_modulus: Optional[float]
    energy_above_hull: Optional[float]
    properties: Dict[str, Any]
    dft_functional: Optional[str]
    dft_accuracy: Optional[str]
    ml_potential_used: bool
    ml_model_version: Optional[str]
    project_id: Optional[str]
    workflow_id: Optional[str]
    pareto_rank: Optional[int]
    overall_score: Optional[float]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    calculated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class ScreeningResultList(BaseModel):
    items: List[ScreeningResultResponse]
    total: int
    page: int
    page_size: int


class PropertyRange(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None


class ScreeningFilter(BaseModel):
    project_id: Optional[str] = None
    formula_contains: Optional[str] = None
    property_ranges: Dict[str, PropertyRange] = Field(default_factory=dict)
    ml_only: bool = False
    dft_only: bool = False


class ScreeningComparison(BaseModel):
    structure_ids: List[str] = Field(..., min_length=2, max_length=10)
    properties: List[str] = Field(default_factory=lambda: ["formation_energy", "band_gap", "ionic_conductivity"])
