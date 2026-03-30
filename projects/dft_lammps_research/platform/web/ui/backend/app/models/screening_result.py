"""
Screening Result Model - High-throughput screening results
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import String, DateTime, ForeignKey, Integer, Float, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.database import Base


class ScreeningResult(Base):
    __tablename__ = "screening_results"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    structure_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    
    # Structure info
    formula: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    space_group: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    lattice_parameters: Mapped[Dict[str, float]] = mapped_column(JSON, default=dict)  # a, b, c, alpha, beta, gamma
    
    # File paths
    structure_file: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    
    # Calculated properties
    formation_energy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # eV/atom
    band_gap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # eV
    ionic_conductivity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # S/cm
    bulk_modulus: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # GPa
    shear_modulus: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # GPa
    energy_above_hull: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # eV/atom
    
    # Additional properties (flexible storage)
    properties: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Calculation metadata
    dft_functional: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    dft_accuracy: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # low, normal, high
    ml_potential_used: Mapped[bool] = mapped_column(Boolean, default=False)
    ml_model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Source
    project_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("projects.id"), nullable=True)
    workflow_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("workflows.id"), nullable=True)
    
    # Ranking
    pareto_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    overall_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    calculated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "structure_id": self.structure_id,
            "formula": self.formula,
            "space_group": self.space_group,
            "lattice_parameters": self.lattice_parameters,
            "structure_file": self.structure_file,
            "formation_energy": self.formation_energy,
            "band_gap": self.band_gap,
            "ionic_conductivity": self.ionic_conductivity,
            "bulk_modulus": self.bulk_modulus,
            "shear_modulus": self.shear_modulus,
            "energy_above_hull": self.energy_above_hull,
            "properties": self.properties,
            "dft_functional": self.dft_functional,
            "dft_accuracy": self.dft_accuracy,
            "ml_potential_used": self.ml_potential_used,
            "ml_model_version": self.ml_model_version,
            "project_id": self.project_id,
            "workflow_id": self.workflow_id,
            "pareto_rank": self.pareto_rank,
            "overall_score": self.overall_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "calculated_at": self.calculated_at.isoformat() if self.calculated_at else None,
        }
