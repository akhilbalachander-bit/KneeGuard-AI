"""Request/response models for the KneeGuard API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .workload_model import AthleteInput

Surface = Literal["grass_dry", "grass_wet", "turf_dry", "turf_wet"]
Sex = Literal["male", "female", "unspecified"]


class AssessmentRequest(BaseModel):
    """The workload questionnaire, plus an optional prior scan to fuse in."""

    age: int = Field(17, ge=8, le=60, description="Athlete age in years")
    sex: Sex = Field(
        "unspecified",
        description="Used only for the documented sex difference in ACL incidence",
    )
    minutes_last_7d: int = Field(180, ge=0, le=1500, description="Match + training minutes, last 7 days")
    minutes_prior_28d: int = Field(600, ge=0, le=6000, description="Minutes over the previous 4 weeks")
    consecutive_days: int = Field(2, ge=0, le=21, description="Consecutive days played without a rest day")
    surface: Surface = Field("grass_dry", description="Surface played on most this week")
    soreness: int = Field(4, ge=1, le=10, description="Muscle soreness, 1 (none) to 10 (severe)")
    sleep_hours: float = Field(8.0, ge=0, le=16, description="Average sleep per night this week")
    prior_injury: bool = Field(False, description="Previous knee ligament injury")
    contact_events: int = Field(3, ge=0, le=50, description="Heavy collisions or tackles taken this week")
    slide_tackles: int = Field(1, ge=0, le=50, description="Sliding tackles made this week")

    scan_id: str | None = Field(
        None, description="ID returned by /api/scan, to fuse a movement scan into this assessment"
    )
    explain: bool = Field(True, description="Generate the plain-language explanation")

    def to_athlete(self) -> AthleteInput:
        return AthleteInput(
            age=self.age,
            sex=self.sex,
            minutes_last_7d=self.minutes_last_7d,
            minutes_prior_28d=self.minutes_prior_28d,
            consecutive_days=self.consecutive_days,
            surface=self.surface,
            soreness=self.soreness,
            sleep_hours=self.sleep_hours,
            prior_injury=self.prior_injury,
            contact_events=self.contact_events,
            slide_tackles=self.slide_tackles,
        )


class ScanResponse(BaseModel):
    scan_id: str
    scan: dict
    overlay_image: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    pose_model_ready: bool
    risk_model_ready: bool
    claude_configured: bool
    claude_model: str
