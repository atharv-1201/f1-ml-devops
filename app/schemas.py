from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class PredictionRequest(BaseModel):
    driver: str = Field(..., json_schema_extra={"example": "VER"}, description="3-letter F1 Driver Code (e.g. VER, NOR, LEC, HAM)")
    qualifying_time: Optional[float] = Field(None, json_schema_extra={"example": 70.669}, description="Qualifying time in seconds")
    rain_probability: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Rain probability between 0.0 and 1.0")
    temperature: Optional[float] = Field(22.0, description="Ambient temperature in degrees Celsius")

class PredictionResponse(BaseModel):
    driver: str
    team: str
    qualifying_time: float
    rain_probability: float
    temperature: float
    predicted_lap_time: float
    unit: str = "seconds"

class DriverStanding(BaseModel):
    position: int
    driver: str
    team: str
    qualifying_time: Optional[float]
    predicted_lap_time: float
    gap_to_leader: float

class GridPredictionRequest(BaseModel):
    rain_probability: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Rain probability (0.0 to 1.0)")
    temperature: Optional[float] = Field(22.0, description="Track/Ambient temperature in Celsius")
    custom_qualifying: Optional[Dict[str, float]] = Field(None, description="Optional map of Driver Code to Qualifying time in seconds")

class GridPredictionResponse(BaseModel):
    race: str
    weather: Dict[str, float]
    podium: List[str]
    standings: List[DriverStanding]

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
