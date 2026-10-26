from pydantic import BaseModel, Field

class AnalysisResponse(BaseModel):
    prediction: dict
    guidance: str
    matched_materials: list[dict] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    provider: str
