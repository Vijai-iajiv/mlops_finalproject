from fastapi import FastAPI
from pydantic import BaseModel, Field

from predict import predict

app = FastAPI(title="Breast Cancer Prediction API")

# sklearn's breast_cancer dataset uses space-separated column names (e.g. "mean radius").
# Pydantic field names can't contain spaces, so we expose underscored aliases here and
# map them back to the original names before calling the model.
FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
    "mean fractal dimension", "radius error", "texture error", "perimeter error",
    "area error", "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error", "worst radius",
    "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry",
    "worst fractal dimension",
]


class PredictionRequest(BaseModel):
    mean_radius: float = Field(alias="mean radius")
    mean_texture: float = Field(alias="mean texture")
    mean_perimeter: float = Field(alias="mean perimeter")
    mean_area: float = Field(alias="mean area")
    mean_smoothness: float = Field(alias="mean smoothness")
    mean_compactness: float = Field(alias="mean compactness")
    mean_concavity: float = Field(alias="mean concavity")
    mean_concave_points: float = Field(alias="mean concave points")
    mean_symmetry: float = Field(alias="mean symmetry")
    mean_fractal_dimension: float = Field(alias="mean fractal dimension")
    radius_error: float = Field(alias="radius error")
    texture_error: float = Field(alias="texture error")
    perimeter_error: float = Field(alias="perimeter error")
    area_error: float = Field(alias="area error")
    smoothness_error: float = Field(alias="smoothness error")
    compactness_error: float = Field(alias="compactness error")
    concavity_error: float = Field(alias="concavity error")
    concave_points_error: float = Field(alias="concave points error")
    symmetry_error: float = Field(alias="symmetry error")
    fractal_dimension_error: float = Field(alias="fractal dimension error")
    worst_radius: float = Field(alias="worst radius")
    worst_texture: float = Field(alias="worst texture")
    worst_perimeter: float = Field(alias="worst perimeter")
    worst_area: float = Field(alias="worst area")
    worst_smoothness: float = Field(alias="worst smoothness")
    worst_compactness: float = Field(alias="worst compactness")
    worst_concavity: float = Field(alias="worst concavity")
    worst_concave_points: float = Field(alias="worst concave points")
    worst_symmetry: float = Field(alias="worst symmetry")
    worst_fractal_dimension: float = Field(alias="worst fractal dimension")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    features = request.model_dump(by_alias=True)
    result = predict(features)
    return result
