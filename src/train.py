import os

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import load_data, split_data

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "breast-cancer-classification"
REGISTERED_MODEL_NAME = "breast_cancer_classifier"
LOCAL_MODEL_PATH = "models/model.joblib"

MODELS = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        {"max_iter": 1000},
    ),
    "random_forest": (
        RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        {"n_estimators": 200, "max_depth": 6},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
        {"n_estimators": 150, "learning_rate": 0.1},
    ),
}


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    best_run_id, best_model_name, best_f1 = None, None, -1.0

    for name, (estimator, params) in MODELS.items():
        with mlflow.start_run(run_name=name) as run:
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, y_pred, y_proba)

            mlflow.log_param("model_type", name)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            signature = infer_signature(X_train, pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(3),
            )

            print(f"[{name}] {metrics}")

            if metrics["f1"] > best_f1:
                best_f1, best_run_id, best_model_name = metrics["f1"], run.info.run_id, name

    print(f"\nBest model: {best_model_name} (run_id={best_run_id}, f1={best_f1:.4f})")

    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    print(f"Registered '{REGISTERED_MODEL_NAME}' version {result.version}")

    client = MlflowClient()
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=result.version,
        stage="Production",
        archive_existing_versions=True,
    )

    os.makedirs("models", exist_ok=True)
    best_pipeline = mlflow.sklearn.load_model(model_uri)
    joblib.dump(best_pipeline, LOCAL_MODEL_PATH)
    print(f"Saved local copy for serving to {LOCAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
