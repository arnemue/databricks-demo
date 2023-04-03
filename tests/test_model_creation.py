import mlflow
import mlflow.models.utils
import numpy as np
import pandas as pd
import pytest
from mlflow.utils.environment import _mlflow_conda_env
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from recomate.model import SklearnModelWrapper

PYSPARK_OUTPUT_VERSION = {"Pyspark_version": "3.3.0"}


@pytest.fixture
def train_data() -> tuple[pd.DataFrame, pd.Series]:
    white_wine = pd.read_csv("data/winequality-white.csv", sep=";")
    red_wine = pd.read_csv("data/winequality-red.csv", sep=";")

    red_wine["is_red"] = 1
    white_wine["is_red"] = 0

    data = pd.concat([red_wine, white_wine], axis=0)

    # Remove spaces from column names
    data.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)
    high_quality = (data.quality >= 7).astype(int)
    data.quality = high_quality

    X = data.drop(["quality"], axis=1)
    y = data.quality
    return (X, y)


def test_model_wrapper(train_data: tuple[pd.DataFrame, pd.Series]):
    X_train, X_rem, y_train, y_rem = train_test_split(
        train_data[0], train_data[1], train_size=0.6, random_state=123
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem, test_size=0.5, random_state=123
    )
    with mlflow.start_run(run_name="untuned_random_forest"):
        n_estimators = 10
        mlflow.log_param("n_estimators", n_estimators)

        model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=np.random.RandomState(123)
        )
        model.fit(X_train, y_train)
        predictions_test = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, predictions_test)
        mlflow.log_param("n_estimators", n_estimators)
        # Use the area under the ROC curve as a metric.
        mlflow.log_metric("auc", auc_score)
        wrapped_model = SklearnModelWrapper(model=model)

        conda_env_ = _mlflow_conda_env(
            additional_conda_deps=None,
            additional_pip_deps=["requests"],
            additional_conda_channels=None,
        )
        mlflow.pyfunc.log_model(
            "random_forest_model", python_model=wrapped_model, conda_env=conda_env_
        )

        run_id = (
            mlflow.search_runs(
                filter_string='tags.mlflow.runName = "untuned_random_forest"'
            )
            .iloc[0]
            .run_id
        )
        model_name = "wine_quality"
        model_version = mlflow.register_model(
            f"runs:/{run_id}/random_forest_model", model_name
        )
        mlflow.models.utils.add_libraries_to_model(f"models:/{model_name}/latest")

    assert (
        wrapped_model.predict(context="some", model_input=X_train)
        == PYSPARK_OUTPUT_VERSION
    )
