import mlflow
import pyspark


class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        # return {"Pyspark_version": "3.3.0"}
        return {"Pyspark_version": pyspark.__version__}
