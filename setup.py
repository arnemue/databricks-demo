from setuptools import find_packages, setup

__version__ = "0.1.0"

setup(
    name="recomate",
    version=__version__,
    description="Dummy package implementing an wrapper for a mlflow recommender model",
    packages=find_packages(exclude=["tests"]),
    zip_safe=False,
)
