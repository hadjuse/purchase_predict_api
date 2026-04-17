import os

from dotenv import load_dotenv

load_dotenv()

for env_var in ["ENV", "MLFLOW_SERVER", "MLFLOW_REGISTRY_NAME"]:
    if not os.getenv(env_var):
        raise Exception(
            "Environment variable {} must be defined.".format(env_var)
        )