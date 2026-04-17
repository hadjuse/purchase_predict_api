import pandas as pd
import os

def main():
    # N'oubliez pas de mettre le path du projet kedro ici
    path_kedro = r"/data/Documents/Ing3/mlops/kedro/project-mlops"
    dataset = pd.read_csv(os.path.join(path_kedro, r"data/03_primary/rimary.csv"))
    dataset = dataset.drop(["user_session", "user_id", "purchased"], axis=1)

if __name__ == "__main__":
    main()
