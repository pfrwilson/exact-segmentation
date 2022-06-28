from dotenv import load_dotenv

load_dotenv()

from exactvu.resources import metadata

metadata()
cores = metadata()["core_specifier"]
cores = list(cores)

from src.ensemble_predict import predict

predict(cores, "/mnt/data2/paul/cores_dataset", "checkpoints.yaml")
