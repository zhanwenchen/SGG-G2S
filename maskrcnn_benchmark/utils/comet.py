from os import environ as os_environ
from hashlib import sha1
from comet_ml import API, Experiment, ExistingExperiment
from my_secrets import COMET_API_KEY, COMET_PROJECT_NAME, COMET_WORKSPACE


def get_experiment(run_id):
    experiment_id = sha1(run_id.encode("utf-8")).hexdigest()
    os_environ["COMET_EXPERIMENT_KEY"] = experiment_id

    api = API(api_key=COMET_API_KEY)  # Assumes API key is set in config/env
    api_experiment = api.get_experiment_by_key(experiment_id)

    if api_experiment is None:
        return Experiment(
            api_key=COMET_API_KEY,
            project_name=COMET_PROJECT_NAME,
            workspace=COMET_WORKSPACE,
        )
    else:
        return ExistingExperiment(project_name=COMET_PROJECT_NAME)
