import optuna
from optuna.study import StudyDirection

import config

study_name = "50epoch-64bs"
storage_name = f"sqlite:///{config.STUDIES_PATH}/{study_name}.db"
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)

df = study.trials_dataframe(attrs=("value", "params", "state")).sort_values(by="value")
print(df)

print("Best model:")
acc = float(study.best_trial.user_attrs["Accuracy"])*100
print(f" Value: {study.best_value}")
print(f" Accuracy: {acc}")
print(f" Best params:")
for key, value in study.best_params.items():
        print("    {}: {}".format(key, value))