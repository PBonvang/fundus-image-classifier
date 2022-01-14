import optuna
from optuna.study import StudyDirection

study_name = "thomas-study-100epochs-2fold"
storage_name = f"sqlite:///{study_name}.db"
study = optuna.create_study(study_name=study_name, storage=storage_name, direction=StudyDirection.MAXIMIZE, load_if_exists=True)

df = study.trials_dataframe(attrs=("value", "params", "state")).sort_values(by="value", ascending=False)
print(df)

print("Best model:")
print(f" Value: {study.best_value}")
print(f" Best params:")
for key, value in study.best_params.items():
        print("    {}: {}".format(key, value))