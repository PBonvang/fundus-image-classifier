import optuna
from optuna.study import StudyDirection
from optuna.trial import TrialState
from optuna.visualization import plot_param_importances

import config

def print_top_5_trials(study: optuna.Study):
    print("Top 5:")
    print("-------------------------------------")

    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    complete_trials.sort(key=lambda t: t.value)

    top_five = complete_trials[:5]
    for i, trial in enumerate(top_five):
        print(f"\nTrial {i+1}:")
        print_trial(trial)

def print_trial(trial: optuna.Trial):
    acc = float(trial.user_attrs["Accuracy"])*100
    
    print("  Value: ", trial.value)
    print(f"  Accuracy: {acc:.5f} %")
    print("  Parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

study_name = "50epochs-64bs"
storage_name = f"sqlite:///{config.STUDIES_PATH}/{study_name}.db"
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)

df = study.trials_dataframe(attrs=("value", "params", "state")).sort_values(by="value")
print(df.to_string())

print_top_5_trials(study)

param_importance_fig = plot_param_importances(study)
param_importance_fig.show()