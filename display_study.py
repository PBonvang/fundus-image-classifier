import optuna
from optuna.study import StudyDirection
from optuna.trial import TrialState
from optuna.visualization import plot_param_importances, plot_parallel_coordinate

import config

def print_top_5_trials(study: optuna.Study):
    print("Top 5:")
    print("-------------------------------------")
    sort_type = input("Do you want to sort by loss(L) or accuracy(A): ").upper()

    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    if sort_type == "L":
        complete_trials.sort(key=lambda t: t.value)
    elif sort_type == "A":
        complete_trials.sort(key=lambda t: t.user_attrs["Accuracy"], reverse=True)

    top_five = complete_trials[:5]
    for i, trial in enumerate(top_five):
        print(f"\nTrial {i+1}:")
        print_trial(trial)

def print_trial(trial: optuna.Trial):
    acc = float(trial.user_attrs["Accuracy"])*100
    
    print("  Value: ", trial.value)
    print(f"  Accuracy: {acc:.1f} %")
    print("  Parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    study_name = "32-bs_above-is-not-possible-to-allocate"
    storage_name = f"sqlite:///{config.STUDIES_PATH}/{study_name}.db"
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    
    show_trial_table = input("Do you want to print trial table? [Y/N]: ")
    if show_trial_table.upper() == "Y":
        df = study.trials_dataframe(attrs=("value", "params", "state")).sort_values(by="value")
        print(df.to_string())

    print_top_5_trials(study)

    show_plot = input("Do you wish to view plots? [Y/N]: ")
    if show_plot.upper() == "Y":
        param_importance_fig = plot_param_importances(study)
        param_importance_fig.show()

        para_cord_fig = plot_parallel_coordinate(study)
        para_cord_fig.show()