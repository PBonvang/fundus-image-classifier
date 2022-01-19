#%% Setup
from datetime import datetime
import config
import torch
import os

from torch.utils.data.dataloader import DataLoader

from utils.dataloading import get_dataset
from utils.evaluation import evaluate_model
from utils.RunInfo import RunInfo
from baseline_models import Densenet201, GNet, ResNet101, ResNet152

evaluations_info_file = os.path.join(
	"evaluations","evaluations.csv"
)

baselines = [
	{"model": Densenet201, "net": "baselines\\Densenet201\\20220118_111822.pth"},
	{"model": GNet, "net": "baselines\\GNet\\20220118_140049.pth"},
	{"model": ResNet101, "net": "baselines\\ResNet101\\20220118_161426.pth"},
	{"model": ResNet152, "net": "baselines\\ResNet152\\20220118_184821.pth"}
]

val_transforms = baselines[0].get("model").get_model().validation_transforms
batch_size = 32

validation_ds = get_dataset(
    config.TEST_INFO, config.TEST, val_transforms)

val_dl = DataLoader(
	validation_ds,
	batch_size=batch_size
)

#%% Validation
accuracies = []
for baseline in baselines:
	model_type = baseline.get("model")
	model_name = model_type.__name__.split(".")[-1]
	model = model_type.get_model()
	net_path = baseline.get("net")
	model.network.load_state_dict(torch.load(net_path))
	network = model.network.to(config.DEVICE)

	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	save_path = os.path.join("evaluations",model_name, timestamp)
	os.makedirs(save_path)

	run_info = RunInfo(
		model,
		save_path,
		len(validation_ds),
		name=model_name,
		base_model_path=net_path
	)

	accuracy = evaluate_model(model, val_dl, save_path)*100
	run_info.set_test_accuracy(accuracy)
	print(f'{model_name} accuracy: {accuracy:.5f} %')

	run_info.save_to_csv(evaluations_info_file)

	accuracies.append(
		{"model":model_name, "accuracy":accuracy}
	)

#%% Overview
print("\nAccuracy overview:")
print("--------------------------------")
for acc in accuracies:
	print("Model: {: <15} Accuracy: {} %".format(acc.get("model"), acc.get("accuracy")))