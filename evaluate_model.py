from datetime import datetime
import config
import torch
import os

from torch.utils.data.dataloader import DataLoader

from utils.dataloading import get_dataset
from utils.evaluation import evaluate_model
from Model import get_model
from utils.RunInfo import RunInfo

evaluations_info_file = os.path.join(
	"evaluations","evaluations.csv"
)

model = get_model()
net_path = input("Specify path to model state dictionary (.pth file): ")
model.network.load_state_dict(torch.load(net_path))
network = model.network.to(config.DEVICE)

validation_ds = get_dataset(
    config.TEST_INFO, config.TEST, model.validation_transforms)

val_dl = DataLoader(
	validation_ds,
	batch_size=model.batch_size
)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_path = os.path.join("evaluations",timestamp)
os.makedirs(save_path)

run_info = RunInfo(
	model,
	save_path,
	len(validation_ds),
	name=timestamp,
	base_model_path=net_path
)

accuracy = evaluate_model(model, val_dl, save_path)*100
run_info.set_test_accuracy(accuracy)
print(f'Accuracy: {accuracy:.5f} %')

run_info.save_to_csv(evaluations_info_file)