# NN-experimenter
A set of scripts and utility functions was made to make it easy to experiment with Neural Networks for the fundus dataset provided by StatuManu.

## Use
Modified: 14/01/2022  

### Prerequisites
1. Set environment config in [config.py](config.py)
2. Build dataset by running [build_dataset.py](build_dataset.py)

### Setting up model
Open [Model.py](Model.py) and define your model by following the instructions
    - Examples can be found under [models/examples](models/examples)

## Metadata
The final step in the trainer script, saves the model, its blueprint and metadata.

The metadata includes attributes like learning rate, epochs, optimizer function...
and can be found in [model_info.csv](model_info.csv).  
This info file was designed to be used as a pandas dataframe or with excel to provide a searchable database.

### Training
If the model has been set up correctly the model can be trained using [trainer.py](trainer.py)

#### Training stats
Training stats are saved as tensorboard runs located in the *runs* folder.  
To view the stats, open tensorboard and find your run.

## Dataset Weights
Sample: 3.492063492


Full: 0.366838129

pos_weight = num_neg_samples / num_pos_samples

## References
- Setting up a transfer learning network: https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/7