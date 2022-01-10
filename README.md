# NN-experimenter
A set of scripts and utility functions setout to make it easy to setup, train and evaluate a Neural Network.

## Walkthrough
The idea is that you define your model in [Model.py](Model.py) following the instructions in the file. Then training, evaluating and saving metadata for the defined model, to give an overview over different models and their accuracy.

Steps:  
1. Edit [config.py](config.py) to fit your setup
2. Define your model in [Model.py](Model.py)
3. Train your model by running the [trainer.py](trainer.py) script

## Metadata
The final step in the trainer script, saves the model, its blueprint and metadata.

The metadata includes attributes like learning rate, epochs, optimizer function...
and can be found in [model_info.csv](model_info.csv).  
This info file was designed to be used as a pandas dataframe or with excel to provide a searchable database.

## Weights
Sample: ```pos_weight=torch.tensor([3.492063492]).to(config.DEVICE)```
Full: ```pos_weight=torch.tensor([0.366838129]).to(config.DEVICE)```

## References
- Setting up a transfer learning network: https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/7