# Fundus image-classifier
A set of scripts and utility functions made to make it easy to experiment with different Neural Network architectures for the fundus dataset provided by StatuManu.

## Use
Modified: 21/01/2022  

### Prerequisites
1. Set environment config in [config.py](config.py)
2. Build dataset by running [build_dataset.py](build_dataset.py)

### Setting up model
Open [Model.py](Model.py) and define your model by following the instructions
    - Examples can be found under [models/examples](models/examples)

## Metadata
The final step in the trainer script, saves the model and the run information.

The metadata includes attributes like learning rate, epochs, samples seen, model base...
and can be found in will be saved to the `MODELS_INFO_FILE` specified in [config.py](config.py).
This info file was designed to be used as a [pandas](https://pandas.pydata.org/) dataframe or with excel to provide a searchable database.

### Training
If the model has been set up correctly the model can be trained using [train_model.py](train_model.py)

#### Training stats
Training stats are saved as tensorboard events saved in the run folder, a subdirectory of the models directory specified in `MODELS_PATH` of [config.py](config.py).  
To view the stats, open tensorboard and find your run.

## Dataset Weights
If the dataset was build with [build_dataset.py](build_dataset.py) you'll find a info.txt file in the base of it, which included the `pos_weight` for the dataset.  

`pos_weight` is calculated as: $\frac{\text{\# neg samples}}{\text{\# pos samples}}$

## References
- How can I replace the forward method of a predefined torchvision model with my customized forward function? - [discuss.pytorch.org](https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/7)
- How to apply constraint to weights in convolutional layer? - [discuss.pytorch.org](https://discuss.pytorch.org/t/how-to-apply-constraint-to-weights-in-convolutional-layer/13856)
- [MIT 6.S191 (2020): Convolutional Neural Networks](https://www.youtube.com/watch?v=iaSUYvmCekI&feature=youtu.be)
- [TensorFlow. Intro to Machine Learning (ML Zero to Hero)](https://www.youtube.com/watch?v=KNAWp2S3w94)
- [Christian Versloot. How to use K-fold Cross Validation with PyTorch? machinecurve.com, 2021](https://www.machinecurve.com/index.php/2021/02/03/how-to-use-k-fold-cross-validation-with-pytorch/)
- [Adrian Rosebrock. PyTorch: Transfer Learning and Image Classification. pyimagesearch, 2021](https://www.pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/)

### Papers
- [Asha Gnana Priya Henry and Anitha Jude. Convolutional neural-network-based classification of retinal images with
different combinations of filtering techniques. Open Computer Science, 11(1):480–490, 2021.](https://www.degruyter.com/document/doi/10.1515/comp-2020-0177/html)
- [Mikkel Schou Andersen, Christian Bonde Pedersen, and Frantz Rom Poulsen. A new novel method for assessing
intracranial pressure using non-invasive fundus images: a pilot study. Scientific Reports, 10(1):13062, August 2020.](https://www.nature.com/articles/s41598-020-70084-0)
- [Wouter Van Gansbeke and Simon Vandenhende and Stamatios Georgoulis and Marc Proesmans and Luc Van Gool. SCAN: Learning to Classify Images without Labels. 2020](https://arxiv.org/abs/2005.12320)


## Improvement ideas
- Continue model training where it left off with epochs and perhaps steps
    - Continueing tensorboard overview
    - Using run specifications to keep track of epochs, samples, dataset...
- Provide a better model comparison using the model_info file