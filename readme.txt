0. Clone this repository, download DSD100 from https://sigsep.github.io/datasets/dsd100.html and extract it to the dataset folder
1. Add pytorch channel to conda with conda config --append channels pytorch
2. create virtual conda enviroment from the conda-requirements.txt
3. install required pip packages with pip install from pip-requirements.txt
4. activate the environment with conda activate
5. Depending on your setup, make a script that runs the command

python ./scripts/training.py --layers $layers --channels $channels 
where $layers refers to the amount of DWS layers in the arcitechture and $channels is the amount of CNN channels.
For example: python ./scripts/training.py --layers 5 --channels 64

5.1 Alternatively if using an IDE, one can manually change the default parser settings in training.py to the desired hyperparameters. 
 

