# MMNav

## Installation
* Pytorch 1.10.0
* Python 3.8.20

### Installing Dependencies
* Install habitat-sim
```
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; git checkout tags/challenge-2022; 
pip install -r requirements.txt; 
python setup.py install --headless
```

* Install habitat-lab
```
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab; git checkout tags/challenge-2022; 
pip install -e .
```

* Replace the habitat folder in habitat-lab repo for the multi-robot setting
```
mv -r habitat your-path/habitat-lab
```

* Download the prediction-related files and models [here](https://huggingface.co/bowen666212/MMNav_model_files).

After unzipping, place the prediction-related files in the root directory, and also place `iter.pth` in the root directory. Place `mask_rcnn_R_101_cat9.pth` in the `nav/agent/utils` directory

## Dataset

Download the MMNav dataset [here](https://huggingface.co/datasets/bowen666212/MMNav). It include train/val/test split.

## Evaluating the navigation agent

Evaluate the agent's capability using the `script main.sh`

## Training the refine model

Run `refine.py` to train the refinement model

## Collection of Semantic Map Dataset

We follow the [PEANUT](https://github.com/ajzhai/PEANUT/tree/master) to collect the semantic map dataset and use it to train the object probability prediction model and refinement model.
The corresponding dataset can be downloaded from [here](https://huggingface.co/datasets/bowen666212/MMNav_Maps).
  
