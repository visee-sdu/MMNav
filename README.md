# MMNav

## Installation
* Pytorch 2.1.1
* Python 3.9.18

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

* replace the habitat folder in habitat-lab repo for the multi-robot setting
```
mv -r habitat your-path/habitat-lab
```
