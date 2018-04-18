# Intro to Machine Learning and Short Keras Tutorial

Materials for "Introduction to Machine Learning Workshop" talk at IIHE ULB-VUB. 
A live version of the slides can be found at https://zhampel.github.io/intro-machine-learning/.


## Installation

The dependencies for generating the slides are:

- `numpy`
- `scipy`
- `jupyter`
- `scikit-learn`
- `pillow`
- `matplotlib`
- `h5py`
- `Theano`
- `tensorflow`
- `keras`

## Usage

The slides for this talk can be generated via:

```bash
jupyter nbconvert intro-machine-learning.ipynb --to slides --post serve
```

To run the corresponding notebook use

```bash
jupyter notebook ml-examples.ipynb
```
