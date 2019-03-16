# ML Tutorials

A collection of tutorials walking through some key ML concepts. I adapted these
from some of the
[TensorFlow tutorials](https://www.tensorflow.org/tutorials).

## Installing the environment

You need to have `conda` installed (from
[Miniconda](https://conda.io/miniconda.html) or
[Anaconda](https://www.anaconda.com/download)).

Create the environment:

```bash
$ conda env create -f environment.yml
```

After packages are installed, you can activate the environment, and start
the notebook server.

```bash
$ conda activate ml_tutorials
$ jupyter notebook
```

## Specific instructions for specific tutorials

### Tutorial 3: Text classification

Download and extract data to `data/raw` as:

```bash
python -m tutorial.scripts.download_imdb_dataset data/raw
```
