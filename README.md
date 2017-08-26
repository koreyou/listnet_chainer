
# Chainer Implementation of ListNet

## Introduction

ListNet ranking model. This is a Chainer implementation of ["Learning to rank: from pairwise approach to listwise approach" by Cao et al.](http://dl.acm.org/citation.cfm?doid=1273496.1273513).

Code explanation is given at http://qiita.com/koreyou/items/a69750696fd0b9d88608 (Japanese).

## How to run

### Prerequisite

```python
pip install -r requirements.py
export PYTHONPATH="`pwd`:$PYTHONPTH"
```

Download LETOR dataset from: http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2007.rar .
Unrar data into "build" directory such that directory is organized `build/MQ2007`.

### Running the code

```
python bin/train.py
```


## Licence

This code is distributed under CC-BY.
