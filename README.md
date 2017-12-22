
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

## Experiment

I have run MQ2007 on [LETOR 4.0](http://research.microsoft.com/en-us/um/beijing/projects/letor/letor4dataset.aspx).
I have only tested it on Fold 1.

### Result

Here is the performance metrics in mean average precision (MAP).

```
TRAIN: 0.4693
DEV:   0.4767
TEST:  0.4877
```

This is [the official result](https://1drv.ms/u/s!Aqi9ONgj3OqPgSS45WACJ5uKK-ok) on the same dataset.

```
TRAIN: 0.4526
DEV:   0.4790
TEST:  0.4884
```

## Licence

This code is distributed under CC-BY.
