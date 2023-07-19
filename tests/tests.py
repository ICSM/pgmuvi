# import unittest

from ..lightcurve import LightCurve, Transformer, MinMax, ZScore, RobustZScore
import numpy as np



def test_transformer():
    transformer = Transformer()
    

def test_minmax_zero():
    transformer = MinMax()
    assert transformer.transform(np.array([0, 0, 0, 0, 1])).min() == 0

def test_minmax_one():
    transformer = MinMax()
    assert transformer.transform(np.array([0, 0, 0, 0, 1])).max() == 1