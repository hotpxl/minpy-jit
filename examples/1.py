#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import numpy as np
import minpy


def user_func():
    a = np.random.rand(3, 3)
    b = a + 3
    return b * a


print(user_func())
