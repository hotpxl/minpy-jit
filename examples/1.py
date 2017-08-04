#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import numpy as np
import minpy


@minpy.jit
def user_func():
    a = np.random.normal(size=(3, 3))
    b = a + 3
    if 0 < a.sum():
        print('<0')
    else:
        print('>0')
    return b * a


class C():
    @minpy.jit
    def method(self):
        return 'method'


print(user_func())
print(user_func())
print(user_func())
print(C().method())
