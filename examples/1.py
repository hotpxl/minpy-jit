#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from mxnet import nd
sys.path.append('../')
import minpy


@minpy.jit
def user_func():
    a = nd.random_normal(shape=(3, 3))
    b = a + 3
    if 0 < a.asnumpy().sum():
        b1 = b + 3
        print('<0')
    else:
        b2 = b + 3
        b3 = b2 + 3
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
