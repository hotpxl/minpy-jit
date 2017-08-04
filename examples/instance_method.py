#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import minpy


class A():
    def __init__(self):
        self.attr = 100

    @minpy.jit_instance_method
    def instance_method(self):
        print(self.attr)
