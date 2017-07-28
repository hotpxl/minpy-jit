#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools


def atomic(f):
    f.__dict__['__minpy_atomic'] = True
    return f
