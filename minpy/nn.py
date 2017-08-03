#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import inspect
import textwrap
import functools

from . import core

# CR(gaiyu): To my understanding, you are depending on inheritance
# to make sure `compile` gets called. And `compile` takes a class
# and turns the forward method into a jittable free function. It
# would not work if either the user did not inherit from Module or
# the user did not call the function forward. To make it more
# general, I suggest you provide a decorator that lifts the method
# to a free function. So the user can annotate whatever function
# they want, and don't have to worry about inheritance.
#
# Something like this:
# def decorator(func):
#     get source
#     dedent func
#     compile func (common code)
#     def wrapped_fn(*args, **kwargs):
#         return compiled_fn(*args, **kwargs)
#     return wrapped_fn
#
# XCR(yutian): I am not sure whether it is feasible to lift a bound method
# to free function via a decorator. For example:
#
# def decorator(f):
#     ...
#
# class C:
#     @decorator
#     def wrapped_method(self):
#         return self.attr
#
# When received by `decorator`, `wrapped_method` is actually "free", meaning that
# it does not contain any reference to `self`. However, in order to lift a bound method,
# I must be able to access `self` and put references to all attributes of `self`
# that are used in `wrapped_method` in an auxiliary namespace. In pseudocode:
#
# namespace = {}
# used_attrs = search_for_used_attrs_via_ast(f)
# for attr in used_attrs:
#     namespace[attr] = getattr(self, attr) # `self` is unavailable!
# compile_in_namespace(f, namespace)
#
# But I think it is possible to remove the constraint that only `forward` can be lifted.
#
# XCR(gaiyu): I think you probably have misconceptions towards Python
# instance methods. The method definition itself is never "bound". You
# cannot access instance variables WITHOUT going through `self`. In
# fact, in Python implementation, the "binding" process happens at
# access time (when you say "obj.func"). So the definition itself
# SHOULD look no different than a normal free function. It is very
# different from a C++/Java instance method definition.  I have taken
# the liberty to rewrite it in terms of existing functions. Check if
# this achieves the intended behavior.

_reentrance = False


def jit_instance_method(func):
    global _reentrance
    if _reentrance:
        return func
    _reentrance = True
    source_code = textwrap.dedent(inspect.getsource(func))
    function_ast = ast.parse(source_code, mode='exec')
    # logic to modify source, share workflow logic with jit, here we just print the ast
    print(core.pretty_print(function_ast))
    free_function = core.evaluate_function_definition(function_ast,
                                                      func.__globals__, [], [])

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        return free_function(*args, **kwargs)

    _reentrance = False
    return wrapped_func
