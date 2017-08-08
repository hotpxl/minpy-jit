#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from . import core
from . import segment


@core.return_on_reentrance
def jit(f):
    function_ast = core.parse_function_definition(f)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        with jit.reentrance_guard():
            nonlocal function_ast
            closure_parameters = []
            closure_arguments = []

            traced_function_ast, i, j = core.add_type_tracing(function_ast)
            closure_parameters.extend(i)
            closure_arguments.extend(j)
            traced_function_ast, i, j = core.add_function_tracing(
                traced_function_ast)
            closure_parameters.extend(i)
            closure_arguments.extend(j)

            new_function = core.evaluate_function_definition(
                traced_function_ast, f.__globals__, closure_parameters,
                closure_arguments)

            ret = new_function(*args, **kwargs)
            print(core.pretty_print(function_ast, include_attributes=False))
            function_ast = segment.segment(function_ast, True)
            print(core.pretty_print(function_ast, include_attributes=False))
            return ret

    return wrapper
