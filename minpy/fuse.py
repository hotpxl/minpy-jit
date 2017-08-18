#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import mxnet

_ndarray_funcs = vars(mxnet.nd).values()


# the thing is, every single rule is (approximately) one visit_**
# method here. but adding rules sohuld be very careful! not thinking
# about this formally will result in a UNSOUND fuse
# and also, here we are dealing only with expressions. we deal with
# statements later
# formal method is noted down here: https://goo.gl/B691Py
class NodeTransformer(ast.NodeTransformer):
    def visit(self, node):
        # Always recurse and then call corresponding method.
        method = 'visit_' + type(node).__name__
        node = self.generic_visit(node)
        return getattr(self, method, lambda x: x)(node)

    def get_fuse(self, node):
        return getattr(node, 'fuse', False)

    def visit_Name(self, node):
        if issubclass(getattr(node, 'type', object), mxnet.nd.NDArray):
            node.fuse = True
        return node

    def visit_BinOp(self, node):
        if self.get_fuse(node.left) and self.get_fuse(node.right):
            node.fuse = True
        return node

    def visit_Tuple(self, node):
        if isinstance(
                node.ctx,
                ast.Load) and all(map(lambda x: self.get_fuse(x), node.elts)):
            node.fuse = True
        return node

    def visit_Call(self, node):
        # what if the function itself is a complicated attribute? even
        # dynamic?
        # func_dict[function_name](arguments) like this?
        # how to deal with this formally?
        if hasattr(node, 'ref'):
            if getattr(node.ref, '__minpy_atomic',
                       False) or node.ref in _ndarray_funcs:
                if all(map(lambda x: self.get_fuse(x), node.args)):
                    node.fuse = True
        return node

    # statements below
    def visit_Assign(self, node):
        # TODO(yutian) multiple targets
        if len(node.targets) == 1 and isinstance(
                node.targets[0], ast.Name) and self.get_fuse(node.value):
            node.fuse = True
        return node

    # Why do we need an explicit expr now we have dealt with
    # expressions before?  Because here it means a statement with a
    # single expression.
    def visit_Expr(self, node):
        if self.get_fuse(node.value):
            node.fuse = True
        return node


def fuse(function_ast):
    function_ast = NodeTransformer().visit(function_ast)
    return function_ast
