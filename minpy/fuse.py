#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import mxnet

_ndarray_funcs = vars(mxnet.nd).values()


class NodeTransformer(ast.NodeTransformer):
    def visit(self, node):
        # Always recurse and then call corresponding method.
        method = 'visit_' + type(node).__name__
        node = self.generic_visit(node)
        return getattr(self, method, lambda x: x)(node)


def get_static(node):
    return getattr(node, 'static', False)


def get_fuse(node):
    return getattr(node, 'fuse', False)


def get_type(node):
    return getattr(node, 'type', object)


class StaticAnalyzer(NodeTransformer):
    def visit_Name(self, node):
        node.static = True
        return node

    def visit_Attribute(self, node):
        if get_static(node.value):
            node.static = True
        return node

    def visit_Subscript(self, node):
        if get_static(node.value) and get_static(node.slice):
            node.static = True
        return node

    def visit_Index(self, node):
        if isinstance(node.value, (ast.Num, ast.Str)):
            node.static = True
        return node


# the thing is, every single rule is (approximately) one visit_**
# method here. but adding rules sohuld be very careful! not thinking
# about this formally will result in a UNSOUND fuse
# and also, here we are dealing only with expressions. we deal with
# statements later
# formal method is noted down here: https://goo.gl/B691Py
class FuseAnalyzer(ast.NodeTransformer):
    def visit(self, node):
        # Always recurse and then call corresponding method.
        method = 'visit_' + type(node).__name__
        node = self.generic_visit(node)
        return getattr(self, method, lambda x: x)(node)

    def visit_Name(self, node):
        if issubclass(get_type(node), mxnet.nd.NDArray):
            node.fuse = True
        return node

    def visit_Num(self, node):
        node.fuse = True
        return node

    def visit_BinOp(self, node):
        if get_fuse(node.left) and get_fuse(node.right):
            node.fuse = True
        return node

    def visit_Tuple(self, node):
        if isinstance(node.ctx,
                      ast.Load) and all(map(lambda x: get_fuse(x), node.elts)):
            node.fuse = True
        return node

    def visit_Call(self, node):
        # We assumed that the functions do not change across calls. So
        # we don't deal with the staticness of the function itself.
        if hasattr(node, 'ref'):
            if getattr(node.ref, '__minpy_atomic',
                       False) or node.ref in _ndarray_funcs:
                if all(map(lambda x: get_fuse(x), node.args)):
                    node.fuse = True
        return node

    def visit_Subscript(self, node):
        if isinstance(node.value, mxnet.nd.NDArray) and get_static(node.slice):
            node.fuse = True
        if get_static(node.value) and get_static(node.slice):
            node.fuse = True
        return node

    # Statements below.

    def visit_Assign(self, node):
        # TODO(yutian) multiple targets
        if len(node.targets) == 1 and isinstance(
                node.targets[0], ast.Name) and get_fuse(node.value):
            node.fuse = True
        return node

    # Why do we need an explicit expr now we have dealt with
    # expressions before?  Because here it means a statement with a
    # single expression.
    def visit_Expr(self, node):
        if get_fuse(node.value):
            node.fuse = True
        return node


def fuse(function_ast):
    function_ast = StaticAnalyzer().visit(function_ast)
    function_ast = FuseAnalyzer().visit(function_ast)
    # get consec stmt blablabla
    # get input/output blabla
    return function_ast
