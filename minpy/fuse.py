#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import mxnet


class NodeTransformer(ast.NodeTransformer):
    def visit(self, node):
        # Always recurse and then call corresponding method.
        method = 'visit_' + node.__class__.__name__
        node = self.generic_visit(node)
        return getattr(self, method, lambda x: x)(node)

    def visit_Name(self, node):
        if issubclass(getattr(node, 'type', object), mxnet.nd.NDArray):
            node.fuse = True
        return node

    def visit_BinOp(self, node):
        if getattr(node.left, 'fuse', False) and getattr(
                node.right, 'fuse', False):
            node.fuse = True
        return node

    def visit_Tuple(self, node):
        if isinstance(node.ctx, ast.Load) and all(
                map(lambda x: getattr(x, 'fuse', False), node.elts)):
            node.fuse = True
        return node

    def visit_Call(self, node):
        if hasattr(node, 'ref') and getattr(
                node.ref, '__minpy_atomic', False) and all(
                    map(lambda x: getattr(x, 'fuse', False), node.args)):
            node.fuse = True
        return node


def fuse(function_ast):
    function_ast = NodeTransformer().visit(function_ast)
    return function_ast
