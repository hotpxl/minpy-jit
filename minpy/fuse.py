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


def infer_inputs_and_outputs_given_nodes(nodes):
    """Given a/a list of ast-node, infer the input and output variables
    Parameters
    ----------
    nodes: a single node or a lsit of nodes

    Returns
    -------
    ins: the input variable names
    outs: the output variable names
    """

    def infer_inputs_and_outputs_given_node(node):
        """Given a ast-node, infer the input and output variables
        The node should be either assign-statement or expression

        Parameters
        ----------
        node: a single node

        Returns
        -------
        ins: the input variable names
        outs: the output variable names
        """
        if isinstance(node, ast.Assign):
            # get inputs from its value expression
            ins, _ = infer_inputs_and_outputs_given_node(node.value)
            # treat all the targets as outputs
            outs = collect_names_given_exprs(node.targets)
            return ins, outs
        elif isinstance(node, (ast.expr, ast.Expr)):
            return collect_names_given_exprs(node), set([])
        else:
            raise TypeError(
                'Type {} not handled yet in inputs and outputs inference'.
                format(type(node).__name__))

    def collect_names_given_exprs(expr):
        """Given a ast-node, infer the input variables
        As expression cannot define a new variable, output is not inferred

        Parameters
        ----------
        expr: an expression

        Returns
        -------
        ins: the input variable names

        TODO:
          - handle the slice object
          - need to know the actual type of the left operand of attribute
            - if it's module or class, then return []
        """
        if isinstance(expr, list):
            return set.union(*map(collect_names_given_exprs,
                                  expr)) if expr else set()
        elif isinstance(expr, ast.Call):
            return collect_names_given_exprs(expr.args)
        elif isinstance(expr, ast.Expr):
            return collect_names_given_exprs(expr.value)
        elif isinstance(expr, ast.BinOp):
            return collect_names_given_exprs([expr.left, expr.right])
        elif isinstance(expr, ast.UnaryOp):
            return collect_names_given_exprs(expr.operand)
        elif isinstance(expr, ast.Tuple):
            return collect_names_given_exprs(expr.elts)
        elif isinstance(expr, ast.Attribute):
            # Assumption: left operand is a Name
            assert isinstance(expr.expr, ast.Name)
            return set([expr.expr.id + "." + expr.attr])
        elif isinstance(expr, ast.Subscript):
            # Assumption: left operand is a Name
            assert isinstance(expr.expr, ast.Name)
            return set([expr.expr.id + "_subscript_"])
        elif isinstance(expr, ast.Name):
            return set([expr.id])
        elif isinstance(expr, (ast.Num, ast.Str, ast.Bytes)):
            return set()

        raise TypeError('{} not handled yet in inference of inputs'.format(
            type(expr).__name__))

    if isinstance(nodes, list):
        ins = []
        outs = []
        for node in nodes:
            sub_ins, sub_outs = infer_inputs_and_outputs_given_node(node)
            ins += [x for x in sub_ins if x not in outs]
            outs += sub_outs
        return [e for e in (set(ins))], [e for e in set(outs)]
    else:
        return infer_inputs_and_outputs_given_node(nodes)


class EnvironmentAnalyzer(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        # Merge consecutive statements.
        body = node.body
        i = j = 0
        while j < len(body):
            if not get_fuse(body[j]):
                if i < j:
                    print(i, j)
                    print(infer_inputs_and_outputs_given_nodes(body[i:j]))
                i = j + 1
            j += 1
        if i < j:
            print(i, j)
            print(infer_inputs_and_outputs_given_nodes(body[i:j]))
        return node


def fuse(function_ast):
    function_ast = StaticAnalyzer().visit(function_ast)
    function_ast = FuseAnalyzer().visit(function_ast)
    function_ast = EnvironmentAnalyzer().visit(function_ast)
    # get consec stmt blablabla
    # get input/output blabla
    return function_ast
