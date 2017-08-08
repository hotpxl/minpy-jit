#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import copy
import inspect
import textwrap
import functools
import contextlib


def parse_function_definition(func):
    source_code = textwrap.dedent(inspect.getsource(func))
    function_ast = ast.parse(source_code, mode='exec')
    return function_ast


def evaluate_function_definition(function_ast, global_namespace,
                                 closure_parameters, closure_arguments):
    function_name = function_ast.body[0].name
    evaluation_context = ast.Module(body=[
        ast.FunctionDef(
            name='evaluation_context',
            args=ast.arguments(
                args=[
                    ast.arg(arg=i, annotation=None) for i in closure_parameters
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]),
            body=[
                function_ast.body[0],
                ast.Return(value=ast.Name(id=function_name, ctx=ast.Load()))
            ],
            decorator_list=[],
            returns=None)
    ])
    ast.fix_missing_locations(evaluation_context)
    local_namespace = {}
    exec(
        compile(evaluation_context, filename='<ast>', mode='exec'),
        global_namespace, local_namespace)
    ret = local_namespace['evaluation_context'](*closure_arguments)
    return ret


def add_type_tracing(function_ast):
    type_traced_function_ast = copy_ast(function_ast)
    closure_parameters = []
    closure_arguments = []
    closure_parameters.append('__type_tracing')

    node_cells = []

    def type_tracing(expr, node_id):
        node_cells[node_id].type = type(expr)
        return expr

    closure_arguments.append(type_tracing)

    class NodeTransformer(ast.NodeTransformer):
        def trace_node(self, node):
            node = self.generic_visit(node)
            # Do not add tracing if it already has type information.
            if hasattr(node.stem_node, 'type'):
                return node
            node_cells.append(node.stem_node)
            ret = ast.Call(
                func=ast.Name(id='__type_tracing', ctx=ast.Load()),
                args=[node, ast.Num(n=len(node_cells) - 1)],
                keywords=[])
            return ret

        # Do not recurse into inner definitions.
        def visit_FunctionDef(self, node):
            return node

        def visit_AsyncFunctionDef(self, node):
            return node

        def visit_ClassDef(self, node):
            return node

        def visit_Lambda(self, node):
            return node

        def visit_BoolOp(self, node):
            return self.trace_node(node)

        def visit_BinOp(self, node):
            return self.trace_node(node)

        def visit_UnaryOp(self, node):
            return self.trace_node(node)

        def visit_IfExp(self, node):
            return self.trace_node(node)

        def visit_Call(self, node):
            return self.trace_node(node)

        def visit_Name(self, node):
            if not isinstance(node.ctx, ast.Load):
                return node
            else:
                return self.trace_node(node)

    type_traced_function_ast.body[0].body = [
        NodeTransformer().visit(i)
        for i in type_traced_function_ast.body[0].body
    ]
    return (type_traced_function_ast, closure_parameters, closure_arguments)


def add_function_tracing(function_ast):
    function_traced_function_ast = copy_ast(function_ast)
    closure_parameters = []
    closure_arguments = []
    closure_parameters.append('__function_tracing')
    node_cells = []

    def function_tracing(f, node_id):
        node_cells[node_id].ref = f
        return f

    closure_arguments.append(function_tracing)

    class NodeTransformer(ast.NodeTransformer):
        # Do not recurse into inner definitions.
        def visit_FunctionDef(self, node):
            return node

        def visit_AsyncFunctionDef(self, node):
            return node

        def visit_ClassDef(self, node):
            return node

        def visit_Lambda(self, node):
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            # Do not add tracing if it already has function information.
            if hasattr(node.stem_node, 'ref'):
                return node
            node_cells.append(node.stem_node)
            node.func = ast.Call(
                func=ast.Name(id='__function_tracing', ctx=ast.Load()),
                args=[node.func, ast.Num(n=len(node_cells) - 1)],
                keywords=[])
            return node

    function_traced_function_ast.body[0].body = [
        NodeTransformer().visit(i)
        for i in function_traced_function_ast.body[0].body
    ]
    return (function_traced_function_ast, closure_parameters,
            closure_arguments)


def pretty_print(node,
                 annotate_fields=True,
                 include_attributes=False,
                 extra_attributes=['type', 'ref'],
                 indent='  '):
    def format(node, level=0):
        if isinstance(node, ast.AST):
            fields = [(i, format(j, level)) for i, j, in ast.iter_fields(node)]
            if include_attributes and node._attributes:
                fields.extend([(i, format(getattr(node, i), level))
                               for i in node._attributes])
            for i in extra_attributes:
                if hasattr(node, i):
                    fields.append((i, getattr(node, i).__name__))
            return ''.join([
                type(node).__name__, '(',
                ', '.join(('{}={}'.format(*field) for field in fields)
                          if annotate_fields else (i for _, i in fields)), ')'
            ])
        elif isinstance(node, list):
            lines = [indent * (level + 2) + format(i, level + 2) for i in node]
            if 0 < len(lines):
                return '[\n' + ',\n'.join(lines) + '\n' + indent * (
                    level + 1) + ']'
            else:
                return '[]'
        else:
            return repr(node)

    if not isinstance(node, ast.AST):
        raise TypeError(
            'Expected ast.AST, got {}.'.format(type(node).__name__))
    return format(node)


def tree_print(node, extra_attributes=['type', 'ref']):
    def tree_print_lines(node):
        childs = list(map(tree_print_lines, ast.iter_child_nodes(node)))
        fields = list(iter_non_node_children(node))
        for i in extra_attributes:
            if hasattr(node, i):
                fields.append((i, getattr(node, i).__name__))
        ret = [
            type(node).__name__ + '(' +
            ', '.join(map(lambda pair: '{}={}'.format(*pair), fields)) + ')'
        ]
        for c in childs[:-1]:
            for i, j in enumerate(c):
                ret.append(('+--' if i == 0 else '|  ') + j)
        if 0 < len(childs):
            for i, j in enumerate(childs[-1]):
                ret.append(('+--' if i == 0 else '   ') + j)
        return ret

    return '\n'.join(tree_print_lines(node))


def copy_ast(function_ast):
    original_nodes = []

    class NodeSequencer(ast.NodeVisitor):
        def generic_visit(self, node):
            original_nodes.append(node)
            for i in ast.iter_child_nodes(node):
                self.visit(i)

    NodeSequencer().visit(function_ast)

    new_ast = copy.deepcopy(function_ast)

    class NodeTransformer(ast.NodeTransformer):
        def generic_visit(self, node):
            n = original_nodes.pop(0)
            node.original_node = n
            node.stem_node = getattr(n, 'stem_node', n)
            for i in ast.iter_child_nodes(node):
                self.visit(i)
            return node

    NodeTransformer().visit(new_ast)

    return new_ast


def return_on_reentrance(f):
    reentry = False

    @functools.wraps(f)
    def wrapper(arg):
        nonlocal reentry
        if reentry:
            return arg
        reentry = True
        ret = f(arg)
        reentry = False
        return ret

    @contextlib.contextmanager
    def reentrance_guard():
        nonlocal reentry
        old_reentry, reentry = reentry, True
        yield
        reentry = False

    wrapper.reentrance_guard = reentrance_guard

    return wrapper


def iter_non_node_children(node):
    for name, field in ast.iter_fields(node):
        if not isinstance(field, (ast.AST, list)):
            yield name, field
        elif isinstance(field, list):
            for item in field:
                if not isinstance(item, ast.AST):
                    yield name, item
