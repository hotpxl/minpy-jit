#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import ast
import types
import inspect

from collections import OrderedDict
from functools import wraps, reduce

from mxnet import nd
from . import core

_segment_cnt = 0


# CR(haoran): i feel it is confusing with both `__dict__` and `getattr`, `setattr`
# let's use `get/setattr` all the way and avoid `__dict__` directly
def segment_reform(function_ast, print_new_segment):
    class InfoHelper:
        def __init__(self,
                     name,
                     init_value,
                     default_value,
                     update_func=None,
                     rewrite_cond=None):
            self.name = name
            self.init_value = init_value
            self.default_value = default_value
            self.update_func = update_func
            self.rewrite_cond = rewrite_cond

        def set(self, node, value):
            # CR(haoran): can we just do `setattr(node, name, value)`?
            node.__dict__[self.name] = value

        def get(self, node):
            # CR(haoran): same here, use getattr uniformly.
            # `getattr(node, name, default_value)`
            return node.__dict__.get(self.name, self.default_value)

        def do_rewrite(self, node):
            return self.rewrite_cond(self.get(node))

        def update(self, *values):
            return self.update_func(*values)

    class InfoCollector(ast.NodeTransformer):
        def __init__(self, info_helper, funcs=[]):
            super(InfoCollector, self).__init__()
            self.info_helper = info_helper
            self.funcs = {func.__name__: func for func in funcs}

        def collect_info(self, node, attrs=[], funcs=[]):
            self.generic_visit(node)
            info = self.info_helper.init_value
            for name in attrs:
                child = getattr(node, name)
                if isinstance(child, list):
                    for e in child:
                        info = self.info_helper.update(info,
                                                       self.info_helper.get(e))
                else:
                    info = self.info_helper.update(info,
                                                   self.info_helper.get(child))

            for name in funcs:
                info = self.info_helper.update(info, self.funcs[name](node))

            self.info_helper.set(node, info)
            return node

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            return node

        def visit_If(self, node):
            self.generic_visit(node)
            return node

        def visit_Assign(self, node):
            return self.collect_info(node, attrs=['value'])

        def visit_Call(self, node):
            return self.collect_info(
                node, attrs=['args'], funcs=['is_atomic_func'])

        def visit_BinOp(self, node):
            return self.collect_info(
                node, attrs=['left', 'right'], funcs=['is_ndarray_type'])

        def visit_Name(self, node):
            return self.collect_info(node, funcs=['is_ndarray_type'])

        def visit_Num(self, node):
            return self.collect_info(node)

        def visit_Attribute(self, node):
            # Treat an attribute expr as a whole
            return self.collect_info(node, funcs=['is_ndarray_type'])

        def visit_Subscript(self, node):
            # Treat a subscript expr as a whole
            return self.collect_info(node, funcs=['is_ndarray_type'])

    class NodeRewriter(ast.NodeTransformer):
        def __init__(self, info_helper):
            super(NodeRewriter, self).__init__()
            self.info_helper = info_helper

        def fuse_consecutive_assignments(self, stmts):
            def make_ast_call(func_name, ins, outs):
                return ast.Assign(
                    targets=[
                        ast.Tuple(
                            elts=[
                                ast.Name(id=e, ctx=ast.Store()) for e in outs
                            ],
                            ctx=ast.Store())
                    ],
                    value=ast.Call(
                        func=ast.Name(id=func_name, ctx=ast.Load()),
                        args=[ast.Name(id=e, ctx=ast.Load()) for e in ins],
                        keywords=[]))

            def make_ast_function_def(func_name, stmts, ins, outs):
                return ast.FunctionDef(
                    name=func_name,
                    args=ast.arguments(
                        args=[ast.arg(arg=e, annotation=None) for e in ins],
                        vararg=None,
                        kwonlyargs=[],
                        kw_defaults=[],
                        kwarg=None,
                        defaults=[]),
                    body=[
                        *stmts,
                        ast.Return(value=ast.Tuple(
                            elts=[
                                ast.Name(id=e, ctx=ast.Load()) for e in outs
                            ],
                            ctx=ast.Load()))
                    ],
                    decorator_list=[],
                    returns=None)

            def fuse(nodes):
                ins, outs = infer_inputs_and_outputs_given_nodes(nodes)

                global _segment_cnt
                if print_new_segment:
                    print('Segment {} info: '.format(_segment_cnt))
                    print('\tinput list: ', ins)
                    print('\toutput list: ', outs)
                    for i, e in enumerate(nodes):
                        print('\t ast node {} {}'.format(i, type(e).__name__))
                    print('\n')
                func_name = '_fuse_func_{}'.format(_segment_cnt)
                _segment_cnt += 1

                func_def = make_ast_functionDef(func_name, nodes, ins, outs)
                call_node = make_ast_call(func_name, ins, outs)
                new_funcdefs.append(func_def)
                return call_node

            def get_consecutive_assignments(stmts):
                pos, leng = (0, 0)
                while pos < len(stmts):
                    if isinstance(stmts[pos],
                                  ast.Assign) and self.info_helper.do_rewrite(
                                      stmts[pos]):
                        leng += 1
                    else:
                        if leng > 0:
                            yield (pos - leng, leng)
                            leng = 0
                    pos += 1
                if leng > 0:
                    yield (pos - leng, leng)

            removed_num = 0
            for (st, leng) in get_consecutive_assignments(stmts):
                st -= removed_num
                stmts[st] = fuse(stmts[st:st + leng])
                removed_num += leng - 1
                del stmts[st + 1:st + leng]

        def visit_FunctionDef(self, node):
            if not jit_helper.get(node):
                return node
            self.generic_visit(node)
            self.fuse_consecutive_assignments(node.body)
            return node

        def visit_If(self, node):
            self.generic_visit(node)
            self.fuse_consecutive_assignments(node.body)
            self.fuse_consecutive_assignments(node.orelse)
            return node

    def is_ndarray_type(node):
        return hasattr(node, 'type') and issubclass(node.type, nd.NDArray)

    def is_atomic_func(node):
        # CR(haoran): ditto
        # 1. `__dict__` always exists
        # 2. nd.__dict__.values() might be a performance issue (everything else is O(1) and this is O(n))
        if hasattr(node, 'ref') and hasattr(node.ref, '__dict__'):
            return node.ref.__dict__.get(
                '__minpy_atomic', False) or node.ref in nd.__dict__.values()
        else:
            return False

    fuse_helper = InfoHelper('fuse_as_whole', True, False, lambda x, y: x & y,
                             lambda x: x)
    jit_helper = InfoHelper('jit_func', True, False)

    collector = InfoCollector(
        fuse_helper, funcs=[is_ndarray_type, is_atomic_func])
    collector.generic_visit(function_ast)

    rewriter = NodeRewriter(fuse_helper)
    new_funcdefs = []
    jit_helper.set(function_ast.body[0], jit_helper.init_value)
    rewriter.generic_visit(function_ast)

    function_ast.body[0].body[0:0] = new_funcdefs
    return function_ast


def segment(function_ast, print_new_segment):
    """Segment a function ast given its information collected in the runtime

    Parameters
    ------
    function_ast:  function Ast

    print_new_segment: print new segments if True
    """

    class AstTypeHelper:
        """The helper class that categorizes the AST classes by purposes"""
        always_segment_types = (
            # Module, Function, Class Related
            ast.Module,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.Lambda,
            ast.arguments,
            ast.ClassDef,
            # Control Flow
            ast.IfExp,
            ast.Return,
            ast.Delete,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.If,
            # Special Ops
            ast.With,
            ast.AsyncWith,
            ast.Raise,
            ast.Try,
            ast.Assert,
            ast.Import,
            ast.ImportFrom,
            ast.keyword,
            # More Special Ops
            ast.Global,
            ast.Nonlocal,
            ast.Expr,
            ast.Pass,
            ast.Break,
            ast.Continue,
            ast.Str)

        # Check its or the computed result's type
        maybe_segment_types = (ast.Name, ast.BinOp, ast.UnaryOp, ast.Compare,
                               ast.BoolOp, ast.Attribute, ast.Subscript)

        # Types that are not doing any checking:
        never_segment_types = (
            # Assignment
            ast.Assign,
            ast.AugAssign,
            # Basic Data Structure
            ast.List,
            ast.Tuple,
            ast.Dict,
            ast.Set,
            ast.Num,
            # Context Related Function
            ast.Load,
            ast.Store,
            # Operators that are covered by BinOp and UnaryOp
            ast.operator,
            ast.boolop,
            ast.unaryop,
            ast.cmpop,
            # arg
            ast.arg)

        # TODO: handle fuse of expression
        never_fuse_types = (ast.arg, ast.Name, ast.expr, ast.expr_context,
                            ast.operator, ast.cmpop)

        @classmethod
        def fuse_check(cls, node):
            if getattr(node, 'fused', False):
                return False

            if isinstance(node, cls.always_segment_types):
                return False

            if isinstance(node, ast.Call):
                return is_atomic_func(node)

            if isinstance(node, cls.maybe_segment_types):
                return is_ndarray_type(node)

            if isinstance(node, cls.never_segment_types):
                return True

            raise TypeError('Type {} not handled yet in fuse check'.format(
                type(node).__name__))

    def is_ndarray_type(node):
        # CR(haoran): why FunctionType is ndarray type?
        # use `issubclass`
        # XCR(yutian): The previous commit is WIP. See more in #L237
        return hasattr(node, 'type') and issubclass(node.type, nd.NDArray)

    def is_atomic_func(node):
        if hasattr(node, 'ref') and hasattr(node.ref, '__dict__'):
            return node.ref.__dict__.get(
                '__minpy_atomic', False) or node.ref in nd.__dict__.values()
        else:
            return False

    def make_fuse_func_def(func_name, statements, ins, outs):
        return ast.FunctionDef(
            name=func_name,
            args=ast.arguments(
                args=[ast.arg(arg=e, annotation=None) for e in ins],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]),
            body=[
                *statements,
                ast.Return(value=ast.Tuple(
                    elts=[ast.Name(id=e, ctx=ast.Load()) for e in outs],
                    ctx=ast.Load()))
            ],
            decorator_list=[],
            returns=None)

    def make_call(func_name, ins, outs):
        return ast.Assign(
            targets=[
                ast.Tuple(
                    elts=[ast.Name(id=e, ctx=ast.Store()) for e in outs],
                    ctx=ast.Store())
            ],
            value=ast.Call(
                func=ast.Name(id=func_name, ctx=ast.Load()),
                args=[ast.Name(id=e, ctx=ast.Load()) for e in ins],
                keywords=[]))

    new_funcdefs = []

    def fuse(nodes):
        """Fuse the node or the list of nodes

        Parameters
        ------
        nodes:  the list of ast nodes

        The expression could be re-writen to 'run_segment(inputs)'
        The assignment statement should kept its outputs  'outputs = run_segments(inputs)'
        """
        # Do nothing on unit op
        if len(nodes) == 1 and isinstance(nodes[0],
                                          AstTypeHelper.never_fuse_types):
            return nodes[0]

        ins, outs = infer_inputs_and_outputs_given_nodes(nodes)

        global _segment_cnt
        if print_new_segment:
            print('Segment {} info: '.format(_segment_cnt))
            print('\tinput list: ', ins)
            print('\toutput list: ', outs)
            for i, e in enumerate(nodes):
                print('\t ast node {} {}'.format(i, type(e).__name__))
            print('\n')
        func_name = '_fuse_func_{}'.format(_segment_cnt)
        _segment_cnt += 1

        # TODO: handle subscript and attribute opertion
        func_def = make_fuse_func_def(func_name, nodes, ins, outs)
        call_node = make_call(func_name, ins, outs)
        func_def.fused = True
        call_node.fused = True
        new_funcdefs.append(func_def)
        return call_node

    def get_consecutive_assignments(values, signs):
        pos, leng = (0, 0)
        while pos < len(values):
            if isinstance(values[pos], ast.Assign) and signs[pos]:
                leng += 1
            else:
                if leng > 0:
                    yield (pos - leng, leng)
                    leng = 0
            pos += 1

        if leng > 0:
            yield (pos - leng, leng)

    def iterate_and_fuse(node):
        """
        Iterate over the AST by DFS and fuse the expr/stmt

        Parameters
        ------
        node
            The ast node

        Returns
        ------
        bool
            True, if all the children nodes can be fused. And fusion is done by some of its ancestor nodes
            False, otherwise
        """
        if getattr(node, 'fused', False):
            return False

        atom_signs = {}
        fuse_entire_node = True
        # CR(haoran): use iter_child_nodes so you don't have to check
        # whether it's a list
        # (https://github.com/python/cpython/blob/3.6/Lib/ast.py#L178)
        # XCR(yutian): Yes, I understand the benefit of iter_child_nodes
        # Actually, the code below is copied from its definition
        # The initial purpose of writing in the list-checking way is to expose the list information,
        # i.e. to know which nodes belong to the same list in order to fuse them
        # Under the current fusing rules, iter_child_nodes is actually sufficient.
        # I suggest we keep this and change to iter_child_nodes later if above observation still holds
        for name, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                # ad-hoc: skip func attr of ast.Call,
                # which could be an ast.Name with the function type
                if isinstance(node, ast.Call) and name == 'func':
                    atom_signs[name] = False
                    continue
                atom_signs[name] = iterate_and_fuse(value)
                fuse_entire_node &= atom_signs[name]
            elif isinstance(value, list):
                atom_signs[name] = []
                # ad-hoc: left operand of assign has no type
                if isinstance(node, ast.Assign) and name == 'targets':
                    atom_signs[name] = [False] * len(value)
                    continue
                for i, e in enumerate(value):
                    if isinstance(e, ast.AST):
                        atom_signs[name].append(iterate_and_fuse(e))
                        fuse_entire_node &= atom_signs[name][i]

        if fuse_entire_node:
            fuse_entire_node &= AstTypeHelper.fuse_check(node)

        # If all child nodes are atomic and the operation itself is good, then
        # leave it to its parent
        if fuse_entire_node:
            return True

        # Rule 1: fuse consecutive atomic asssign statements in the body
        for attr in ['body', 'orelse']:
            if hasattr(node, attr):
                values = getattr(node, attr)
                signs = atom_signs[attr]
                removed_num = 0
                for (st, leng) in get_consecutive_assignments(values, signs):
                    st -= removed_num
                    values[st] = fuse(values[st:st + leng])
                    signs[st] = False
                    removed_num += leng - 1
                    del values[st + 1:st + leng]
                    del signs[st + 1:st + leng]

        # CR(haoran): seems you are compiling atomic functions
        # individually. Consider this case:
        #
        # a = a + 1 assignment
        # atomic_fn(a) atomic call
        # a = a + 1 assignment
        #
        # Would you be able to fuse all three together?
        # XCR(yutian): i thnk atomic call could be handled if the definition of rule 1 is extended,
        # i.e. fuse consecutive atomic assignments/expressions
        # XCR(haoran): tested. doesn't work.
        # i feel like you could separate treatment of stmt and expr
        # refer to the AST documentation page of difference between two
        # expr is almost always a nested structure and can be dealt with easily using recursion
        # stmt is only used at top-level (since we are only considering one level of function definition)
        # write function that handle a couple of cases of stmt, and then write another one to
        # handle expr cases
        for name, value in ast.iter_fields(node):
            if isinstance(value, ast.AST) and (atom_signs[name]):
                new_value = fuse([value])
                setattr(node, name, new_value)
            elif isinstance(value, list):
                new_values = []
                for i, e in enumerate(value):
                    if isinstance(e, ast.AST) and atom_signs[name][i]:
                        e = fuse([e])
                    new_values.append(e)
                value[:] = new_values
        return False

    if iterate_and_fuse(function_ast):
        function_ast = fuse([function_ast])
    # Insert new fuse function definitions to function body
    function_ast.body[0].body[0:0] = new_funcdefs
    return function_ast


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
        elif isinstance(node, ast.expr):
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
        elif isinstance(expr, ast.BinOp):
            return collect_names_given_exprs([expr.left, expr.right])
        elif isinstance(expr, ast.UnaryOp):
            return collect_names_given_exprs(expr.operand)
        elif isinstance(expr, ast.Tuple):
            return collect_names_given_exprs(expr.elts)
        # CR(haoran): it would be more convenient to handle nested
        # attributes/subscripts as a whole. for example,
        # `nd.random.normal` would be treated as one single input,
        # same for `array[2][3][4]`.
        # XCR(yutian): handled in reformed segment
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
