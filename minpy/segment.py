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
_ndarray_funcs = nd.__dict__.values()


def segment_reform(function_ast, print_new_segment):
    # CR(haoran): I feel this class definition is largly
    # unnecessary. The functionality is quite specific and doesn't
    # offer much generalization. Besides, try use `map` and `reduce`
    # to generalize on functions instead of data structure, i.e. try
    # to write in a functional fashion.
    # XCR(yutian): The main reason to abstract this class out is I
    # think it may be helpful when the program needs to walk through
    # the ast nodes for collecting some information/doing
    # computationsm, which relies on its children's result.  I think
    # this scenario may be common.
    # XCR(haoran): "it may be helpful when the program needs to walk
    # through the ast nodes" -> that's why we have ast.NodeVisitor and
    # ast.NodeTransformer
    # if you look at it more closely, you will find that it is not at
    # all generic. InfoCollector's visit functions are not uniform;
    # they depend on the exact type of node that is being visited. and
    # this logic is particular to segmentation logic. a good rule of
    # thumb is DRY (take it with a grain of salt though)
    # this is why i propose the separation of rules (when can a node
    # be fused)
    # WHILE you are separating the logic of aggregation of rules but
    # not the rules itself
    # i think it also deals with problems mentioned below (ln 120 and
    # 133). i'm still working on it. i'm trying to work from your
    # existing rules and see if i can come up with a SOUND (but not
    # COMPLETE) version. you might go ahead working on the codegen
    # part with minjie in the mean time. at least the code runs
    # smoothly now
    # XCR(yutian): the last part of my comment "for collecting some
    # information/doing computationsm, which relies on its children's
    # result" -> this is the purpose for infohelper, which is originally
    # written within visit_Node function.
    # Its genericness/versatility is heavily related to the type-tracing
    # result.
    # And you're right that it seems poor given current type info.
    # We would have better judgement on that once your changes are done.
    # If still poor, I would remove it.
    # For the point of "not the rules itself", I think it's possible
    # to add more rules by making classes like 'NodeRewriterRuleA',
    # 'NodeRewriterRuleB'.
    class InfoHelper():
        def __init__(self,
                     name,
                     init_value,
                     get_default_value,
                     update_func=None,
                     rewrite_cond=None):
            self._name = name
            self.init_value = init_value
            self._get_default_value = get_default_value
            self._update_func = update_func
            self._rewrite_cond = rewrite_cond

        def set(self, node, value):
            setattr(node, self._name, value)

        def get(self, node):
            return getattr(node, self._name, self._get_default_value)

        def do_rewrite(self, node):
            return self._rewrite_cond(self.get(node))

        def update(self, *values):
            return self._update_func(*values)

    class InfoCollector(ast.NodeTransformer):
        def __init__(self, info_helper, funcs=[]):
            super(InfoCollector, self).__init__()
            self._info_helper = info_helper
            self._funcs = {func.__name__: func for func in funcs}

        def _collect_info(self, node, attrs=[], funcs=[]):
            self.generic_visit(node)
            info = self._info_helper.init_value
            for name in attrs:
                child = getattr(node, name)
                if isinstance(child, list):
                    info = reduce(
                        self._info_helper.update,
                        [info] + list(map(self._info_helper.get, child)))
                else:
                    info = self._info_helper.update(
                        info, self._info_helper.get(child))

            info = reduce(
                self._info_helper.update, [info] +
                list(map(lambda name: self._funcs[name](node), funcs)))

            self._info_helper.set(node, info)
            return node

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            return node

        def visit_If(self, node):
            self.generic_visit(node)
            return node

        def visit_Assign(self, node):
            return self._collect_info(node, attrs=['value'])

        def visit_Call(self, node):
            # CR(haoran): atomic functions could also take lists or
            # dictionaries of ndarrays, or read-only objects. how do
            # you deal with that?
            # On the other hand, prevent stuff like `atomic(3 if
            # some_flag else 2, ...)` from fusing
            # I don't have a solution but i feel there is a simple
            # solution
            #
            # XCR(yutian): It doesn't check the input list yet.
            #
            # I don't have a simple solution yet.
            #
            # List several questions come to my mind:
            # - how to get the elements, and their types, of dict/list?
            # - how to figure which elements of the list/dict
            # are created inside the function?
            # - let's assume we could get the function definition,
            # how to handle the recursive function call?
            # - how to do above things in a simple way?
            return self._collect_info(
                node, attrs=['args'], funcs=['is_atomic_func'])

        def visit_BinOp(self, node):
            # CR(haoran): incorrect? numpy.ndarray + integer_literal
            # is also a valid fusable operation
            # XCR(yutian): fixed
            # XCR(haoran): this is incorrect either! The correct
            # condition is: either or both sides is NDArray. Not
            # including the case where both sides are numbers
            # XCR(yutian): Take a = b + (c + d), where b is NDArray
            # and c,d are numeric.
            # For Binary Op, we might allow both-numeric-value case
            # and add the NDArray checking at the very end, e.g.
            # the type of right operand of assignment operation
            # in this case.
            # This final checkingis missing at present. I'll work
            # on this.
            return self._collect_info(
                node, attrs=['left', 'right'], funcs=['is_ndarray_or_numeric'])

        def visit_Name(self, node):
            return self._collect_info(node, funcs=['is_ndarray_or_numeric'])

        def visit_Num(self, node):
            return self._collect_info(node)

        def visit_Attribute(self, node):
            # Treat an attribute expr as a whole
            return self._collect_info(node, funcs=['is_ndarray_or_numeric'])

        def visit_Subscript(self, node):
            # Treat a subscript expr as a whole
            return self._collect_info(node, funcs=['is_ndarray_or_numeric'])

    class NodeRewriter(ast.NodeTransformer):
        def __init__(self, info_helper):
            super(NodeRewriter, self).__init__()
            self._info_helper = info_helper

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

                func_def = make_ast_function_def(func_name, nodes, ins, outs)
                call_node = make_ast_call(func_name, ins, outs)
                new_funcdefs.append(func_def)
                return call_node

            def get_consecutive_assignments(stmts):
                pos, leng = (0, 0)
                while pos < len(stmts):
                    if isinstance(stmts[pos],
                                  ast.Assign) and self._info_helper.do_rewrite(
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

    def is_ndarray_or_numeric(node):
        return hasattr(node, 'type') and issubclass(node.type,
                                                    (nd.NDArray, int, float))

    def is_atomic_func(node):
        if hasattr(node, 'ref') and hasattr(node.ref, '__dict__'):
            return node.ref.__dict__.get('__minpy_atomic',
                                         False) or node.ref in _ndarray_funcs
        else:
            return False

    fuse_helper = InfoHelper('fuse_as_whole', True, False, lambda x, y: x & y,
                             lambda x: x)
    jit_helper = InfoHelper('jit_func', True, False)

    # CR(haoran): to my understanding, you need two kinds of information
    # 1. recursive call of "fusability". this is basically what
    # "fuse_helper" is doing right now. this acts on expressions,
    # because only expressions can nest in itself
    # 2. gather input and output arguments. this is what Rewriter is doing IIRC.
    # one implication is, this information is only gathered from
    # statements. BECAUSE, only assign statements can change the
    # environment/scope/$$\Gamma$$
    #
    # of course after two steps you then need to find consecutive
    # STATEMENTS and merge them together (ps here is bug, i mentioned
    # down in the old code)
    #
    # So my proposition is, instead of abstracting data structure into
    # InfoCollector and Helper, write the NodeTransformer or Visitor
    # as is and just hard code the logic. NodeTransformer/Visitor
    # itself is already using visitor pattern so should largely handle
    # the plumbing
    collector = InfoCollector(
        fuse_helper, funcs=[is_ndarray_or_numeric, is_atomic_func])
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
