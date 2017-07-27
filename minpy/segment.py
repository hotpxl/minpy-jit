from __future__ import print_function
import ast
import types
import inspect

from collections import OrderedDict
from functools import wraps, reduce

from mxnet import nd

def atomic(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return wrapper
    return wrapper

def segment(node, global_namespace, visualize_mode=False):
    """Given an annotated AST, return a segmented AST

    This is the only interface to call after type annotation.
    Parameters
    ----------
    node: annotated AST node

    global_namespace: globa NameSpace of the function

    visualize_mode: print the segments if True


    Returns
    -------
    ast: segmented AST

    """
    def is_ndarray_type(x):
        return x.type == nd.NDArray
    return do_segment(node, global_namespace, is_ndarray_type, visualize_mode)

def test_segment(f, visualize_mode=False):
    """The function to test segment implementation

    The interface to test segment function without the annotation information
    The input is the function object instead of parsed ast node.

    Parameters
    ----------
    f: function to segment

    visualize_mode: print the segments if True
    """
    def is_ndarray_type_fake(x):
        # The ast is not annotated with type attribute
        return True
    node = ast.parse(inspect.getsource(f))
    node = do_segment(node, f.__globals__, is_ndarray_type_fake, visualize_mode)
    node.body[0].name += '_rewritten'
    func_name = node.body[0].name
    global_namespace = f.__globals__.copy()
    exec(
        compile(node, filename='<ast>', mode='exec'), global_namespace)

    def wrapper(*args, **kwargs):
        return global_namespace[func_name](*args, **kwargs)
    return wrapper


def do_segment(node, global_namespace, is_ndarray_type, visualize_mode):
    """Segment a node given its information collected in the runtime

    Parameters
    ------
    node:  ast.Ast

    global_namespace: function's module namespace

    is_ndarray_type: the func for checking the types for ast.Name, ast.Call return values

    # Puzzles:whether we need to record the function type?
        - it depends on whether @Jit/Atomic decorator is removed in the annotation stage.

    # Potential missing Input: variable liveness, i.e. whether one variable is accessed in the future or not
        - Let's assume all the output variables are accessed, i.e. the worst case
    """

    class AstTypeHelper:
        """The helper class that categorizes the AST classes by purposes"""
        seg_list = (
            # Module, Function, Class Related
            ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.arguments, ast.ClassDef,
            # Control Flow
            ast.IfExp, ast.Return, ast.Delete, ast.For, ast.AsyncFor, ast.While, ast.If,
            # Special Ops
            ast.With, ast.AsyncWith, ast.Raise, ast.Try, ast.Assert, ast.Import, ast.ImportFrom,
            # More Special Ops
            ast.Global, ast.Nonlocal, ast.Expr, ast.Pass, ast.Break, ast.Continue, ast.Str
        )

        func_checking_list = (ast.Call)

        # Check its or the computed result's type
        type_checking_list = (
            ast.Name,
            ast.BinOp,
            ast.UnaryOp,
            ast.Compare,
            ast.BoolOp,
            ast.Attribute,
            ast.Subscript)

        # Types that are not doing any checking:
        non_check_list = (
            # Assignment
            ast.Assign, ast.AugAssign,
            # Basic Data Structure
            ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Num,
            # Context Related Function
            ast.Load, ast.Store,
            # Operators that are covered by BinOp and UnaryOp
            ast.operator, ast.boolop, ast.unaryop, ast.cmpop,
            # arg
            ast.arg
        )

        skip_fuse_list = (ast.arg, ast.Name)

        @staticmethod
        def fuse_check(node):
            if isinstance(node, AstTypeHelper.seg_list):
                return False

            if isinstance(node, AstTypeHelper.func_checking_list):
                return is_atomic_func(node)

            if isinstance(node, AstTypeHelper.type_checking_list):
                return is_ndarray_type(node)

            if isinstance(node, AstTypeHelper.non_check_list):
                return True

            raise TypeError('Type {} not handled yet in fuse check'.format(type(node)))

    def is_atomic_func(node):
        # TODO: add a cache here
        if (isinstance(node.func, ast.Attribute)):
            # return node.type == NDArray Type
            return True

        try:
            f = global_namespace[node.func.id]
            assert(isinstance(f, types.FunctionType))
            func_def = ast.parse(inspect.getsource(f))
            for e in func_def.body[0].decorator_list:
                if (global_namespace[e.id] == atomic):
                    return True
        except Exception:
            # f is a built-in func or f is not a funct
            print('is_atomic_func fails', type(node))
            return False
        print('is_atomic_func fails', type(node))
        return False


    segment_id = 0

    def fuse(node):
        """Fuse the node or the list of nodes

        Parameters
        ------
        node:  ast.Ast  | the list of ast.Ast

        The expression could be re-writen to 'run_segment(inputs)'
        The assignment statement should kept its outputs  'outputs = run_segments(inputs)'
        """

        nonlocal segment_id
        # Do nothing on unit op
        if (isinstance(node, AstTypeHelper.skip_fuse_list)):
            return node

        if (visualize_mode):
            print('Segment {} info: '.format(segment_id))
            segment_id += 1
            ins, outs = infer_inputs_and_outputs_given_nodes(node)
            print('\tinput list: ', ins)
            print('\toutput list: ', outs)
            if isinstance(node, list):
                for i, e in enumerate(node):
                    print('\tast node {} '.format(i), e)
            else:
                print('\tast node 0 ', node)
            print('\n')

        # TODO: Add aggregation code here
        return node

    def get_consec_assign(values, signs):
        pos, leng = (0, 0)
        while pos < len(values):
            if (isinstance(values[pos], ast.Assign)):
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
        atom_signs = {}
        all_atom = True
        for name, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                atom_signs[name] = iterate_and_fuse(value)
                all_atom &= atom_signs[name]
            elif isinstance(value, list):
                atom_signs[name] = []
                for i, e in enumerate(value):
                    if (isinstance(e, ast.AST)):
                        atom_signs[name].append(iterate_and_fuse(e))
                        all_atom &= atom_signs[name][i]

        if (all_atom):
            all_atom &= AstTypeHelper.fuse_check(node)

        # If all child nodes are atomic and the operation itself is good, then
        # leave it to its parent
        if all_atom:
            return True

        # Rule 1: fuse consecutive atomic asssign statements in the body
        if hasattr(node, 'body'):
            values = node.body
            signs = atom_signs['body']
            removed_num = 0
            for (st, leng) in get_consec_assign(values, signs):
                if not visualize_mode:
                    st -= removed_num
                    values[st] = fuse(values[st:st + leng])
                    # Already being fused
                    signs[st] = False
                    removed_num += leng - 1
                    del values[st + 1:st + leng - 1]
                    del signs[st + 1:st + leng - 1]
                else:
                    fuse(values[st:st + leng])
                    for i in range(st, st + leng):
                        signs[i] = False

        for name, value in ast.iter_fields(node):
            if isinstance(value, ast.AST) and (atom_signs[name]):
                new_value = fuse(value)
                setattr(node, name, new_value)
            elif isinstance(value, list):
                new_values = []
                for i, e in enumerate(value):
                    if isinstance(e, ast.AST) and atom_signs[name][i]:
                        e = fuse(e)
                    new_values.append(e)
                value[:] = new_values
        return False

    if (iterate_and_fuse(node)):
        node = fuse(node)
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
            outs = [name.id for name in node.targets]
            return ins, outs
        elif isinstance(node, ast.expr):
            return infer_inputs_given_exprs(node), []
        else:
            raise TypeError('Type {} not handled yet in inputs and outputs inference'.format(type(node)))

    def infer_inputs_given_exprs(expr):
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
            return list(OrderedDict.fromkeys(reduce(lambda x, y: x + y,
                                                    [infer_inputs_given_exprs(e) for e in expr])))
        elif isinstance(expr, ast.Call):
            return infer_inputs_given_exprs(expr.args)
        elif isinstance(expr, ast.BinOp):
            return infer_inputs_given_exprs([expr.left, expr.right])
        elif isinstance(expr, ast.UnaryOp):
            return infer_inputs_given_exprs(expr.operand)
        elif isinstance(expr, ast.Tuple):
            return infer_inputs_given_exprs(expr.elts)
        elif isinstance(expr, ast.Attribute):
            # Assumption: left operand is a Name
            assert(isinstance(expr.expr, ast.Name))
            return [expr.expr.id + "." + expr.attr]
        elif isinstance(expr, ast.Subscript):
            # Assumption: left operand is a Name
            assert(isinstance(expr.expr, ast.Name))
            return [expr.expr.id + "_subscript_"]
        elif isinstance(expr, ast.Name):
            return [expr.id]
        elif isinstance(expr, (ast.Num, ast.Str, ast.Bytes)):
            return []

        raise TypeError('{} not handled yet in inference of inputs'.format(type(expr)))

    if isinstance(nodes, list):
        ins = []
        outs = []
        for node in nodes:
            sub_ins, sub_outs = infer_inputs_and_outputs_given_node(node)
            ins += [x for x in sub_ins if x not in outs]
            outs += sub_outs
        return list(
            OrderedDict.fromkeys(ins)), list(
            OrderedDict.fromkeys(outs))
    else:
        return infer_inputs_and_outputs_given_node(nodes)
