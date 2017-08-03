from __future__ import print_function
import ast
import inspect
import textwrap
from .segment import do_segment


class Module:
    F = __import__('mxnet').ndarray

    def __init__(self):
        super().__setattr__('_modules', set())
        super().__setattr__('_parameters', set())

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

    def __setattr__(self, attr, value):
        assert attr not in self._modules and \
          attr not in self._parameters
        if isinstance(value, Module):
            compiled = value.compile()
            super().__setattr__(attr, compiled)
            self._modules.add(attr)
        elif isinstance(value, self.F.NDArray):
            self._parameters.add(attr)
            super().__setattr__(attr, value)
        else:
            # attributes other than modules and parameters are permitted
            # but `self.compile` will raise an exception if it detects these attributes in `self.forward`
            super().__setattr__(attr, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        """ To be overridden. These assumptions applies:
        - No `*args` or `**kwargs`
        - No `setattr` of attributes in `self._modules` or `self._parameters`
        - No `setattr` or `getattr` of attrtibutes other than modules and parameters

        Parameters
        ----------

        """
        raise NotImplementedError()

    def compile(self):
        """ Convert `self.forward` to static function and segment the converted function.
        """
        # CR(gaiyu): the main logic here is:
        # 1. get source code
        # 2. pre processing (dedent)
        # 3. compile
        # At least part 3 you can use common code. Or I can hook that up later.
        source = inspect.getsource(self.forward)
        source = textwrap.dedent(source)
        node = ast.parse(source)
        f_def = node.body[0]
        self_id = f_def.args.args[0].arg
        arg_id_list = []
        namespace = {}

        class AttributeTransformer(ast.NodeTransformer):
            # TODO (gaiyu) handle attributes other than parameters
            def __init__(self, module):
                super().__init__()
                self._module = module

            def visit_Attribute(self, node):
                self.generic_visit(node)
                if isinstance(node.value,
                              ast.Name) and node.value.id == self_id:
                    if node.attr == 'F':  # mxnet.ndarray
                        transformed = ast.Name(id='F', ctx=ast.Load())
                        namespace['F'] = self._module.F
                    elif node.attr in self._module._parameters:
                        arg_id = '__arg%s' % node.attr
                        arg_id_list.append(arg_id)
                        arg = getattr(self._module, node.attr)
                        namespace[arg_id] = arg
                        transformed = ast.Name(
                            id=arg_id,
                            ctx=ast.Load())  # node.ctx must be ast.Load
                    elif node.attr in self._module._modules:
                        module_id = '__module%s' % node.attr
                        module = getattr(self._module, node.attr)
                        namespace[module_id] = module
                        transformed = ast.Name(
                            id=module_id,
                            ctx=ast.Load())  # node.ctx must be ast.Load
                    else:
                        raise Exception(
                            'Attributes other than parameters and modules: %s'
                            % node.attr)
                    transformed = ast.copy_location(transformed, node)
                    return transformed
                return node

        node = AttributeTransformer(self).visit(node)
        f_def.args.args.pop(0)
        if len(f_def.args.args) > 0:
            locator = f_def.args.args[-1]  # arguments on multiple lines
        else:
            locator = f_def
        for arg_id in arg_id_list:
            arg_node = ast.arg(arg=arg_id)
            arg_node = ast.copy_location(arg_node, locator)
            f_def.args.args.append(arg_node)
            default_node = ast.Name(id=arg_id, ctx=ast.Load())
            default_node = ast.copy_location(default_node, locator)
            f_def.args.defaults.append(default_node)
        '''
        always_true = lambda _: True
        node = do_segment(node, always_true, always_true, True)
        '''
        exec(compile(node, 'ast', 'exec'), namespace)
        self.__class__.__call__ = staticmethod(namespace['forward'])
        return namespace['forward']


class Linear(Module):
    def __init__(self, in_features, out_features):
        """ Affine transformation.

        Parameters
        ----------
        in_features: dimension of input data
        out_features: dimension of output data
        """

        super().__init__()
        self._weight = self.F.zeros((in_features, out_features))
        self._bias = self.F.zeros((out_features, ))

    def forward(self, data):
        """ Forward.

        Parameters
        ----------
        data: NDArray
        """

        return self.F.dot(data, self._weight) + self._bias
