#!/usr/bin/env python3 # noqa T499
#
# Copyright (c) XXXX-XXXX., Ltd
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""
Defines the global scope and function scope for Python AST.
"""
import ast
import typing
from typing import Any
from typing import Dict
from typing import List


class Scope:
    """
    Sets up scope in a function.
    """
    _f_locals: typing.List[Dict[str, Any]] = []

    def __init__(self, node: ast.AST, f_locals: Dict[str, Any]) -> None:
        self._node = node
        self._f_locals = f_locals  # noqa T484

    @classmethod
    def init_global(cls, f_locals) -> "Scope":
        """
        Initialization of globals
        """
        scope = cls("<module>", f_locals)  # noqa T484
        return scope

    def assign_name(self, name, value):
        """
        Updates the name stack with the given python name and air type.
        """
        self._f_locals[name] = value

    def get_locals(self):
        """Return the local variables of this scope."""
        return self._f_locals


class ScopeStack:
    """
    Sets up variable stack in a function.
    """

    _stack: List[Scope] = []
    _global_scope: Scope = None  # noqa T484

    def __init__(self, f_locals) -> None:
        self._global_scope = Scope.init_global(f_locals)
        self._stack = [self._global_scope]

    def globals(self):
        """
        Returns the current globals name table, as if the built-in globals()
        is called.
        """
        if self._global_scope is None:
            raise AssertionError(
                "attempted to call globals() on a stack without a global "
                "scope"
            )
        return self._global_scope.get_locals()

    def locals(self):
        """
        Returns the current locals name table, as if the built-in locals() is
        called.
        """
        if len(self._stack) == 0:
            raise AssertionError(
                "attempted to call locals() on an empty stack"
            )

        return self._stack[-1].get_locals()

    def resolve_name(self, name):
        """
        Returns the air variable for a given python name
        """
        # for local_vars in reversed(self._names_frames_stack):
        #    if name in local_vars:
        #        return local_vars[name]
        # raise NameError(f"name '{name}' is not defined")
        if name in self.locals():
            return self.locals()[name]

        if name in self.globals():
            return self.globals()[name]

        raise NameError(f"name '{name}' is not defined")

    def assign_name(self, name, value):
        """
        Updates the name stack with the given python name and air type.
        """
        if len(self._stack) == 0:
            raise AssertionError(
                "attempted to call assign_name() on an empty stack"
            )
        self._stack[-1].assign_name(name, value)

    def resolve_attr_chain(self, chain: ast.AST):
        """
        Resolve a chain of Attributes in AST format.

        Returns a list of attribute name resolutions starting with the name
        resolution of the rightmost attribute name.

        E.g. `chain=ast.parse("a.b.c").body[0].value` returns `[a, a.b, a.b.c]`,
        assuming everything in the list exists in the scope stack.

        Name node is assumed to be an attribute chain of size 1.
        """
        if isinstance(chain, ast.Attribute):  # chain length > 1
            attrname = chain.attr
            next_chain = chain.value
            current_result = self.resolve_attr_chain(next_chain)
            try:
                return [*current_result, getattr(current_result[-1], attrname)]
            except AttributeError:
                raise Exception(f"Attribute '{attrname}' not found on '{current_result[-1]}'")
        elif isinstance(chain, ast.Name):  # base case: chain length = 1
            cur_id = chain.id
            return [self.resolve_name(cur_id)]
        else:
            raise Exception(
                f"Unexpected AST node type {type(chain).__qualname__} in attribute chain"
            )

    def new_func_scope(self, ast_func: ast.FunctionDef, f_locals):
        """
        Establish a new function scope.

        The caller is responsible for generating the variables for the
        function's arguments and passing them into func_args.
        """

        self._stack.append(Scope(ast_func, f_locals))
