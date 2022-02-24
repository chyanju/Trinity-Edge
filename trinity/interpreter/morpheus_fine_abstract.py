'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Note: This interpreter is currently out of sync. Check before you use.
'''

import pandas as pd
import numpy as np

from collections import namedtuple
from typing import Tuple, List, Iterator, Any

from .. import spec as S
from ..dsl import Node, AtomNode, ParamNode, ApplyNode
from ..visitor import GenericVisitor
from ..spec.interval import *
from ..utils.table import make_abs, assemble_abstract_table, abs_intersected

from .interpreter import Interpreter
from .post_order import PostOrderInterpreter, FineAbstractInterpreter
from .error import InterpreterError, GeneralError

class NodeVisitor(GenericVisitor):
    _interp: FineAbstractInterpreter
    _inputs: Any

    def __init__(self, interp, inputs):
        self._interp = interp
        self._inputs = inputs

    # note: for atom node, to support parameter level conflict-driven learning, 
    #       use ??? to get the value, not the original eval_??? methods in Trinity
    #       and eventually return the node itself
    # note: in this version, every atom node is required to have tag and "cpos" field
    def visit_atom_node(self, atom_node: AtomNode):
        tmp_prod_id = atom_node.production.id
        # note: use self._interp to refer to the self in eval
        self._interp._current_combination += (tmp_prod_id,)
        # note: sanity checking temporarily for new deduction, remove at some point
        if atom_node.tag["cpos"] != self._interp._last_cpos+1:
            raise Exception("Panic: last cpos is: {}, current cpos is: {}".format(
                self._interp._last_cpos, atom_node.tag["cpos"]))
        self._interp._last_cpos = atom_node.tag["cpos"]
        return atom_node

    def visit_param_node(self, param_node: ParamNode):
        param_index = param_node.index
        if param_index >= len(self._inputs):
            msg = 'Input parameter access({}) out of bound({})'.format(
                param_index, len(self._inputs))
            raise GeneralError(msg)
        return self._inputs[param_index]

    def visit_apply_node(self, apply_node: ApplyNode):
        in_values = [self.visit(x) for x in apply_node.args]
        method_name = self._eval_method_name(apply_node.name)
        method = getattr(self._interp, method_name,
                         self._method_not_found)
        return method(apply_node, in_values)

    def _method_not_found(self, apply_node: ApplyNode, arg_values: List[Any]):
        msg = 'Cannot find required eval method: "{}"'.format(
            self._eval_method_name(apply_node.name))
        raise NotImplementedError(msg)

    @staticmethod
    def _eval_method_name(name):
        return 'eval_' + name

# fixme: currently abstract interpretation rules are hard coded
#        need to dynamically load it from the spec in the next version
class MorpheusFineAbstractInterpreter(FineAbstractInterpreter):
    '''
    An extended abstract interpreter that works on non-full skeleton level (i.e., considers some arguments).
    '''

    def __init__(self, spec: S.TrinitySpec, config=None, *args, **kwargs):
        super(MorpheusFineAbstractInterpreter, self).__init__(*args, **kwargs)

        self._spec = spec

        # ======== shared infrastructure in line with LineSkeletonEnumerator ======== #
        # note: a stateful variable that keeps track of interpretation combination
        #       which is connected to LineSkeletonEnumerator
        #       typical combination can be (1, 4, 1, 2, None, None)
        #       where None indicates anything and integers indicate production id
        #       LineSkeletonEnumerator will capture and utilize it to speed up enumeration
        self._current_iter_ptr = None
        self._current_nslot = None
        self._current_context = None
        self._current_combination = None
        # helper class for partial evaluation
        self.visitor_class = NodeVisitor
        # ======== shared infrastructure in line with LineSkeletonEnumerator ======== #

        # note: a sanity testing variable temporarily for the new deduction design
        #       need to make sure that the cpos values visited are increasing
        #       i.e., the order the atom node is composed is the same as the atom node is visited
        # note: sanity checking temporarily for new deduction, remove at some point
        self._last_cpos = -1

    # fixme: forward to utils.table.make_abs
    def make_abs(self):
        return make_abs()

    # fixme: forward to utils.table.assemble_abstract_table
    def assemble_abstract_table(self, arg_tb0, arg_tb1):
        return assemble_abstract_table(arg_tb0, arg_tb1)

    # fixme: forward to utils.table.abs_intersected
    def abs_intersected(self, abs0, abs1):
        return abs_intersected(abs0, abs1)

    # hijack eval function: transform the inputs to abstract values before feeding
    def eval(self, prog: Node, inputs: List[Any], is_concrete=True) -> Any:
        if is_concrete:
            abstract_inputs = [
                self.assemble_abstract_table(inputs[0], p)
                for p in inputs
            ]
        else:
            abstract_inputs = inputs

        node_visitor = NodeVisitor(self, abstract_inputs)
        # note: sanity checking temporarily for new deduction, remove at some point
        self._last_cpos = -1
        try:
            # try if this node is a root node ("skeleton" field only exists in root node)
            if prog.tag is not None:
                if "skeleton" in prog.tag:
                    # yes it's root
                    # then initialize set the _current_combination
                    self._current_iter_ptr = prog.tag["iter_ptr"]
                    self._current_nslot = prog.tag["nslot"]
                    self._current_context = ()
                    self._current_combination = ()
            return node_visitor.visit(prog)
        except InterpreterError as e:
            raise

    def _get_context(self, comb, nodes):
        # this extracts the interpreter context (not the visitor context) from the interpreter combination
        return tuple([
            None if i in [p.tag["cpos"] for p in nodes] else comb[i]
            for i in range(len(comb))
        ])

    # ================================== #
    # ======== enum productions ======== #
    # ================================== #

    # fixme: merge with NodeVisitor later
    def _eval_method_name(self, name):
        return 'eval_' + name

    # main entrance of evaluating an atom node
    def _eval_atom_node(self, node):
        # print("# [DEBUG] _eval_atom_node got: {}".format(node))
        node_type = node.type.name
        method_name = self._eval_method_name(node_type)
        method = getattr(self, method_name)
        return method(node.data)

    # note: use this method in EnumAssertion
    def _eval_enum_prod(self, prod):
        prod_type = prod.lhs.name
        method_name = self._eval_method_name(prod_type)
        method = getattr(self, method_name)
        return method(prod.rhs[0])

    # can only be called by _eval_atom_node
    def eval_ColInt(self, v):
        return int(v)

    # can only be called by _eval_atom_node
    # def eval_SmallInt(self, v):
    #     return int(v)
    def eval_ConstVal(self, v):
        if v.endswith("@Float"):
            return float(v[:-6])
        elif v.endswith("@Int"):
            return int(v[:-4])
        elif v.endswith("@Str"):
            return v[:-4]
        else:
            raise InterpreterError("Exception evaluating ConstVal.")

    # can only be called by _eval_atom_node
    def eval_ColList(self, v):
        # question: is this true?
        return [int(p) for p in v]

    # can only be called by _eval_atom_node
    def eval_AggrFunc(self, v):
        return v

    # can only be called by _eval_atom_node
    def eval_NumFunc(self, v):
        return v

    # can only be called by _eval_atom_node
    def eval_BoolFunc(self, v):
        return v

    # ====================================== #
    # ======== function productions ======== #
    # ====================================== #

    # this validates that the collist is not stupid in a fast way that won't check column overflow
    # note: this should be called with in assertEnum, and before you call the explain function
    def fast_validate_collist(self, arg_collist):
        # arg_collist is the original collist (before explanation)

        if len(arg_collist) != len(list(set(arg_collist))):
            # don't include duplicates
            # e.g., -1, -1, will cause ValueError in .remove(x) in explain_collist
            return False

        # note-important: don't mix positive and negative ints
        if max(arg_collist) >= 0 and min(arg_collist) < 0:
            return False

        for p in arg_collist:
            if p == 0:
                if -99 in arg_collist:
                    return False
                else:
                    continue
            elif p == -99:
                if 0 in arg_collist:
                    return False
                else:
                    continue
            elif p > 0:
                if -p in arg_collist:
                    return False
                else:
                    continue
            elif p < 0:
                if -p in arg_collist:
                    return False
                else:
                    continue

        return True

    def eval_select(self, node, args):
        arg_tb, node_collist = args
        arg_collist = self._eval_atom_node(node_collist)

        self.assertEnum(
            node=node, 
            context=self._current_context, 
            comb=self._current_combination,
            # this makes sure the original colist is not stupid
            cond=lambda comb: ( 
                lambda x: self.fast_validate_collist(x) 
            )(
                self._eval_enum_prod( self._spec.get_production( comb[node_collist.tag["cpos"]] ) )
            ),
            tag="ks:abs:select:0",
        )

        out = self.make_abs()
        out["row"] = interval_binary_op("==", out["row"], arg_tb["row"])
        out["head"] = interval_binary_op("<=", out["head"], arg_tb["head"])
        out["content"] = interval_binary_op("<=", out["content"], arg_tb["content"])
        # precise tracking
        # out["col"] = interval_binary_op("<", out["col"], arg_tb["col"])
        if arg_collist[0]<0:
            out["col"] = interval_binary_op(
                "==", 
                out["col"],
                interval_binary_op("-", arg_tb["col"], Interval(len(arg_collist),len(arg_collist)))
            )
        else:
            out["col"] = interval_binary_op("==", out["col"], Interval(len(arg_collist),len(arg_collist)))

        # good to set context if you are here
        self._current_context += (node_collist.production.id,)

        return out

    def eval_unite(self, node, args):
        arg_tb, node_col0, node_col1 = args
        arg_col0 = self._eval_atom_node(node_col0)
        arg_col1 = self._eval_atom_node(node_col1)

        self.assertEnum(
            node=node, 
            context=self._current_context, 
            comb=self._current_combination,
            # note: nested lambda to store temp variable
            cond=lambda comb: (
                lambda x0,x1: x0 != x1
            )(
                self._eval_enum_prod( self._spec.get_production( comb[node_col0.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_col1.tag["cpos"]] ) )
            ),
            tag="ks:abs:unite:0",
        )

        out = self.make_abs()
        out["row"] = interval_binary_op("==", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op(
            "==",
            out["col"],
            interval_binary_op("-", arg_tb["col"], Interval(1,1)),
        )
        out["head"] = interval_binary_op(
            "<=",
            out["head"],
            interval_binary_op("+", arg_tb["head"], Interval(1,1)),
        )
        out["content"] = interval_binary_op(
            ">=",
            out["content"],
            interval_binary_op("+", arg_tb["content"], Interval(1,1)),
        )

        # good to set context if you are here
        self._current_context += (node_col0.production.id, node_col1.production.id)

        return out

    def eval_separate(self, node, args):
        arg_tb, node_col = args
        arg_col = self._eval_atom_node(node_col)

        out = self.make_abs()
        out["row"] = interval_binary_op("==", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op(
            "==",
            out["col"],
            interval_binary_op("+", arg_tb["col"], Interval(1,1)),
        )
        out["head"] = interval_binary_op(
            "<=",
            out["head"],
            interval_binary_op("+", arg_tb["head"], Interval(2,2)),
        )
        out["content"] = interval_binary_op(
            ">=",
            out["content"],
            interval_binary_op("+", arg_tb["content"], Interval(2,2)),
        )

        # good to set context if you are here
        self._current_context += (node_col.production.id,)

        return out

    def eval_gather(self, node, args):
        arg_tb, node_collist = args
        arg_collist = self._eval_atom_node(node_collist)

        self.assertEnum(
            node=node, 
            context=self._current_context, 
            comb=self._current_combination,
            # x0: this makes sure the original colist is not stupid
            # self.explain_collist(x0): normal gather check
            # note-important: to ensure x0 comes before self.explain_collist(x0) checks, merge them into one assertEnum
            cond=lambda comb: ( 
                lambda x0: self.fast_validate_collist(x0) 
            )(
                self._eval_enum_prod( self._spec.get_production( comb[node_collist.tag["cpos"]] ) ),
            ),
            tag="ks:abs:gather:0",
        )

        out = self.make_abs()
        out["row"] = interval_binary_op(">=", out["row"], arg_tb["row"])
        out["head"] = interval_binary_op(
            "<=",
            out["head"],
            interval_binary_op("+", arg_tb["head"], Interval(2,2)),
        )
        out["content"] = interval_binary_op(
            "<=",
            out["content"],
            interval_binary_op("+", arg_tb["content"], Interval(2,2)),
        )
        # precise tracking
        # out["col"] = interval_binary_op("<=", out["col"], arg_tb["col"])
        if arg_collist[0]<0:
            out["col"] = interval_binary_op(
                "==",
                out["col"],
                interval_binary_op(
                    "+",
                    Interval(len(arg_collist),len(arg_collist)),
                    Interval(2,2)
                )
            )
        else:
            out["col"] = interval_binary_op(
                "==",
                out["col"],
                interval_binary_op(
                    "+",
                    interval_binary_op("-", arg_tb["col"], Interval(len(arg_collist),len(arg_collist))),
                    Interval(2,2)
                )
            )

        # good to set context if you are here
        self._current_context += (node_collist.production.id,)

        return out

    def eval_spread(self, node, args):
        arg_tb, node_col0, node_col1 = args
        arg_col0 = self._eval_atom_node(node_col0)
        arg_col1 = self._eval_atom_node(node_col1)

        self.assertEnum(
            node=node, 
            context=self._current_context, 
            comb=self._current_combination,
            # note: nested lambda to store temp variable
            cond=lambda comb: (
                lambda x0, x1: x0 != x1
            )(
                self._eval_enum_prod( self._spec.get_production( comb[node_col0.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_col1.tag["cpos"]] ) )
            ),
            tag="ks:abs:spread:0",
        )

        out = self.make_abs()
        out["row"] = interval_binary_op("<=", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op(">=", out["col"], arg_tb["col"])
        out["head"] = interval_binary_op("<=", out["head"], arg_tb["content"])
        out["content"] = interval_binary_op("<=", out["content"], arg_tb["content"])

        # good to set context if you are here
        self._current_context += (node_col0.production.id, node_col1.production.id)

        return out

    def eval_mutate(self, node, args):
        arg_tb, node_op, node_col0, node_col1 = args
        arg_op = self._eval_atom_node(node_op)
        arg_col0 = self._eval_atom_node(node_col0)
        arg_col1 = self._eval_atom_node(node_col1)

        self.assertEnum(
            node=node,
            context=self._current_context, 
            comb=self._current_combination,
            # note: nested lambda to store temp variable
            cond=lambda comb: (
                lambda x0, x1: x0 != x1
            )(
                self._eval_enum_prod( self._spec.get_production( comb[node_col0.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_col1.tag["cpos"]] ) )
            ),
            tag="ks:abs:mutate:0",
        )

        out = self.make_abs()
        out["row"] = interval_binary_op("==", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op(
            "==",
            out["col"],
            interval_binary_op("+", arg_tb["col"], Interval(1,1)),
        )
        out["head"] = interval_binary_op(
            "==",
            out["head"],
            interval_binary_op("+", arg_tb["head"], Interval(1,1)),
        )
        out["content"] = interval_binary_op(">", out["content"], arg_tb["content"])
        out["content"] = interval_binary_op(
            "<=",
            out["content"],
            interval_binary_op("+", arg_tb["content"], arg_tb["row"]),
        )

        # good to set context if you are here
        self._current_context += (node_op.production.id, node_col0.production.id, node_col1.production.id)

        return out

    def eval_filter(self, node, args):
        arg_tb, node_op, node_col, node_int = args
        arg_op = self._eval_atom_node(node_op)
        arg_col = self._eval_atom_node(node_col)
        arg_int = self._eval_atom_node(node_int)

        out = self.make_abs()
        out["row"] = interval_binary_op("<", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op("==", out["col"], arg_tb["col"])
        out["head"] = interval_binary_op("==", out["head"], arg_tb["head"])
        out["content"] = interval_binary_op("<=", out["content"], arg_tb["content"])

        # good to set context if you are here
        self._current_context += (node_op.production.id, node_col.production.id, node_int.production.id)

        return out

    def eval_group(self, node, args):
        arg_tb, node_collist, node_op, node_col = args
        arg_collist = self._eval_atom_node(node_collist)
        arg_op = self._eval_atom_node(node_op)
        arg_col = self._eval_atom_node(node_col)

        self.assertEnum(
            node=node, 
            context=self._current_context, 
            comb=self._current_combination,
            # this makes sure the original colist is not stupid
            # note-important: to ensure x0 comes before self.explain_collist(x0)/y checks, merge them into one assertEnum
            cond=lambda comb: ( 
                lambda x0,y: self.fast_validate_collist(x0)
            )(
                self._eval_enum_prod( self._spec.get_production( comb[node_collist.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_col.tag["cpos"]] ) )
            ),
            tag="ks:abs:group:0",
        )

        out = self.make_abs()
        out["row"] = interval_binary_op("<=", out["row"], arg_tb["row"])
        out["head"] = interval_binary_op(">", out["head"], Interval(0,0))
        out["head"] = interval_binary_op(
            "<=",
            out["head"],
            interval_binary_op("+", arg_tb["head"], Interval(1,1)),
        )
        # note: originally it's: out.content <= in.content + in.group + 1
        #       since we don't track group, it becomes: out.content <= in.content + in.row + 1
        out["content"] = interval_binary_op(
            "<=",
            out["content"],
            interval_binary_op(
                "+",
                arg_tb["content"],
                interval_binary_op("+", arg_tb["row"], Interval(1,1)),
            ),
        )
        # precise tracking
        # out["col"] = interval_binary_op(
        #     "<=",
        #     out["col"],
        #     interval_binary_op("+", arg_tb["col"], Interval(1,1)),
        # )
        if arg_collist[0]<0:
            out["col"] = interval_binary_op(
                "==",
                out["col"],
                interval_binary_op(
                    "+",
                    interval_binary_op("-", arg_tb["col"], Interval(len(arg_collist),len(arg_collist))),
                    Interval(1,1)
                )
            )
        else:
            out["col"] = interval_binary_op(
                "==",
                out["col"],
                interval_binary_op("+", Interval(len(arg_collist),len(arg_collist)), Interval(1,1))
            )

        # good to set context if you are here
        self._current_context += (node_collist.production.id, node_op.production.id, node_col.production.id)

        return out
       










