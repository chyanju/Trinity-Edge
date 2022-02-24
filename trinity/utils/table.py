import pandas as pd

from ..spec.interval import *

def get_row(arg_tb):
    if isinstance(arg_tb, pd.DataFrame):
        # precise table
        v = arg_tb.shape[0]
        return Interval(v, v)
    else:
        raise NotImplementedError("Unsupported type of table for get_row, got: {}.".format(type(arg_tb)))

def get_col(arg_tb):
    if isinstance(arg_tb, pd.DataFrame):
        # precise table
        v = arg_tb.shape[1]
        return Interval(v, v)
    else:
        raise NotImplementedError("Unsupported type of table for get_col, got: {}.".format(type(arg_tb)))

def get_head(arg_base, arg_tb):
    if isinstance(arg_base, pd.DataFrame):
        if isinstance(arg_tb, pd.DataFrame):
            # precise table
            head0 = set(arg_base.columns)
            content0 = set(arg_base.values.flatten().tolist())
            head1 = set(arg_tb.columns)
            v = len(head1 - head0 - content0)
            return Interval(v, v)
        else:
            raise NotImplementedError("Unsupported type of table for get_head, got: {}.".format(type(arg_tb)))
    else:
        raise NotImplementedError("Unsupported type of base table for get_head, got: {}.".format(type(arg_base)))

def get_content(arg_base, arg_tb):
    if isinstance(arg_base, pd.DataFrame):
        if isinstance(arg_tb, pd.DataFrame):
            # precise table
            content0 = set(arg_base.values.flatten().tolist())
            content1 = set(arg_tb.values.flatten().tolist())
            v = len(content1 - content0)
            return Interval(v, v)
        else:
            raise NotImplementedError("Unsupported type of table for get_head, got: {}.".format(type(arg_tb)))
    else:
        raise NotImplementedError("Unsupported type of base table for get_head, got: {}.".format(type(arg_base)))

def make_abs():
    return {
        "row": Interval(IMIN,IMAX),
        "col": Interval(IMIN,IMAX),
        "head": Interval(IMIN,IMAX),
        "content": Interval(IMIN,IMAX),
    }

def assemble_abstract_table(arg_tb0, arg_tb1):
    # arg_tb0 is the base, i.e., one of the exampe input(s)
    return {
        "row": get_row(arg_tb1),
        "col": get_col(arg_tb1),
        "head": get_head(arg_tb0, arg_tb1),
        "content": get_content(arg_tb0, arg_tb1),
    }

def abs_intersected(abs0, abs1):
    # within this framework, abs0 and abs1 have the same key set
    for p in abs0.keys():
        if not interval_is_intersected(abs0[p], abs1[p]):
            return False
    return True