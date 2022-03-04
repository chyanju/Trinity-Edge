spec = """
# First, specify the types that will be used
enum ColInt {
    "0", "1", "2", "3", "4", "5"
}

enum ConstVal {
    "@Str"
}

enumset ColList[2] {
    "0", "1", "2", "3", "4", "5", "-99", "-1", "-2", "-3", "-4", "-5"
}

enum AggrFunc {
    "min", "max", "sum", "mean", "count"
}

enum NumFunc {
    "/"
}
enum BoolFunc {
    "==", ">", "<"
}

value Table {
    col: int;
    row: int;
    head: int;
    content: int;
}

value Empty;

# Next, specify the input/output of the synthesized program
program Morpheus(Table) -> Table;

# Finally, specify the production rules
func empty: Empty -> Empty;

func select: Table r -> Table a, ColList b {
    row(r) == row(a);
    col(r) < col(a);
}

func unite: Table r -> Table a, ColInt b, ColInt c {
    row(r) == row(a);
    col(r) == col(a) - 1;
    head(r) <= head(a) + 1;
    content(r) >= content(a) + 1;
}

func separate: Table r -> Table a, ColInt b {
    row(r) == row(a);
    col(r) == col(a) + 1;
}

func gather: Table r -> Table a, ColList b {
    row(r) >= row(a);
    col(r) <= col(a);
    head(r) <= head(a) + 2;
    content(r) <= content(a) + 2;
}

func spread: Table r -> Table a, ColInt b, ColInt c {
    row(r) <= row(a);
    col(r) >= col(a);
    head(r) <= content(a);
    content(r) <= content(a);
}

func mutate: Table r -> Table a, NumFunc b, ColInt c, ColInt d {
    row(r) == row(a);
    col(r) == col(a) + 1;
}

func filter: Table r -> Table a, BoolFunc b, ColInt c, ConstVal d {
    row(r) < row(a);
    col(r) == col(a);
}

func group: Table r -> Table a, ColList b, AggrFunc c, ColInt d {
    row(r) < row(a);
    col(r) <= col(a) + 1;
}

func group_all: Table r -> Table a, AggrFunc c, ColInt d {
    row(r) < row(a);
    col(r) <= col(a) + 1;
}
"""
prog_json = [['function', 'spread'], [['function', 'gather'], ['param', 0], ['enum', 'ColList', ['0', '1']]], ['enum', 'ColInt', '2'], ['enum', 'ColInt', '3']]

import pandas as pd
import io
input_df = pd.read_csv(io.StringIO(""",Shop_ID,Employee_ID,Start_from,Is_full_time
0,1,1,2009,T
1,1,2,2003,T
2,8,3,2011,F
3,4,4,2012,T
4,5,5,2013,T
5,2,6,2010,F
6,6,7,2008,T"""))

import trinity.spec
import trinity.dsl

def trinity_evaluate(spec, expr_json, inputs):
    spec_obj = trinity.spec.parse(spec)
    builder = trinity.dsl.Builder(spec_obj)
    compiled_trinity = builder.from_jsonexp(expr_json)

    def tag_trick(arg_node, ind=trinity.dsl.node.Mutable(0)):
        if isinstance(arg_node, trinity.dsl.node.AtomNode):
            arg_node._tag = {"cpos":ind.v}
            ind.v += 1
        elif isinstance(arg_node, trinity.dsl.node.ParamNode):
            pass
        elif isinstance(arg_node, trinity.dsl.node.ApplyNode):
            for p in arg_node.args:
                tag_trick(p)
    
    tag_trick(compiled_trinity)

    interpreter = trinity.interpreter.morpheus.MorpheusInterpreter( spec=spec_obj )
    interpreter._current_combination = tuple([None for _ in range(30)])
    interpreter._current_context = tuple()
    interpreter._assert_enum = False
    interpreter._suppress_pandas_exception = False
    return interpreter.eval(compiled_trinity, inputs), interpreter.equal_tb

trinity_evaluate(spec, prog_json, [input_df])