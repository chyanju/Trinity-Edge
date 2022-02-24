# First, specify the types that will be used
enum ColInt {
    "0", "1", "2", "3", "4", "5"
}

enum ConstVal {
    "Nn@Str"
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