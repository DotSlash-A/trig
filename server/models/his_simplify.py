from typing import Union
import re


class Expr:
    """Base class for symbolic expressions."""

    def __eq__(self, other):
        return type(self) == type(other) and self.args == other.args

    def __str__(self):
        raise NotImplementedError("String representation not implemented.")

    def replace(self, func):
        """Applies a transformation function to sub-expressions, if any."""
        if not hasattr(self, "args"):  # If no `args`, just return itself
            return self
        new_args = [
            arg.replace(func) if isinstance(arg, Expr) else arg for arg in self.args
        ]
        return func(self.__class__(*new_args))


class Number(Expr):
    """Represents numeric constants."""

    def __init__(self, value: Union[int, float]):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        return isinstance(other, Number) and self.value == other.value

    def __hash__(self):
        return hash(self.value)


class Symbol(Expr):
    """Represents a variable like x, y."""

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Symbol) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Add(Expr):
    """Represents addition (x + y)."""

    def __init__(self, *args):
        self.args = tuple(args)

    def __str__(self):
        return "(" + " + ".join(map(str, self.args)) + ")"

    def __hash__(self):
        return hash(("Add", tuple(self.args)))


class Mul(Expr):
    """Represents multiplication (x * y)."""

    def __init__(self, *args):
        self.args = tuple(args)

    def __str__(self):
        return "(" + " * ".join(map(str, self.args)) + ")"

    def __hash__(self):
        return hash(("Mul", tuple(self.args)))


class Pow(Expr):
    """Represents exponentiation (x^y)."""

    def __init__(self, base, exp):
        self.base = base
        self.exp = exp
        self.args = (base, exp)

    def __str__(self):
        return f"({self.base}^{self.exp})"

    def __hash__(self):
        return hash(("Pow", tuple(self.args)))


# -----------------------------------------------
# Parsing String Expression into Expr Objects
# -----------------------------------------------
def parse_expression(expr_str: str) -> Expr:
    """Converts a string expression into Expr objects."""
    expr_str = expr_str.replace(" ", "")  # Remove spaces

    # Handle numbers
    if re.fullmatch(r"-?\d+(\.\d+)?", expr_str):
        return Number(float(expr_str) if "." in expr_str else int(expr_str))

    # Handle variables (symbols)
    if re.fullmatch(r"[a-zA-Z]", expr_str):
        return Symbol(expr_str)

    # Handle exponentiation (x^y)
    if "^" in expr_str:
        base, exp = expr_str.split("^")
        return Pow(parse_expression(base), parse_expression(exp))

    # Handle multiplication (x*y)
    if "*" in expr_str:
        terms = expr_str.split("*")
        return Mul(*[parse_expression(term) for term in terms])

    # Handle addition (x + y)
    if "+" in expr_str:
        terms = expr_str.split("+")
        return Add(*[parse_expression(term) for term in terms])

    raise ValueError(f"Invalid expression: {expr_str}")


# -----------------------------------------------
# Simplification Rules
# -----------------------------------------------
def collect_like_terms(expr):
    """Combine like terms in an Add expression."""
    if isinstance(expr, Add):
        term_dict = {}
        for term in expr.args:
            if isinstance(term, Mul):
                coeff, symbol = (
                    term.args if isinstance(term.args[0], Number) else (Number(1), term)
                )
            elif isinstance(term, Number):
                coeff, symbol = term, None
            else:
                coeff, symbol = Number(1), term

            if symbol in term_dict:
                term_dict[symbol] = Number(term_dict[symbol].value + coeff.value)
            else:
                term_dict[symbol] = coeff

        new_terms = [Mul(v, k) if k else v for k, v in term_dict.items()]
        return Add(*new_terms) if len(new_terms) > 1 else new_terms[0]

    return expr


def expand(expr):
    """Distributes multiplication over addition: (x + y) * z → x*z + y*z."""
    if isinstance(expr, Mul):
        expanded_terms = []
        for term in expr.args:
            if isinstance(term, Add):
                for subterm in term.args:
                    expanded_terms.append(
                        Mul(subterm, *[t for t in expr.args if t != term])
                    )
                return Add(*expanded_terms)

    return expr


def factor(expr):
    """Factor out common terms."""
    if isinstance(expr, Add):
        common_factors = (
            set(expr.args[0].args) if isinstance(expr.args[0], Mul) else {expr.args[0]}
        )
        for term in expr.args[1:]:
            if isinstance(term, Mul):
                common_factors.intersection_update(term.args)
            else:
                common_factors.intersection_update({term})

        if common_factors:
            common_factor = min(common_factors, key=str)  # Pick the simplest factor
            factored_terms = [
                (
                    Mul(*(set(term.args) - {common_factor}))
                    if isinstance(term, Mul)
                    else Number(1)
                )
                for term in expr.args
            ]
            return Mul(common_factor, Add(*factored_terms))

    return expr


def powsimp(expr):
    """Simplify power expressions: (x^a) * (x^b) → x^(a+b)."""
    if isinstance(expr, Mul):
        base_exponents = {}
        for factor in expr.args:
            if isinstance(factor, Pow):
                base, exp = factor.args
                base_exponents[base] = base_exponents.get(base, Number(0)) + exp
            else:
                base_exponents[factor] = base_exponents.get(factor, Number(1))

        return Mul(
            *[Pow(b, e) if e != Number(1) else b for b, e in base_exponents.items()]
        )

    return expr


# -----------------------------------------------
# Scoring Heuristic for Best Simplification
# -----------------------------------------------
def expression_score(expr):
    """Assigns a heuristic complexity score (lower is better)."""
    if isinstance(expr, Number) or isinstance(expr, Symbol):
        return 1
    return 1 + sum(expression_score(arg) for arg in expr.args)


# -----------------------------------------------
# Main Simplification Function
# -----------------------------------------------
# def simplify(expr):
#     """Applies multiple strategies and picks the best one."""
#     strategies = [collect_like_terms, expand, factor, powsimp]

#     best_expr = expr
#     min_score = expression_score(expr)

#     for strategy in strategies:
#         new_expr = strategy(expr)
#         new_score = expression_score(new_expr)

#         if new_score < min_score:
#             best_expr = new_expr
#             min_score = new_score


#     return best_expr
def count_ops(expr):
    if isinstance(expr, (Number, Symbol)):
        return 0
    return 1 + sum(count_ops(arg) for arg in expr.args)


def simplify(expr, measure=count_ops, ratio=1.7):
    """Main simplification function"""
    original = expr
    best = expr
    min_complexity = measure(expr)

    print(f"🔍 Original Expression: {expr} (Complexity: {min_complexity})")  # Debug

    # Try different simplification strategies
    strategies = [
        lambda x: x.replace(expand),
        lambda x: x.replace(factor),
        lambda x: x.replace(powsimp),
    ]

    for strategy in strategies:
        simplified = strategy(expr)
        current_complexity = measure(simplified)

        print(
            f"➡ Trying {strategy.__name__}: {simplified} (Complexity: {current_complexity})"
        )  # Debug

        if current_complexity < min_complexity:
            best = simplified
            min_complexity = current_complexity

    # Check complexity ratio constraint
    if measure(best) > ratio * measure(original):
        print("❌ Simplification rejected (complexity ratio exceeded)")  # Debug
        return original

    print(f"✅ Final Simplified Expression: {best}")
    return best


# -----------------------------------------------
# Testing the Simplifier
# -----------------------------------------------
if __name__ == "__main__":
    expr_str = "2*x + 3*x"
    parsed_expr = parse_expression(expr_str)
    simplified_expr = simplify(parsed_expr)

    print(f"Original: {parsed_expr}")
    print(f"Simplified: {simplified_expr}")
