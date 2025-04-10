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
        # Flatten nested Add expressions and sort arguments for consistency
        new_args = []
        for arg in args:
            if isinstance(arg, Add):
                new_args.extend(arg.args)
            else:
                new_args.append(arg)
        self.args = tuple(sorted(new_args, key=str))

    def __str__(self):
        return "(" + " + ".join(map(str, self.args)) + ")"

    def __hash__(self):
        return hash(("Add", tuple(self.args)))


class Mul(Expr):
    """Represents multiplication (x * y)."""

    def __init__(self, *args):
        # Flatten nested Mul expressions and sort arguments for consistency
        new_args = []
        for arg in args:
            if isinstance(arg, Mul):
                new_args.extend(arg.args)
            else:
                new_args.append(arg)
        self.args = tuple(sorted(new_args, key=str))

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
def split_by_operator(expr_str: str, operator: str) -> list[str]:
    """Splits a string expression by an operator, respecting parentheses."""
    parts = []
    current_part = ""
    balance = 0
    for char in expr_str:
        if char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
        elif char == operator and balance == 0:
            parts.append(current_part)
            current_part = ""
            continue
        current_part += char
    current_part += ""  # Ensure the last part is added
    parts.append(current_part)
    return [part for part in parts if part]  # Remove empty parts


def parse_expression(expr_str: str) -> Expr:
    """Converts a string expression into Expr objects."""
    expr_str = expr_str.replace(" ", "")  # Remove spaces

    if expr_str.startswith("(") and expr_str.endswith(")"):
        balance = 0
        outermost = True
        for i, char in enumerate(expr_str):
            if char == "(":
                balance += 1
            elif char == ")":
                balance -= 1
            if balance == 0 and i < len(expr_str) - 1:
                outermost = False
                break
        if balance == 0 and outermost and len(expr_str) > 1:
            return parse_expression(expr_str[1:-1])

    # Handle numbers
    if re.fullmatch(r"-?\d+(\.\d+)?", expr_str):
        return Number(float(expr_str) if "." in expr_str else int(expr_str))

    # Handle variables (symbols)
    if re.fullmatch(r"[a-zA-Z]", expr_str):
        return Symbol(expr_str)

    # Handle exponentiation (x^y)
    if "^" in expr_str:
        base, exp = expr_str.split("^", 1)  # Split only once
        return Pow(parse_expression(base), parse_expression(exp))

    # Handle multiplication (x*y)
    if "*" in expr_str:
        terms = split_by_operator(expr_str, "*")
        return Mul(*[parse_expression(term) for term in terms])

    # Handle addition (x + y)
    if "+" in expr_str:
        terms = split_by_operator(expr_str, "+")
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
            coeff = Number(1)
            symbol = None
            if isinstance(term, Mul):
                num_factors = [f for f in term.args if isinstance(f, Number)]
                sym_factors = [f for f in term.args if not isinstance(f, Number)]
                if num_factors:
                    # Combine all numeric factors
                    coeff_val = 1
                    for num in num_factors:
                        coeff_val *= num.value
                    coeff = Number(coeff_val)
                    symbol = Mul(*sym_factors) if sym_factors else None
                elif sym_factors:
                    symbol = (
                        Mul(*sym_factors) if len(sym_factors) > 1 else sym_factors[0]
                    )
            elif isinstance(term, Number):
                coeff = term
            else:
                symbol = term

            if symbol in term_dict:
                term_dict[symbol] = Number(term_dict[symbol].value + coeff.value)
            else:
                term_dict[symbol] = coeff

        new_terms = []
        for symbol, coeff in term_dict.items():
            if coeff == Number(0):
                continue
            if coeff != Number(1) or symbol is None:
                new_terms.append(Mul(coeff, symbol) if symbol else coeff)
            else:
                new_terms.append(symbol)

        if not new_terms:
            return Number(0)
        elif len(new_terms) == 1:
            return new_terms[0]
        else:
            return Add(*new_terms)

    return expr


def expand(expr):
    """Distributes multiplication over addition: (x + y) * z → x*z + y*z."""
    if isinstance(expr, Mul):
        add_factors = [arg for arg in expr.args if isinstance(arg, Add)]
        if not add_factors:
            return expr

        other_factors = [arg for arg in expr.args if not isinstance(arg, Add)]
        first_add = add_factors[0]
        remaining_add = add_factors[1:]

        expanded_terms = []
        for term in first_add.args:
            new_mul_args = other_factors + [term] + remaining_add
            if len(new_mul_args) > 1:
                expanded_terms.append(Mul(*new_mul_args))
            else:
                expanded_terms.append(new_mul_args[0])

        return Add(*expanded_terms).replace(expand)  # Recursively expand

    return expr


def factor(expr):
    """Factor out common terms."""
    if isinstance(expr, Add):
        if not expr.args:
            return Number(0)

        first_term_factors = (
            set(expr.args[0].args) if isinstance(expr.args[0], Mul) else {expr.args[0]}
        )
        common_factors = first_term_factors.copy()

        for term in expr.args[1:]:
            term_factors = set(term.args) if isinstance(term, Mul) else {term}
            common_factors.intersection_update(term_factors)

        if common_factors:
            # Find the simplest common factor
            common_factor = min(common_factors, key=str)

            factored_terms = []
            for term in expr.args:
                if isinstance(term, Mul):
                    new_args = [arg for arg in term.args if arg != common_factor]
                    factored_terms.append(Mul(*new_args) if new_args else Number(1))
                elif term == common_factor:
                    factored_terms.append(Number(1))
                else:
                    # If a term doesn't have the common factor, we can't factor out this common factor from the entire sum
                    return expr
            return Mul(common_factor, Add(*factored_terms))

    return expr


def powsimp(expr):
    """Simplify power expressions: (x^a) * (x^b) → x^(a+b)."""
    if isinstance(expr, Mul):
        base_exponents = {}
        other_factors = []
        for factor in expr.args:
            if isinstance(factor, Pow):
                base, exp = factor.args
                base_exponents[base] = Add(base_exponents.get(base, Number(0)), exp)
            else:
                other_factors.append(factor)

        pow_terms = [
            Pow(base, exp) if exp != Number(1) else base
            for base, exp in base_exponents.items()
            if exp != Number(0)
        ]

        return (
            Mul(*(other_factors + pow_terms))
            if other_factors or pow_terms
            else Number(1)
        )

    return expr


# -----------------------------------------------
# Scoring Heuristic for Best Simplification
# -----------------------------------------------
def count_ops(expr):
    if isinstance(expr, (Number, Symbol)):
        return 1
    return 1 + sum(count_ops(arg) for arg in expr.args)


def simplify(expr, measure=count_ops, ratio=1.7):
    """Main simplification function"""
    original = expr
    best = expr
    min_complexity = measure(expr)

    print(f"🔍 Original Expression: {expr} (Complexity: {min_complexity})")  # Debug

    # Try different simplification strategies
    strategies = [
        collect_like_terms,  # Directly apply collect_like_terms
        lambda x: x.replace(expand),
        lambda x: x.replace(factor),
        lambda x: x.replace(powsimp),
    ]

    for strategy in strategies:
        if callable(strategy):
            simplified = strategy(expr)
            strategy_name = (
                strategy.__name__ if hasattr(strategy, "__name__") else str(strategy)
            )
        else:
            simplified = expr.replace(strategy)
            strategy_name = strategy.__name__

        current_complexity = measure(simplified)

        print(
            f"➡ Trying {strategy_name}: {simplified} (Complexity: {current_complexity})"
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

    expr_str_2 = "(x + y) * z"
    parsed_expr_2 = parse_expression(expr_str_2)
    simplified_expr_2 = simplify(parsed_expr_2)
    print(f"Original: {parsed_expr_2}")
    print(f"Simplified: {simplified_expr_2}")

    expr_str_3 = "x^2 * x^3"
    parsed_expr_3 = parse_expression(expr_str_3)
    simplified_expr_3 = simplify(parsed_expr_3)
    print(f"Original: {parsed_expr_3}")
    print(f"Simplified: {simplified_expr_3}")

    expr_str_4 = "4*x + 2*y + x - 3*y"
    parsed_expr_4 = parse_expression(expr_str_4)
    simplified_expr_4 = simplify(parsed_expr_4)
    print(f"Original: {parsed_expr_4}")
    print(f"Simplified: {simplified_expr_4}")
