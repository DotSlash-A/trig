import re
from functools import reduce


class Expression:
    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __pow__(self, other):
        return Pow(self, other)

    def simplify(self):
        return self


class Number(Expression):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)

    def simplify(self):
        return self


class Variable(Expression):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def simplify(self):
        return self


class Add(Expression):
    def __init__(self, *terms):
        self.terms = terms

    def __repr__(self):
        return " + ".join(map(str, self.terms))

    def simplify(self):
        simplified_terms = [term.simplify() for term in self.terms]
        numbers = sum(
            term.value for term in simplified_terms if isinstance(term, Number)
        )
        variables = [term for term in simplified_terms if not isinstance(term, Number)]

        if numbers == 0:
            return Add(*variables) if variables else Number(0)
        return Add(Number(numbers), *variables)


class Mul(Expression):
    def __init__(self, *factors):
        self.factors = factors

    def __repr__(self):
        return " * ".join(map(str, self.factors))

    def simplify(self):
        simplified_factors = [factor.simplify() for factor in self.factors]
        product = 1
        variables = []

        for factor in simplified_factors:
            if isinstance(factor, Number):
                product *= factor.value
            else:
                variables.append(factor)

        if product == 0:
            return Number(0)
        if product == 1 and variables:
            return Mul(*variables)
        return Mul(Number(product), *variables)


class Pow(Expression):
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent

    def __repr__(self):
        return f"({self.base})^({self.exponent})"

    def simplify(self):
        base = self.base.simplify()
        exponent = self.exponent.simplify()
        if isinstance(base, Number) and isinstance(exponent, Number):
            return Number(base.value**exponent.value)
        return Pow(base, exponent)


def split_by_operator(expr_str, operator):
    return [part.strip() for part in expr_str.split(operator)]


def parse_expression(expr_str):
    expr_str = expr_str.replace(" ", "")  # Remove spaces

    if "+" in expr_str:
        terms = split_by_operator(expr_str, "+")
        return Add(*[parse_expression(term) for term in terms])

    if "-" in expr_str and not expr_str.startswith("-"):
        terms = split_by_operator(expr_str, "-")
        first_term = parse_expression(terms[0])
        remaining_terms = [
            Mul(Number(-1), parse_expression(term)) for term in terms[1:]
        ]
        return Add(first_term, *remaining_terms)

    if "*" in expr_str:
        factors = split_by_operator(expr_str, "*")
        return Mul(*[parse_expression(factor) for factor in factors])

    if "^" in expr_str:
        base, exponent = split_by_operator(expr_str, "^")
        return Pow(parse_expression(base), parse_expression(exponent))

    if expr_str.isdigit() or (expr_str.startswith("-") and expr_str[1:].isdigit()):
        return Number(int(expr_str))

    return Variable(expr_str)


# Example usage
expr = parse_expression("2*x + 3*x")
print("Parsed Expression:", expr)
print("Simplified Expression:", expr.simplify())
