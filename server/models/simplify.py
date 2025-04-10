from typing import Union
import re

class Expr:
	"""Base Class for all symbolic epressions"""
	def __init__(self, *args):
		self.args = args
	def __eq__(self, other):
		return type(self) == type(other) and self.args == other.args
	
	def replace(self, func):
		"""Apply func to all subexpressions"""
		new_args = [arg.replace(func) if isinstance(arg, Expr) else arg for arg in self.args]
		return func(type(self)(*new_args))

class Add(Expr):
	"""Represents mathematical addition"""
	def __str__(self):
		return f"({'+'.join(map(str, self.args))})"
class Mul(Expr):
	"""Represents mathematical multiplication"""
	def __str__(self):
		return f"({'*'.join(map(str, self.args))})"
	
class Pow(Expr):
	"""Represents power operation"""
	def __str__(self):
		base, exp= self.args
		return f"{base}^{exp}"

class Number(Expr):
	def __init__(self, value):
		self.value = value
	def __str__(self):
		return str(self.value)
	
	

#-----------------------
# Parsing String expressions into Expr Objcts
# -----------------------

def parse_expression(expr: str) -> Expr:
	"""Converts a string expression into Expr Obejects"""
	expr_str=expr_str.replace(" ", "")
	expr_str=expr_str.replace("**", "^")

	#Handle Numbers
	if re.fullmatch(r"-?\d+(\.\d+)?", expr_str):
		return Number(float(expr_str) if '.' in expr_str else int(expr_str))
	
	#handles variables
	if re.fullmatch(r"[a-zA-Z]", expr_str):
		return Number(expr_str)
	
	#hanfdles exponentiation
	if "^" in expr_str:
		base, exp = expr_str.split("^")
		return Pow(parse_expression(base), parse_expression(exp))
	
	#handles multiplication and addition
	if "*" in expr_str:
		factors = expr_str.split("*")
		return Mul(*(parse_expression(factor) for factor in factors))
	
	if "+" in expr_str:
		terms = expr_str.split("+")
		return Add(*(parse_expression(term) for term in terms))

	raise ValueError(f"Invalid expression: {expr_str}")

#-----------------------
# Simplifying expressions
# -----------------------

def collect_like_terms(expr):
	"""Combine like terms in an expression"""
	if isinstance(expr, Add):
		term_dict={}
		for term in expr.args:
			if isinstance(term, Mul):
				coeff, symbol=term.args if isinstance(term.args[0], Number) else (Number(1), term)
			elif isinstance(term, None):
				coeff, symbol=Number(1), None
			else:
				coeff, symbol=Number(1), term
			if symbol in term_dict:
				term_dict[symbol]=Number(term_dict[symbol].value + coeff.value)
			else:
				term_dict[symbol]=coeff
		new_terms=[Mul(k,v) if k else v for k,v in term_dict.items()]
		return Add(*new_terms) if len(new_terms) >1 else new_terms[0]
	return expr

def expland(expr):
	"""Distributes multiplication over addition: (x+y)*z--> x*z + y*z"""
	if isinstance(expr, Mul):
		expanded_terms=[]
		for term in expr.args:
			if isinstance(term, Add):
				for subterm in term.args:
					expanded_terms.append(Mul(subterm, *[t for t in expr.args if t !=term]))
				return Add(*expanded_terms)
	return expr


def factor(expr):
    """Factor out common terms."""
    if isinstance(expr, Add):
        common_factors = set(expr.args[0].args) if isinstance(expr.args[0], Mul) else {expr.args[0]}
        for term in expr.args[1:]:
            if isinstance(term, Mul):
                common_factors.intersection_update(term.args)
            else:
                common_factors.intersection_update({term})
        
        if common_factors:
            common_factor = min(common_factors, key=str)  # Pick the simplest factor
            factored_terms = [Mul(*(set(term.args) - {common_factor})) if isinstance(term, Mul) else Number(1) for term in expr.args]
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
        
        return Mul(*[Pow(b, e) if e != Number(1) else b for b, e in base_exponents.items()])
    
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
def simplify(expr):
    """Applies multiple strategies and picks the best one."""
    strategies = [collect_like_terms, expand, factor, powsimp]
    
    best_expr = expr
    min_score = expression_score(expr)

    for strategy in strategies:
        new_expr = strategy(expr)
        new_score = expression_score(new_expr)

        if new_score < min_score:
            best_expr = new_expr
            min_score = new_score

    return best_expr

# -----------------------------------------------
# Testing the Simplifier
# -----------------------------------------------
if __name__ == "__main__":
    expr_str = "2*x + 3*x"
    parsed_expr = parse_expression(expr_str)
    simplified_expr = simplify(parsed_expr)

    print(f"Original: {parsed_expr}")
    print(f"Simplified: {simplified_expr}")