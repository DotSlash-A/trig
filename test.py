derivative = sp.diff("sin(x)**2", "x")
integral = sp.integrate("sin(x)**2", "x")
print(derivative, integral)

expr = sp.sympify("sin(x)**2 + cos(x)**2")
simplified = sp.simplify(expr)
print(simplified)  # Output: 1

solution = sp.solve("sin(x) - 0.5", "x")
print(solution)

sp.trigsimp(expression)

sp.solve(parsed_expr, x)
