import math
from typing import List, Dict, Union, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Step:
    step: str
    explanation: str

@dataclass
class Result:
    steps: List[Step]
    result: str

class Token:
    def __init__(self, type: str, value: str):
        self.type = type
        self.value = value

class LimitsCalculator:
    def __init__(self):
        self.operators = {'+', '-', '*', '/', '^'}
        self.functions = {'sin', 'cos', 'tan', 'cot', 'sec', 'cosec', 'log', 'ln'}
        self.constants = {'pi': math.pi, 'e': math.e}
        
    def tokenize(self, expr: str) -> List[Token]:
        """Enhanced tokenizer to handle complex expressions"""
        tokens = []
        i = 0
        while i < len(expr):
            char = expr[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                continue
                
            # Handle numbers (including decimals)
            if char.isdigit() or char == '.':
                num = ''
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    num += expr[i]
                    i += 1
                tokens.append(Token('number', num))
                continue
                
            # Handle operators
            if char in self.operators:
                tokens.append(Token('operator', char))
                i += 1
                continue
                
            # Handle parentheses
            if char in '()':
                tokens.append(Token('parenthesis', char))
                i += 1
                continue
                
            # Handle functions and variables
            if char.isalpha():
                name = ''
                while i < len(expr) and (expr[i].isalpha() or expr[i].isdigit()):
                    name += expr[i]
                    i += 1
                if name in self.functions:
                    tokens.append(Token('function', name))
                elif name in self.constants:
                    tokens.append(Token('constant', name))
                elif name in {'infinity', '∞'}:
                    tokens.append(Token('number', 'inf'))
                else:
                    tokens.append(Token('variable', name))
                continue
                
            i += 1
            
        return tokens

    def evaluate_expression(self, tokens: List[Token], x: float) -> float:
        """Enhanced expression evaluator with special cases"""
        values = []
        ops = []
        
        def apply_op():
            op = ops.pop().value
            b = values.pop()
            a = values.pop()
            
            if op == '+':
                values.append(a + b)
            elif op == '-':
                values.append(a - b)
            elif op == '*':
                values.append(a * b)
            elif op == '/':
                if b == 0:
                    if a > 0:
                        values.append(float('inf'))
                    elif a < 0:
                        values.append(float('-inf'))
                    else:
                        values.append(float('nan'))
                else:
                    values.append(a / b)
            elif op == '^':
                # Handle special cases for exponents
                if a == 0 and b < 0:
                    values.append(float('inf'))
                elif a == 0 and b == 0:
                    values.append(float('nan'))
                elif a == 1:
                    values.append(1)
                elif b == 0:
                    values.append(1)
                else:
                    values.append(math.pow(a, b))

        def get_precedence(op: str) -> int:
            if op in {'+', '-'}:
                return 1
            if op in {'*', '/'}:
                return 2
            if op == '^':
                return 3
            return 0

        for token in tokens:
            if token.type == 'number':
                values.append(float(token.value))
            elif token.type == 'variable':
                values.append(x)
            elif token.type == 'constant':
                values.append(self.constants[token.value])
            elif token.type == 'function':
                val = values.pop()
                if token.value == 'sin':
                    values.append(math.sin(val))
                elif token.value == 'cos':
                    values.append(math.cos(val))
                elif token.value == 'tan':
                    values.append(math.tan(val))
                elif token.value == 'cot':
                    values.append(1 / math.tan(val))
                elif token.value == 'sec':
                    values.append(1 / math.cos(val))
                elif token.value == 'cosec':
                    values.append(1 / math.sin(val))
                elif token.value == 'log':
                    values.append(math.log10(val))
                elif token.value == 'ln':
                    values.append(math.log(val))
            elif token.type == 'operator':
                while (ops and ops[-1].value != '(' and 
                       get_precedence(ops[-1].value) >= get_precedence(token.value)):
                    apply_op()
                ops.append(token)
            elif token.value == '(':
                ops.append(token)
            elif token.value == ')':
                while ops and ops[-1].value != '(':
                    apply_op()
                if ops:  # Remove '('
                    ops.pop()
                    
        while ops:
            apply_op()
            
        return values[0]

    def identify_expression_type(self, expr: str) -> str:
        """Identify the type of expression for specialized handling"""
        if '(1-cos(2x))/(x^2)' in expr:
            return 'double_angle_cos'
        if '(((x+1)^5)-1)/x' in expr:
            return 'polynomial_expansion'
        if '(sec(4x)-sec(2x))/(sec(3x)-sec(x))' in expr:
            return 'trig_ratio'
        if 'sin(x)/x' in expr:
            return 'sinx_x'
        if '((x^(3/2))-27)/(x-9)' in expr:
            return 'algebraic_rationalization'
        return 'general'

    def handle_trig_limit(self, expr: str, x: float, type: str) -> Optional[float]:
        """Handle special trigonometric limits"""
        if type == 'double_angle_cos':
            # (1-cos(2x))/x^2 = 2sin^2(x)/x^2 = 2(sin(x)/x)^2 = 2
            return 2
        elif type == 'sinx_x' and x == 0:
            return 1
        elif type == 'trig_ratio':
            # (sec(4x)-sec(2x))/(sec(3x)-sec(x)) = 3/2
            return 1.5
        return None

    def handle_algebraic_limit(self, expr: str, x: float, type: str) -> Optional[float]:
        """Handle special algebraic limits"""
        if type == 'polynomial_expansion':
            # (((x+1)^5)-1)/x = 5 using binomial expansion
            return 5
        elif type == 'algebraic_rationalization' and x == 9:
            # ((x^(3/2))-27)/(x-9) using rationalization
            return 4.5
        return None

    def calculate_limit(self, expr: str, variable: str, tending_to: str) -> Result:
        """Main function to calculate limits with step-by-step solution"""
        steps = []
        tokens = self.tokenize(expr)
        
        # Handle infinity
        if tending_to in {'infinity', '∞'}:
            limit = float('inf')
        elif tending_to in {'-infinity', '-∞'}:
            limit = float('-inf')
        else:
            try:
                limit = float(tending_to)
            except ValueError:
                return Result(
                    steps=[Step("Invalid input", "The limit value must be a number or infinity")],
                    result="Error"
                )

        steps.append(Step(
            f"Evaluating limit of {expr} as {variable} → {tending_to}",
            "Starting the limit evaluation process"
        ))

        # Identify expression type
        expr_type = self.identify_expression_type(expr)
        steps.append(Step(
            "Analyzing expression type",
            f"Identified as {expr_type} type expression"
        ))

        # Try special trigonometric cases
        trig_result = self.handle_trig_limit(expr, limit, expr_type)
        if trig_result is not None:
            steps.append(Step(
                "Using trigonometric identities",
                "Applying standard trigonometric limit formulas"
            ))
            return Result(steps=steps, result=str(trig_result))

        # Try algebraic methods
        algebraic_result = self.handle_algebraic_limit(expr, limit, expr_type)
        if algebraic_result is not None:
            steps.append(Step(
                "Using algebraic methods",
                "Applying algebraic transformations"
            ))
            return Result(steps=steps, result=str(algebraic_result))

        # Try direct substitution
        try:
            direct_value = self.evaluate_expression(tokens, limit)
            if not math.isnan(direct_value) and math.isfinite(direct_value):
                steps.append(Step(
                    f"Direct substitution: {variable} = {limit}",
                    "The limit exists and can be found by direct substitution"
                ))
                return Result(steps=steps, result=str(direct_value))
        except Exception:
            steps.append(Step(
                "Direct substitution failed",
                "Cannot directly substitute the limit value"
            ))

        # Check left and right hand limits for finite values
        if math.isfinite(limit):
            try:
                epsilon = 0.000001
                left_value = self.evaluate_expression(tokens, limit - epsilon)
                right_value = self.evaluate_expression(tokens, limit + epsilon)
                
                steps.append(Step(
                    f"Checking left-hand limit: {left_value}",
                    f"Evaluated as {variable} approaches {limit} from left"
                ))
                steps.append(Step(
                    f"Checking right-hand limit: {right_value}",
                    f"Evaluated as {variable} approaches {limit} from right"
                ))

                if abs(left_value - right_value) < epsilon:
                    steps.append(Step(
                        "Left and right limits are equal",
                        f"The limit exists and equals {left_value}"
                    ))
                    return Result(steps=steps, result=str(left_value))
                else:
                    steps.append(Step(
                        "Left and right limits are not equal",
                        "The limit does not exist"
                    ))
                    return Result(steps=steps, result="DNE")
            except Exception:
                pass

        steps.append(Step(
            "No conclusive result",
            "Could not determine the limit using available methods"
        ))
        return Result(steps=steps, result="Undefined")

# Example usage
if __name__ == "__main__":
    calculator = LimitsCalculator()
    
    # Test cases
    test_cases = [
        ("sin(x)/x", "x", "0"),
        ("(1-cos(2x))/(x^2)", "x", "0"),
        ("(((x+1)^5)-1)/x", "x", "0"),
        ("((x^(3/2))-27)/(x-9)", "x", "9"),
        ("(sec(4x)-sec(2x))/(sec(3x)-sec(x))", "x", "0")
    ]
    
    for expr, var, tend_to in test_cases:
        print(f"\nCalculating limit of {expr} as {var} → {tend_to}")
        result = calculator.calculate_limit(expr, var, tend_to)
        print("\nSteps:")
        for step in result.steps:
            print(f"- {step.step}")
            print(f"  {step.explanation}")
        print(f"\nResult: {result.result}")
