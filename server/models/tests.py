# tests.py (or models.py)

from typing import Union, List, Tuple
import re
import math  # For potential float comparisons/checks


# --- Core Classes (Expr, Add, Mul, Pow, Number, Symbol) ---
# (Keep these classes as they were, including __hash__ methods for Number and Symbol)
class Expr:
    """Base Class for all symbolic expressions"""

    def __init__(self, *args):
        # Ensure args are Expr, Number, or Symbol instances if possible (or convert basic types)
        # For simplicity, assume constructor gets correct types for now.
        self.args = args

    def __eq__(self, other):
        return type(self) == type(other) and self.args == other.args

    def __lt__(self, other):
        # Define comparison for sorting (canonical order)
        return str(self) < str(other)

    def __hash__(self):
        # Hash based on type and args tuple
        return hash((type(self), self.args))

    def __repr__(self):
        # Provide a more debug-friendly representation
        args_repr = ", ".join(map(repr, self.args))
        return f"{type(self).__name__}({args_repr})"


class Add(Expr):
    """Represents mathematical addition"""

    def __str__(self):
        # Sort arguments for canonical string representation
        return f"({'+'.join(map(str, sorted(self.args, key=str)))})"


class Mul(Expr):
    """Represents mathematical multiplication"""

    def __str__(self):
        # Sort arguments for canonical string representation
        return f"({'*'.join(map(str, sorted(self.args, key=str)))})"


class Pow(Expr):
    """Represents power operation"""

    def __str__(self):
        base, exp = self.args
        # Add parentheses for clarity if base is Add or Mul? Not strictly needed for internal representation.
        return f"{base}^{exp}"


class Number(Expr):
    def __init__(self, value):
        # Store value, handling potential float conversion to int
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.value = value
        super().__init__()  # No arguments for Number

    @property
    def args(self):  # Override args property for Number
        return ()

    def __eq__(self, other):
        # Handle potential float comparison issues if needed
        return type(self) == type(other) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"Number({self.value})"


class Symbol(Expr):
    def __init__(self, name):
        self.name = name
        super().__init__()  # No arguments for Symbol

    @property
    def args(self):  # Override args property for Symbol
        return ()

    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Symbol('{self.name}')"


# --- Parsing (Simplified - requires Flattening & Correct Precedence Handling) ---
# Using the improved parser from the previous thought process
def parse_expression(expr: str) -> Expr:
    """Converts a string expression into Expr Objects (simple, no parentheses or strict precedence yet)"""
    expr_str = expr.strip().replace(" ", "")
    if not expr_str:
        raise ValueError("Empty expression string")
    expr_str = expr_str.replace("**", "^")

    # Handle addition (lowest precedence for this simple parser)
    # Split only top-level additions (need more robust parsing for parentheses)
    # This simple split assumes no parentheses interfere
    split_add = expr_str.split("+")
    if len(split_add) > 1:
        parsed_terms = [parse_expression(term) for term in split_add]
        # Flatten nested additions
        flat_terms = []
        for t in parsed_terms:
            if isinstance(t, Add):
                flat_terms.extend(t.args)
            elif not (isinstance(t, Number) and t.value == 0):  # Don't add Add(0)
                flat_terms.append(t)
        if not flat_terms:
            return Number(0)
        if len(flat_terms) == 1:
            return flat_terms[0]
        return Add(*flat_terms)

    # Handle multiplication
    split_mul = expr_str.split("*")
    if len(split_mul) > 1:
        parsed_factors = [parse_expression(factor) for factor in split_mul]
        # Flatten nested multiplications
        flat_factors = []
        has_zero = False
        for f in parsed_factors:
            if isinstance(f, Mul):
                flat_factors.extend(f.args)
            elif isinstance(f, Number) and f.value == 0:
                has_zero = True
                break
            elif not (isinstance(f, Number) and f.value == 1):  # Don't add Mul(1)
                flat_factors.append(f)
        if has_zero:
            return Number(0)
        if not flat_factors:
            return Number(1)  # Product of 1s
        if len(flat_factors) == 1:
            return flat_factors[0]
        return Mul(*flat_factors)

    # Handle exponentiation (highest precedence)
    # Split only on the *last* '^' for right-associativity (x^y^z = x^(y^z))
    if "^" in expr_str:
        base_str, exp_str = expr_str.rsplit("^", 1)
        # Need error handling for invalid splits like "^2" or "2^"
        if not base_str or not exp_str:
            raise ValueError(f"Invalid power expression: {expr_str}")
        base = parse_expression(base_str)
        exp = parse_expression(exp_str)
        # Basic power simplifications during parsing
        if isinstance(exp, Number):
            if exp.value == 0:
                return Number(1)
            if exp.value == 1:
                return base
        if isinstance(base, Number):
            if base.value == 0:
                return Number(0)  # Assumes exp > 0
            if base.value == 1:
                return Number(1)
        return Pow(base, exp)

    # Handle Numbers
    if re.fullmatch(r"-?\d+(\.\d+)?", expr_str):
        num_val = float(expr_str) if "." in expr_str else int(expr_str)
        return Number(num_val)

    # Handle variables
    if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", expr_str):  # Allow multi-char variables
        return Symbol(expr_str)

    # Handle potential parentheses (very basic: remove if they surround the whole thing)
    if expr_str.startswith("(") and expr_str.endswith(")"):
        # This is NOT a proper parenthesis handling, just removes outer ones
        # A real parser needs a stack or recursive descent.
        try:
            return parse_expression(expr_str[1:-1])
        except ValueError:  # If content inside wasn't valid
            pass  # Fall through to raise error

    raise ValueError(f"Invalid expression component: {expr_str}")


# --- Simplification Rules ---


def simplify_constants(expr: Expr) -> Expr:
    """Evaluates constant numerical sub-expressions. Assumes args are simplified."""
    if isinstance(expr, Number) or isinstance(expr, Symbol):
        return expr  # Already simplified

    # --- Add ---
    if isinstance(expr, Add):
        new_args = []
        current_sum = 0
        for arg in expr.args:
            if isinstance(arg, Number):
                current_sum += arg.value
            else:
                new_args.append(arg)

        if not new_args:
            return Number(current_sum)  # Only numbers

        # Only add constant term if non-zero or if it's the only term left
        if current_sum != 0 or len(new_args) == 0:
            # Use Number constructor for potential float->int conversion
            new_args.append(Number(current_sum))

        if len(new_args) == 1:
            return new_args[0]
        return Add(*new_args)  # Don't sort here, sorting happens in simplify loop

    # --- Mul ---
    if isinstance(expr, Mul):
        new_args = []
        current_prod = 1
        for arg in expr.args:
            if isinstance(arg, Number):
                current_prod *= arg.value
            else:
                new_args.append(arg)

        if math.isclose(current_prod, 0):
            return Number(0)  # Multiplication by zero (use isclose for floats)

        if not new_args:
            return Number(current_prod)  # Only numbers

        # Only add constant factor if != 1 or if it's the only term left
        if not math.isclose(current_prod, 1) or len(new_args) == 0:
            # Use Number constructor for potential float->int conversion
            new_args.insert(0, Number(current_prod))  # Keep constant first-ish

        if len(new_args) == 1:
            return new_args[0]
        return Mul(*new_args)  # Don't sort here

    # --- Pow ---
    if isinstance(expr, Pow):
        base, exp = expr.args
        # Numerical evaluation
        if isinstance(base, Number) and isinstance(exp, Number):
            # Handle 0^0 -> 1 (convention)
            if math.isclose(base.value, 0) and math.isclose(exp.value, 0):
                return Number(1)
            # Handle 0^neg -> ComplexInfinity (or error) - Let's return original for now
            if math.isclose(base.value, 0) and exp.value < 0:
                return expr  # Cannot represent easily
            try:
                val = base.value**exp.value
                # Check for complex results if needed, handle large numbers/precision
                if isinstance(val, complex):
                    return expr  # Cannot represent complex numbers yet
                return Number(val)  # Constructor handles float->int
            except (OverflowError, ValueError):
                pass  # Cannot simplify numerically

        # Identity simplifications
        if isinstance(exp, Number):
            if math.isclose(exp.value, 0):
                return Number(1)  # x^0 = 1
            if math.isclose(exp.value, 1):
                return base  # x^1 = x
        if isinstance(base, Number):
            # Assumes exp > 0 due to checks above
            if math.isclose(base.value, 0):
                return Number(0)  # 0^x = 0
            if math.isclose(base.value, 1):
                return Number(1)  # 1^x = 1

        # Power of Power: (x^a)^b -> x^(a*b)
        if isinstance(base, Pow):
            inner_base, inner_exp = base.args
            new_exp = simplify(Mul(inner_exp, exp))  # Simplify the new exponent
            return Pow(inner_base, new_exp)

        return Pow(base, exp)  # Cannot simplify

    return expr  # Should not be reached for Add/Mul/Pow


def simplify_identities(expr: Expr) -> Expr:
    """Applies identity rules like Add(x, 0), Mul(x, 1). Assumes args are simplified."""
    if isinstance(expr, Add):
        # Remove Add(0) terms
        new_args = [
            arg
            for arg in expr.args
            if not (isinstance(arg, Number) and math.isclose(arg.value, 0))
        ]
        if len(new_args) == 0:
            return Number(0)
        if len(new_args) == 1:
            return new_args[0]
        # Only return new Add if args actually changed
        if len(new_args) < len(expr.args):
            return Add(*new_args)

    if isinstance(expr, Mul):
        # Remove Mul(1) terms, check for Mul(0)
        new_args = []
        has_zero = False
        for arg in expr.args:
            if isinstance(arg, Number):
                if math.isclose(arg.value, 0):
                    has_zero = True
                    break
                elif not math.isclose(arg.value, 1):
                    new_args.append(arg)
            else:
                new_args.append(arg)

        if has_zero:
            return Number(0)
        if len(new_args) == 0:
            return Number(1)  # Product of 1s
        if len(new_args) == 1:
            return new_args[0]
        # Only return new Mul if args actually changed
        if len(new_args) < len(expr.args):
            return Mul(*new_args)

    # Pow identities are mostly handled in simplify_constants
    if isinstance(expr, Pow):
        base, exp = expr.args
        if isinstance(exp, Number) and math.isclose(exp.value, 1):
            return base
        if isinstance(base, Number) and math.isclose(base.value, 1):
            return Number(1)
        # Could add more here if needed

    return expr  # No change


def collect_like_terms(expr: Add) -> Expr:
    """Combine like terms in an Add expression. Assumes args are simplified."""
    term_dict = (
        {}
    )  # Key: base term (Symbol, Pow, or Mul), Value: coefficient (Number value)
    constant_term_val = 0.0

    for term in expr.args:
        coeff_val = 1.0
        base = term

        if isinstance(term, Number):
            constant_term_val += term.value
            continue

        if isinstance(term, Mul):
            # Extract coefficient and base
            new_mul_args = []
            num_coeff_val = 1.0
            for arg in term.args:
                if isinstance(arg, Number):
                    num_coeff_val *= arg.value
                else:
                    new_mul_args.append(arg)

            if math.isclose(num_coeff_val, 0):
                continue  # Term is zero

            coeff_val = num_coeff_val
            if not new_mul_args:  # Mul was just numbers
                constant_term_val += coeff_val
                continue
            elif len(new_mul_args) == 1:
                base = new_mul_args[0]
            else:
                # Canonical representation for base (sorted Mul)
                base = Mul(*sorted(new_mul_args, key=str))

        # Add coefficient value to term_dict
        current_coeff_val = term_dict.get(base, 0.0)
        term_dict[base] = current_coeff_val + coeff_val

    # Build the new expression
    new_terms = []
    if not math.isclose(constant_term_val, 0.0):
        new_terms.append(Number(constant_term_val))

    for base, coeff_val in term_dict.items():
        if math.isclose(coeff_val, 0.0):
            continue  # Skip terms with zero coefficient

        coeff_num = Number(coeff_val)
        if math.isclose(coeff_val, 1.0):
            new_terms.append(base)
        else:
            # Reconstruct the term: coeff * base
            if isinstance(base, Mul):
                # Add coefficient back into Mul args
                final_args = [coeff_num] + list(base.args)
                new_terms.append(Mul(*final_args))  # Let simplify sort later
            else:
                new_terms.append(Mul(coeff_num, base))

    if not new_terms:
        return Number(0)
    if len(new_terms) == 1:
        # Make sure simplify is called on the result if it came from Mul construction
        # This is handled by the main simplify loop structure
        return new_terms[0]

    # Return new Add expression
    return Add(*new_terms)  # Let simplify sort later


def powsimp(expr: Mul) -> Expr:
    """Simplify power expressions in a Mul: (x^a)*(x^b) -> x^(a+b). Assumes args are simplified."""
    base_exponents = {}  # Key: base (Symbol or Expr), Value: exponent (Expr)
    numeric_coeff_val = 1.0
    other_factors = []  # Factors that are not Pow or Number

    # Pass 1: Group factors by base
    for factor in expr.args:
        if isinstance(factor, Number):
            numeric_coeff_val *= factor.value
            continue

        if math.isclose(numeric_coeff_val, 0):
            return Number(0)  # Optimization

        base = factor
        exp = Number(1)
        if isinstance(factor, Pow):
            base, exp = factor.args

        current_exp = base_exponents.get(base)
        if current_exp is not None:
            # Combine exponents: simplify(Add(current_exp, exp))
            # Crucially, simplify the combined exponent recursively!
            new_exp = simplify(Add(current_exp, exp))
            base_exponents[base] = new_exp
        else:
            # Store base with its exponent (might be Number(1))
            base_exponents[base] = exp

    # Pass 2: Rebuild the expression
    new_factors = []
    if not math.isclose(numeric_coeff_val, 1.0) or not base_exponents:
        # Add numeric coeff unless it's 1 (and there are other terms)
        new_factors.append(Number(numeric_coeff_val))

    for base, exp in base_exponents.items():
        # Check exponent value after potential simplification
        if isinstance(exp, Number):
            if math.isclose(exp.value, 0):
                continue  # x^0 = 1 (absorbed into numeric coeff)
            if math.isclose(exp.value, 1):
                new_factors.append(base)  # x^1 = x
                continue
        # Append as Pow otherwise
        new_factors.append(Pow(base, exp))

    if not new_factors:
        # If factors were only x^0 or numeric coeff was 1
        return Number(numeric_coeff_val if math.isclose(numeric_coeff_val, 0) else 1)
    if len(new_factors) == 1:
        return new_factors[0]

    return Mul(*new_factors)  # Let simplify sort later


# --- Main Simplification Function ---

# Memoization cache
simplify_cache = {}


def simplify(expr: Expr) -> Expr:
    """Recursively simplifies the expression using a set of rules."""
    # Check cache
    if expr in simplify_cache:
        return simplify_cache[expr]

    # Base case: Numbers and Symbols are already simplified.
    if isinstance(expr, Number) or isinstance(expr, Symbol):
        simplify_cache[expr] = expr
        return expr

    # Step 1: Recursively simplify arguments
    try:
        simplified_args = tuple(simplify(arg) for arg in expr.args)
    except Exception as e:
        print(f"Error simplifying args of: {expr}")
        raise e

    # If args didn't change, expr type is same, return original (or cached)
    # This check helps break cycles if rules don't change args but expr hash is same
    if simplified_args == expr.args:
        simplified_expr = expr
    else:
        simplified_expr = type(expr)(*simplified_args)

    # Step 2: Apply simplification rules repeatedly until no further change
    # Use string comparison to detect stability reliably
    previous_expr_str = ""
    while str(simplified_expr) != previous_expr_str:
        previous_expr_str = str(simplified_expr)

        # Apply rules in a chosen order (can affect performance/result)
        # 1. Identities (often expose other simplifications)
        simplified_expr = simplify_identities(simplified_expr)
        # 2. Constant folding (evaluate numerical parts)
        simplified_expr = simplify_constants(simplified_expr)
        # Check if simplified to non-Expr base type
        if isinstance(simplified_expr, (Number, Symbol)):
            break

        # 3. collect_like_terms (for Add)
        if isinstance(simplified_expr, Add):
            simplified_expr = collect_like_terms(simplified_expr)
            if isinstance(simplified_expr, (Number, Symbol)):
                break  # Fully simplified
            # Need to ensure result is simplified if it changed structure (e.g., to Mul)
            if not isinstance(simplified_expr, Add):
                simplified_expr = simplify(simplified_expr)  # Re-simplify result
                continue  # Restart rule loop with potentially different type

        # 4. powsimp (for Mul)
        if isinstance(simplified_expr, Mul):
            simplified_expr = powsimp(simplified_expr)
            if isinstance(simplified_expr, (Number, Symbol)):
                break  # Fully simplified
            if not isinstance(simplified_expr, Mul):
                simplified_expr = simplify(simplified_expr)  # Re-simplify result
                continue  # Restart rule loop

        # 5. Flatten nested Add/Mul (results in canonical form)
        if isinstance(simplified_expr, Add):
            new_args = []
            changed = False
            for arg in simplified_expr.args:
                if isinstance(arg, Add):
                    new_args.extend(arg.args)
                    changed = True
                else:
                    new_args.append(arg)
            if changed:
                simplified_expr = Add(*new_args)

        if isinstance(simplified_expr, Mul):
            new_args = []
            changed = False
            for arg in simplified_expr.args:
                if isinstance(arg, Mul):
                    new_args.extend(arg.args)
                    changed = True
                else:
                    new_args.append(arg)
            if changed:
                simplified_expr = Mul(*new_args)

        # 6. Sort arguments for canonical form (Add/Mul)
        # This should be the *last* step in the loop to ensure stability
        current_str_before_sort = str(simplified_expr)
        if isinstance(simplified_expr, (Add, Mul)):
            simplified_expr = type(simplified_expr)(
                *sorted(simplified_expr.args, key=str)
            )
        # Only continue loop if sorting changed the string representation
        if (
            str(simplified_expr) == current_str_before_sort
            and str(simplified_expr) == previous_expr_str
        ):
            break  # Stable if sorting didn't change and no other rule did

    # Cache the final simplified result
    simplify_cache[expr] = simplified_expr
    # Also cache the result itself, in case it was reached via different paths
    if simplified_expr not in simplify_cache:
        simplify_cache[simplified_expr] = simplified_expr

    return simplified_expr


# --- Testing ---
if __name__ == "__main__":
    test_expressions = [
        "2*x + 3*x",  # Expected: 5*x
        "x*2 + x*3",  # Expected: 5*x (order shouldn't matter)
        "2*x * 3*x",  # Expected: 6*x^2
        "x^2 * x^3",  # Expected: x^5
        "x*y + y*x",  # Expected: 2*x*y (canonical order)
        "(x^2)^3",  # Expected: x^6 (Parser needs Pow of Pow) -> Pow(Pow(x,2),3)
        "x + 0",  # Expected: x
        "x * 1",  # Expected: x
        "x * 0",  # Expected: 0
        "x^1",  # Expected: x
        "x^0",  # Expected: 1
        "1^x",  # Expected: 1
        "0^x",  # Expected: 0 (needs handling for x=0 case)
        "2 + 3 + x",  # Expected: 5+x
        "2 * 3 * x",  # Expected: 6*x
        "x + y + x",  # Expected: 2*x + y
        "x*y*x*z",  # Expected: x^2*y*z
        "1 + 1",  # Expected: 2
        "x*y + x*z",  # Factorization not implemented by default
        "2*x + 4*y",  # Factorization not implemented
        "x^2 * y^3 * x^4",  # Expected: x^6 * y^3
        "x + y",  # Expected: x + y
        "x - x",  # Need subtraction handling -> x + (-1)*x
    ]

    # Simple subtraction preprocessing
    def preprocess_subtraction(expr_str):
        # Replace 'a - b' with 'a + (-1)*b' carefully
        # This needs a more robust regex or parsing approach
        # Basic version (might fail on complex cases like 'a - -b' or 'a - (b+c)')
        expr_str = re.sub(r"-\s*([a-zA-Z0-9\.]+)", r"+(-1)*\1", expr_str)
        # Handle start like '-x' -> '(-1)*x'
        if expr_str.startswith("-"):
            expr_str = "(-1)*" + expr_str[1:]
        return expr_str

    simplify_cache.clear()  # Clear cache before running tests
    for expr_str in test_expressions:
        # Use the function now defined globally
        processed_str = preprocess_subtraction(expr_str)
        print(f"Original:   {expr_str}")
        print(f"Processed:  {processed_str}")  # See preprocessed string
        try:
            parsed_expr = parse_expression(processed_str)
            print(f"Parsed:     {parsed_expr!r}")  # Show repr
            simplified_expr = simplify(parsed_expr)
            print(f"Simplified: {simplified_expr}")
        except ValueError as e:
            print(f"Parse Error: {e}")
        except Exception as e:
            print(f"Simplify Error: {e}")
            import traceback

            traceback.print_exc()
        print("-" * 20)
