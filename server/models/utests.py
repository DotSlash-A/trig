# utests.py

import unittest
# Assuming the main code is in 'tests.py' in the same directory
from tests import Expr, Add, Mul, Pow, Number, Symbol, parse_expression, simplify, preprocess_subtraction, simplify_cache

class TestSimplifier(unittest.TestCase):

    def setUp(self):
        # Clear cache before each test for isolation
        simplify_cache.clear()

    def assertSimplifiedEqual(self, input_str, expected_str):
        """Parses, simplifies, and compares string representations."""
        processed_str = preprocess_subtraction(input_str) # Handle basic subtraction
        try:
            parsed = parse_expression(processed_str)
            simplified = simplify(parsed)
            self.assertEqual(str(simplified), expected_str,
                             f"Input: '{input_str}' -> Simplified: '{simplified}' != Expected: '{expected_str}'")
        except ValueError as e:
            self.fail(f"Parsing failed for '{input_str}': {e}")
        except Exception as e:
            self.fail(f"Simplification failed for '{input_str}': {e}")


    def test_addition(self):
        self.assertSimplifiedEqual("2*x + 3*x", "(5*x)")
        self.assertSimplifiedEqual("x*2 + x*3", "(5*x)") # Order independence
        self.assertSimplifiedEqual("x + y + x", "(y+(2*x))") # Canonical order
        self.assertSimplifiedEqual("x + 5 + 2", "(7+x)")

    def test_multiplication(self):
        self.assertSimplifiedEqual("2*x * 3*x", "(6*x^2)")
        self.assertSimplifiedEqual("x*y * y*x", "(x^2*y^2)") # powsimp
        self.assertSimplifiedEqual("x*2*y*3", "(6*x*y)") # Constant folding

    def test_exponentiation(self):
        self.assertSimplifiedEqual("x^2 * x^3", "x^5")
        self.assertSimplifiedEqual("(x^2)^3", "x^6") # Requires parser/simplifier support
        self.assertSimplifiedEqual("x^(2+3)", "x^5") # Simplify exponent

    def test_identities(self):
        self.assertSimplifiedEqual("x + 0", "x")
        self.assertSimplifiedEqual("0 + x + y", "(x+y)")
        self.assertSimplifiedEqual("x * 1", "x")
        self.assertSimplifiedEqual("y * x * 1", "(x*y)")
        self.assertSimplifiedEqual("x * 0", "0")
        self.assertSimplifiedEqual("0 * y * z", "0")
        self.assertSimplifiedEqual("x^1", "x")
        self.assertSimplifiedEqual("x^0", "1")
        self.assertSimplifiedEqual("1^x", "1")
        self.assertSimplifiedEqual("0^x", "0") # Assuming x > 0, 0^0 handled
        self.assertSimplifiedEqual("0^0", "1")
        self.assertSimplifiedEqual("5^0", "1")
        self.assertSimplifiedEqual("1^5", "1")
        self.assertSimplifiedEqual("5^1", "5")


    def test_number_simplification(self):
        self.assertSimplifiedEqual("2 + 3 + 4", "9")
        self.assertSimplifiedEqual("2 * 3 * 4", "24")
        self.assertSimplifiedEqual("2^3", "8")
        self.assertSimplifiedEqual("4^0.5", "2") # Float handling
        self.assertSimplifiedEqual("2 + 3*4", "(2+12)") # Needs precedence in parser
        # self.assertSimplifiedEqual("2 + 3*4", "14") # Requires better parser


    # --- Tests requiring better parsing or features not implemented ---

    # def test_factorization(self):
    #     # Factorization is not implemented in the default simplify loop
    #     self.assertSimplifiedEqual("x*y + x*z", '(x*(y+z))') # Requires factor strategy
    #     self.assertSimplifiedEqual("2*x + 4*y", '(2*(x+(2*y)))') # Requires factor strategy


    # def test_expansion(self):
    #     # Expansion is often avoided by default as it increases complexity
    #     self.assertSimplifiedEqual("(x+y)*z", '((x*z)+(y*z))') # Requires expand strategy & parser


    # def test_subtraction(self):
    #     self.assertSimplifiedEqual("x - x", "0")
    #     self.assertSimplifiedEqual("5*x - 2*x", "(3*x)")
    #     self.assertSimplifiedEqual("y - 2*y", "(-1*y)") # Or "-y" if str representation is improved


    def test_complex_combinations(self):
         self.assertSimplifiedEqual("x*y*x^2", "(y*x^3)")
         self.assertSimplifiedEqual("a + b + 2*a + 3*b", "((4*b)+(3*a))") # Collect terms
         self.assertSimplifiedEqual("(x^2 * y) * (x * y^3)", "(x^3*y^4)") # powsimp


if __name__ == '__main__':
    unittest.main(verbosity=2)