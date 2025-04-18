import math


def solve_slope_formula(find_var, **kwargs):
    """
    Solves for a missing variable in the slope formula m = (y2 - y1) / (x2 - x1).

    Args:
        find_var (str): The variable to find ('x1', 'y1', 'x2', 'y2', 'm').
        **kwargs: Dictionary containing the known variables and their values.
                  Should contain exactly four key-value pairs from x1, y1, x2, y2, m.

    Returns:
        float or str: The calculated value of the missing variable, or an error message.
    """
    vars = {"x1", "y1", "x2", "y2", "m"}
    if find_var not in vars:
        return (
            "Error: Invalid variable to find. Choose from 'x1', 'y1', 'x2', 'y2', 'm'."
        )
    if len(kwargs) != 4:
        return f"Error: Exactly 4 known variables are required, but {len(kwargs)} were provided."
    if not set(kwargs.keys()).issubset(vars - {find_var}):
        return f"Error: Incorrect known variables provided for finding '{find_var}'."

    x1 = kwargs.get("x1")
    y1 = kwargs.get("y1")
    x2 = kwargs.get("x2")
    y2 = kwargs.get("y2")
    m = kwargs.get("m")

    try:
        if find_var == "x2":
            if m == 0:
                return (
                    "Error: Cannot solve for x2 when slope (m) is 0 (horizontal line)."
                )
            return x1 + (y2 - y1) / m
        elif find_var == "x1":
            if m == 0:
                return (
                    "Error: Cannot solve for x1 when slope (m) is 0 (horizontal line)."
                )
            return x2 - (y2 - y1) / m
        elif find_var == "y2":
            return y1 + m * (x2 - x1)
        elif find_var == "y1":
            return y2 - m * (x2 - x1)
        elif find_var == "m":
            if x2 - x1 == 0:
                return "Error: Cannot calculate slope (m) when x1 and x2 are equal (vertical line)."
            return (y2 - y1) / (x2 - x1)
    except TypeError:
        # This might happen if a required variable for the calculation is None
        return f"Error: Missing one of the required known variables for finding '{find_var}'."
    except ZeroDivisionError:
        # This is another way division by zero might manifest, though explicitly checked
        return "Error: Division by zero encountered during calculation."


if __name__ == "__main__":
    print("Slope Formula Calculator: m = (y2 - y1) / (x2 - x1)")

    while True:
        find_variable = input(
            "Which variable do you want to find? (x1, y1, x2, y2, m): "
        ).lower()
        if find_variable in ["x1", "y1", "x2", "y2", "m"]:
            break
        else:
            print("Invalid input. Please enter one of 'x1', 'y1', 'x2', 'y2', 'm'.")

    known_values = {}
    variables_needed = {"x1", "y1", "x2", "y2", "m"} - {find_variable}

    for var in variables_needed:
        while True:
            try:
                value = float(input(f"Enter the value for {var}: "))
                known_values[var] = value
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    result = solve_slope_formula(find_variable, **known_values)

    if isinstance(result, str) and result.startswith("Error"):
        print(result)
    else:
        print(f"The calculated value for {find_variable} is: {result}")
