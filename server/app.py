from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sympy as sp

app = FastAPI()

class Expression(BaseModel):
    equation: str

@app.post("/solve")
async def solve_expression(expression: Expression):
    try:
        x = sp.symbols('x')  # Define variable 'x'
        parsed_expr = sp.sympify(expression.equation)  # Parse the equation
        simplified_expr = sp.simplify(parsed_expr)  # Simplify the equation
        steps = sp.pretty(simplified_expr, use_unicode=True)  # Generate steps
        return {"result": str(simplified_expr), "steps": steps}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# @app.post("/differentiate")
# async def differentiate_expression(expression: Expression):
#     try:
#         x = sp.symbols('x')  # Define variable 'x'
#         parsed_expr = sp.sympify(expression.equation)  # Parse the equation
#         differentiated_expr = sp.diff(parsed_expr, x)  # Differentiate the equation
#         print(differentiated_expr)
#         print("***********")
#         steps = sp.pretty(differentiated_expr, use_unicode=True)  # Generate steps
#         print(steps)
#         return {"result": str(differentiated_expr), "steps": steps}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))


@app.post("/differentiate")
async def differentiate_expression(expression: Expression):
    try:
        x = sp.symbols('x')  # Define variable 'x'
        parsed_expr = sp.sympify(expression.equation)  # Parse the equation
        differentiated_expr = sp.diff(parsed_expr, x)  # Differentiate the equation
        
        # Generate steps
        steps = []
        steps.append(f"Original expression: {sp.pretty(parsed_expr, use_unicode=True)}")
        steps.append(f"Differentiated expression: {sp.pretty(differentiated_expr, use_unicode=True)}")
        
        # Generate LaTeX code
        latex_code = sp.latex(differentiated_expr)
        
        return {"result": str(differentiated_expr), "steps": steps, "latex": latex_code}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/integrate")
async def integrate_expression(expression: Expression):
    try:
        x = sp.symbols('x')  # Define variable 'x'
        parsed_expr = sp.sympify(expression.equation)  # Parse the equation
        integrated_expr = sp.integrate(parsed_expr, x)  # Integrate the equation
        steps = sp.pretty(integrated_expr, use_unicode=True)  # Generate steps
        return {"result": str(integrated_expr), "steps": steps}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/euclid/{n1}/{n2}")
async def euclids_lemma(n1,n2):
    try:
        print("hi")
        
        dividend=max(n1,n2)
        divisor=min(n1,n2)
        n1=max(n1,n2)
        n2=min(n1,n2)
        while dividend>=0:
            remainder=dividend%divisor
            quotient=dividend//divisor
            mul=divisor*quotient
            dividend=dividend-mul
            
        return {"a:n1","b=n2","q=quotient","r=remainder"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Trigonometry Calculator!"}