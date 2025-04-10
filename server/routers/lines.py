from fastapi import FastAPI, Query, APIRouter
from models.shapes import SlopeCordiantes
from sympy import symbols, Eq, solve, simplify, parse_expr
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application


router = APIRouter()


@router.post("/SlopeCordiantes")
async def slopecordinates(slopecordinates: SlopeCordiantes):
    try:
        x1 = slopecordinates.x1
        y1 = slopecordinates.y1
        x2 = slopecordinates.x2
        y2 = slopecordinates.y2
        m=(y2-y1)/(x2-x1)
        return {"slope": m}
    except Exception as e:
        return {"error": str(e)}
     
    
    
    
  