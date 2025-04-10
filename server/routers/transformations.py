from fastapi import FastAPI, Query, APIRouter
from models.shapes import Original_axes, New_axes
from sympy import symbols, Eq, solve, simplify, parse_expr
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
)


router = APIRouter()


@router.get("/transform_to_new")
async def transformations(axes: Original_axes):
    try:
        X = axes.x - axes.h
        Y = axes.y - axes.k
        return {"(X,Y)": f"({X},{Y})", "(h,k)": f"({axes.h},{axes.k})"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/transform_to_original")
async def transformations(axes: New_axes):
    try:
        x = axes.X + axes.h
        y = axes.Y + axes.k
        return {"x": x, "y": y, "h": axes.h, "k": axes.k}
    except Exception as e:
        return {"error": str(e)}


# @router.get("/transform_equation"):
# async
