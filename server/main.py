from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from routers import transformations
from routers import lines
from routers import complexnums
from routers import circles

from routers import matrices
from routers import finelimits
from routers import derivatives
from routers import integration
from routers import prog
from routers import derivatiesApply
from routers.derivatiesApply import router_continuity
from routers.derivatiesApply import router_differentiability
from routers.derivatiesApply import router_rate_measure
from routers.derivatiesApply import router_approximations
from routers.derivatiesApply import router_tangents_normals
from routers.derivatiesApply import router_monotonicity
from routers import real_numbers_router
from routers import linear_equations_router
from routers import polynomial_quadratic_router

app = FastAPI()

app.include_router(transformations.router)
app.include_router(lines.router)
app.include_router(complexnums.router)
app.include_router(circles.router)

app.include_router(matrices.router)
app.include_router(finelimits.router)
app.include_router(derivatives.router)
app.include_router(integration.router)
app.include_router(prog.router)
app.include_router(derivatiesApply.router_continuity)

app.include_router(router_continuity)
app.include_router(router_differentiability)
app.include_router(router_rate_measure)
app.include_router(router_approximations)
app.include_router(router_tangents_normals)
app.include_router(router_monotonicity)

app.include_router(real_numbers_router.router)
app.include_router(linear_equations_router.router)
app.include_router(polynomial_quadratic_router.router)


# Pydantic model for data validation
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
