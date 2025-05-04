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


# Pydantic model for data validation
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
