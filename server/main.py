from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from routers import transformations
from routers import lines

app = FastAPI()

app.include_router(transformations.router)
app.include_router(lines.router)

# Pydantic model for data validation
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
