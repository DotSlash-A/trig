from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from routers import transformations

app = FastAPI()

app.include_router(transformations.router)

# Pydantic model for data validation
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
