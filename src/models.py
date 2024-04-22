from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi.openapi.models import Response


class QueryModel(BaseModel):
    datasources: List[str]
    most_similar_section: Optional[bool] = False
    filters: Optional[Dict[str, Any]] = None
