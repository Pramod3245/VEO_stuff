from fastapi import FastAPI
from endpoints.script_genrate import router as script_router

app = FastAPI(title="Content Studio V2")

app.include_router(script_router, prefix="/api/v1", tags=["Script Generation"])
