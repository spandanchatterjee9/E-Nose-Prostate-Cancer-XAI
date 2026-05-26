import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.database import engine, Base, SessionLocal
from app.api.endpoints import auth, patients, predict, history, metrics
from app.crud import crud
from app.schemas import schemas

# 1. Initialize Database Tables
print("Initializing database tables...")
Base.metadata.create_all(bind=engine)

# 2. Seed Default User if database is empty
db = SessionLocal()
try:
    default_user = crud.get_user_by_username(db, "admin")
    if not default_user:
        print("Database is empty. Seeding default clinician credentials (admin / admin123)...")
        crud.create_user(db, schemas.UserCreate(
            username="admin",
            password="admin123",
            role="clinician"
        ))
finally:
    db.close()

# 3. Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Explainable AI E-Nose clinical decision support API for Prostate Cancer classification.",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# 4. Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Include API Routers
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["Authentication"])
app.include_router(patients.router, prefix=f"{settings.API_V1_STR}/patients", tags=["Patients"])
app.include_router(predict.router, prefix=f"{settings.API_V1_STR}/predict", tags=["Predictions"])
app.include_router(history.router, prefix=f"{settings.API_V1_STR}/history", tags=["History"])
app.include_router(metrics.router, prefix=f"{settings.API_V1_STR}/metrics", tags=["Metrics"])

@app.get("/")
def read_root():
    return {
        "message": "Welcome to E-Nose Prostate Cancer Prediction API",
        "docs_url": "/docs",
        "version": "1.0.0"
    }
