import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "E-Nose Prostate Cancer Predictor API"
    API_V1_STR: str = "/api/v1"
    
    # JWT Authentication settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "b38d3ad48e9a2637f90f23821034fe2da5e4860b0e5015b6305a415ff6c927bb")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 1 week
    
    # Database Settings: Dynamic path for local SQLite or environment variable
    _base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _db_path = os.path.join(_base_dir, 'enose.db').replace('\\\\', '/').replace('\\', '/')
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        f"sqlite:///{_db_path}"
    )
    
    # CORS Origins
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]

    class Config:
        case_sensitive = True

settings = Settings()
