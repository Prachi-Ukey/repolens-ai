from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.auth import UserRegister, UserLogin, Token, UserResponse
from app.services.auth_service import register_user, authenticate_user, get_current_user
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    return register_user(db, user_data)

@router.post("/login", response_model=Token)
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    auth_data = authenticate_user(db, credentials)
    return Token(access_token=auth_data["access_token"], token_type="bearer")

@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)):
    return current_user

@router.get("/demo-token", response_model=Token)
def demo_token(db: Session = Depends(get_db)):
    user = db.query(User).first()
    if not user:
        from app.utils.security import get_password_hash
        user = User(
            id="default-demo-user",
            username="developer",
            email="developer@repolens.local",
            hashed_password=get_password_hash("password123")
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    from app.utils.security import create_access_token
    token = create_access_token(data={"sub": user.id, "username": user.username})
    return Token(access_token=token, token_type="bearer")
