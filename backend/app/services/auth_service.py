from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.schemas.auth import UserRegister, UserLogin
from app.utils.security import get_password_hash, verify_password, create_access_token, decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def register_user(db: Session, user_data: UserRegister) -> User:
    # Check existing user
    email_clean = user_data.email.strip().lower()
    username_clean = user_data.username.strip()
    if db.query(User).filter(User.email.ilike(email_clean)).first():
        raise HTTPException(status_code=400, detail="Email is already registered.")
    if db.query(User).filter(User.username.ilike(username_clean)).first():
        raise HTTPException(status_code=400, detail="Username is already taken.")

    hashed_pwd = get_password_hash(user_data.password)
    user = User(
        email=email_clean,
        username=username_clean,
        hashed_password=hashed_pwd
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def authenticate_user(db: Session, credentials: UserLogin) -> dict:
    identifier = (credentials.username_or_email or "").strip()
    user = db.query(User).filter(
        (User.username == identifier) | (User.email.ilike(identifier))
    ).first()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password."
        )

    token = create_access_token(data={"sub": user.id, "username": user.username})
    return {"access_token": token, "token_type": "bearer", "user": user}

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token."
        )
    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return user
