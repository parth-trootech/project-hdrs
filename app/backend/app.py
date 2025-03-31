import logging
import os
import re
import uuid

from PIL import Image
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from passlib.context import CryptContext
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from app.backend.schemas import PredictionRequest
from app.backend.schemas import UserCreate, UserLogin
from app.db.models import User, ImageUpload, PredictionResult
# Ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

from app.config import Config

# Database configuration
engine = create_engine(url=Config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize FastAPI
app = FastAPI()


# Dependency to get the DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Utility functions for password hashing
def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# User Signup
@app.post("/signup")
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.user_email == user.user_email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = hash_password(user.user_password)
    new_user = User(user_email=user.user_email, user_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}


# User Login
@app.post("/login")
async def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.user_email == user.user_email).first()
    if not db_user or not verify_password(user.user_password, db_user.user_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    return {"message": "Login successful", "user_id": db_user.user_id, "user_email": db_user.user_email}


# Image Upload
@app.post("/upload_image")
async def upload_image(
        user_id: int = Form(...),
        image: UploadFile = File(...),
        db: Session = Depends(get_db)
):
    try:
        filename = f"{uuid.uuid4().hex}_{image.filename}"
        file_location = os.path.join(UPLOAD_DIR, filename)

        with open(file_location, "wb") as buffer:
            buffer.write(await image.read())

        image_upload = ImageUpload(user_id=user_id, image_path=file_location)
        db.add(image_upload)
        db.commit()
        db.refresh(image_upload)

        return {"message": "Image uploaded successfully", "image_id": image_upload.image_id}

    except Exception as e:
        logging.error(f"Error uploading image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")


# Predict
@app.post("/predict")
async def predict(request: PredictionRequest, db: Session = Depends(get_db)):
    image_upload = db.query(ImageUpload).filter(ImageUpload.image_id == request.image_id).first()
    if not image_upload:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = image_upload.image_path
    image = Image.open(image_path).convert("RGB")

    processor = TrOCRProcessor.from_pretrained('ml_model')
    model = VisionEncoderDecoderModel.from_pretrained('ml_model')
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    generated_text = ''.join(re.findall(r'\d', generated_text))

    prediction_result = PredictionResult(
        image_id=request.image_id,
        predicted_digit=int(generated_text) if generated_text.isdigit() else -1,
        confidence_score=None
    )
    db.add(prediction_result)
    db.commit()
    db.refresh(prediction_result)

    return {"predicted_digit": generated_text, "prediction_id": prediction_result.prediction_id}


