#!/usr/bin/env python3
"""FastAPI inference service for chest X-ray classification.

Usage:
    # Start server with default model (ViT)
    python scripts/serve.py
    
    # Start with specific model
    MODEL_TYPE=mamba python scripts/serve.py
    
    # Specify host and port
    python scripts/serve.py --host 0.0.0.0 --port 8080
    
API Endpoints:
    POST /predict          - Predict single image (multipart/form-data)
    POST /predict/batch    - Predict multiple images
    GET  /health           - Health check
    GET  /docs             - Swagger UI documentation
"""
import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/ChestXray-classification')

import io
import argparse
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
from pydantic import BaseModel

from config import config
from models import create_model
from utils import test_transform, Predictor


# Response models
class PredictionResponse(BaseModel):
    success: bool
    label: str
    confidence: float
    probabilities: dict
    model_type: str


class BatchPredictionResponse(BaseModel):
    success: bool
    predictions: List[dict]
    model_type: str


class HealthResponse(BaseModel):
    status: str
    model_type: str
    device: str
    num_classes: int


# Global predictor
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global predictor
    
    print("Loading model...")
    model = create_model()
    
    # Try to load checkpoint
    checkpoint_path = f"checkpoints/best_{config.model.type}.pth"
    try:
        import os
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=config.device.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}, using random weights")
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
    
    predictor = Predictor(model)
    print(f"Model loaded: {config.model.type}")
    print(f"Device: {config.device.device}")
    
    yield
    
    # Cleanup
    print("Shutting down...")


app = FastAPI(
    title="Chest X-ray Classification API",
    description="AI-powered pneumonia classification using ViT or Vision Mamba",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_type=config.model.type,
        device=config.device.device,
        num_classes=config.model.num_classes
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict pneumonia class from a chest X-ray image.
    
    - **file**: Chest X-ray image (JPG, PNG)
    
    Returns prediction with confidence scores for all classes.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        image_tensor = test_transform(image)
        
        # Predict
        result = predictor.predict(image_tensor)
        
        return PredictionResponse(
            success=True,
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_type=config.model.type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict pneumonia classes from multiple chest X-ray images.
    
    - **files**: List of chest X-ray images
    
    Returns predictions for all images.
    """
    if len(files) > 32:
        raise HTTPException(status_code=400, detail="Maximum 32 images per batch")
    
    predictions = []
    
    for file in files:
        if not file.content_type.startswith("image/"):
            predictions.append({
                "filename": file.filename,
                "error": "File must be an image"
            })
            continue
        
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            image_tensor = test_transform(image)
            result = predictor.predict(image_tensor)
            
            predictions.append({
                "filename": file.filename,
                "label": result["label"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"]
            })
            
        except Exception as e:
            predictions.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return BatchPredictionResponse(
        success=True,
        predictions=predictions,
        model_type=config.model.type
    )


@app.get("/")
async def root():
    """API information."""
    return {
        "name": "Chest X-ray Classification API",
        "version": "2.0.0",
        "model": config.model.type,
        "docs": "/docs",
        "health": "/health"
    }


def main():
    parser = argparse.ArgumentParser(description="Start inference server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Chest X-ray Classification API Server")
    print("=" * 60)
    print(f"Model: {config.model.type}")
    print(f"Device: {config.device.device}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print("=" * 60)
    print()
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
