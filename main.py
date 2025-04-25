# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
import uuid
import os

app = FastAPI()

# Cargar modelo (una vez)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Modelo de entrada
class PromptInput(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"greeting": "Hello, World!", "message": "Stable Diffusion is ready!"}

@app.post("/generate")
async def generate_image(data: PromptInput):
    image = pipe(data.prompt).images[0]
    filename = f"{uuid.uuid4()}.png"
    image_path = f"/tmp/{filename}"
    image.save(image_path)

    # Devolver imagen como base64 o URL temporal (más adelante puedes subirla a un CDN)
    with open(image_path, "rb") as f:
        return {
            "filename": filename,
            "image_base64": f.read().encode("base64").decode("utf-8")  # para prueba rápida
        }
