import os
import random
import time

import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor

app = FastAPI()

# Define paths and device
model_save_dir = "model"  # Path to your saved model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens(
    {"pad_token": "[PAD]"}
)  # Ensure you add the same pad token

model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)

# Load the state dict with ignoring mismatched sizes
model.load_state_dict(
    torch.load(
        os.path.join(model_save_dir, "checkpoint_epoch_149_iter_155.pt"),
        map_location=device,
    ),
    strict=False,
)

model.to(device)
model.eval()  # Set the model to evaluation mode


def process_image(image_path):
    """
    Load and preprocess an image for the model.
    Args:
        image_path (str): Path to the image file.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values
    return image_tensor.to(device)


def generate_caption(image):
    """
    Generate a caption for the input image tensor.
    Args:
        image: path of downloaded image
    Returns:
        str: Generated caption.
    """
    image_tensor = process_image(image)
    with torch.no_grad():
        output = model.generate(pixel_values=image_tensor)
    caption = tokenizer.decode(output[0], skip_special_tokens=True)
    return caption


@app.post("/predict/")
async def predict(input_data: dict):
    file_path = input_data["file-path"]
    print(file_path)
    result = generate_caption(file_path)
    print(result)
    return {"result": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
