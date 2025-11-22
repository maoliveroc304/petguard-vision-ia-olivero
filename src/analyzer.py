# src/analyzer.py
import os
import json
import uuid
from datetime import datetime
import torch
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from transformers import pipeline, TrOCRProcessor, VisionEncoderDecoderModel

# Ruta por defecto en el repo
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "carnet_animal.jpg")
IMAGE_PATH = os.path.normpath(IMAGE_PATH)

DEVICE = 0 if torch.cuda.is_available() else -1

# --- BLIP description
def describe_blip(image_path=IMAGE_PATH, model_name="Salesforce/blip-image-captioning-base"):
    pipe = pipeline("image-to-text", model=model_name, device=DEVICE)
    out = pipe(image_path, max_length=64, num_beams=3)
    return out[0]["generated_text"] if isinstance(out, list) else str(out)

# --- ViT classify (heurística especie)
def classify_vit(image_path=IMAGE_PATH, model_name="google/vit-base-patch16-224", top_k=5):
    clf = pipeline("image-classification", model=model_name, device=DEVICE, top_k=top_k)
    preds = clf(image_path)
    preds_clean = [{"label": p["label"], "score": float(p["score"])} for p in preds]
    animal_keywords = ["dog","cat","puppy","kitten","horse","sheep","cow","bird","rabbit","fish","wolf","fox","hamster","bull","chicken"]
    suggested = None
    for p in preds_clean:
        lbl = p["label"].lower()
        for kw in animal_keywords:
            if kw in lbl:
                suggested = {"species": kw, "matched_label": p["label"], "score": p["score"]}
                break
        if suggested:
            break
    return {"predictions": preds_clean, "suggested_species": suggested}

# --- TrOCR OCR (texto impreso)
def ocr_trocr(image_path=IMAGE_PATH, model_name="microsoft/trocr-base-printed"):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    if DEVICE >= 0:
        model.to(torch.device("cuda"))
    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    if DEVICE >= 0:
        pixel_values = pixel_values.to(torch.device("cuda"))
    generated_ids = model.generate(pixel_values, max_length=512)
    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return preds[0] if preds else ""

# --- Colores dominantes (KMeans)
def dominant_colors(image_path=IMAGE_PATH, n_colors=5, resize=(200,200)):
    img = Image.open(image_path).convert("RGB")
    img.thumbnail(resize)
    arr = np.asarray(img).reshape(-1, 3).astype(float)
    if arr.shape[0] < n_colors:
        n_colors = max(1, arr.shape[0]//10)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(arr)
    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    pct = counts / counts.sum()
    def rgb2hex(rgb): return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    res = [{"hex": rgb2hex(c), "rgb": c.tolist(), "pct": float(p)} for c,p in zip(centers, pct)]
    return sorted(res, key=lambda x: x["pct"], reverse=True)

# --- Pipeline que produce JSON clínico estandar
def produce_clinical_json(image_path=IMAGE_PATH):
    timestamp = datetime.utcnow().isoformat() + "Z"
    uid = str(uuid.uuid4())
    desc = describe_blip(image_path)
    cls = classify_vit(image_path)
    text = ocr_trocr(image_path)
    colors = dominant_colors(image_path)

    clinical = {
        "record_id": uid,
        "timestamp_utc": timestamp,
        "source_image": os.path.basename(image_path),
        "models": {
            "blip": {"name":"Salesforce/blip-image-captioning-base"},
            "vit": {"name":"google/vit-base-patch16-224"},
            "trocr": {"name":"microsoft/trocr-base-printed"},
            "colors": {"method":"kmeans", "n_colors": len(colors)}
        },
        "analysis": {
            "description": desc,
            "classification": cls["predictions"],
            "inferred_species": cls["suggested_species"],
            "detected_text": text,
            "dominant_colors": colors
        }
    }
    return clinical

# --- si se ejecuta como script, producir archivo JSON en clinical_examples/
if __name__ == "__main__":
    out = produce_clinical_json()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "clinical_examples")
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{out['record_id']}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("JSON clínico guardado en:", fname)
