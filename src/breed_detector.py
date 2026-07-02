"""
Breed detection using pretrained MobileNetV2 on ImageNet.
ImageNet contains ~120 dog breeds (classes 151-268) and several cat types
(classes 281-285), so no additional training is needed — the pretrained
weights already know breeds from ImageNet pretraining.

Also provides image quality/property analysis via PIL.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import models, transforms

# ImageNet preprocessing (standard)
_imagenet_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ImageNet class index → readable breed name (cat & dog subsets only)
IMAGENET_PET_CLASSES: dict[int, str] = {
    # Cats
    281: "Tabby Cat",
    282: "Tiger Cat",
    283: "Persian Cat",
    284: "Siamese Cat",
    285: "Egyptian Cat",
    # Dogs — all 118 breed classes (151–268)
    151: "Chihuahua", 152: "Japanese Chin", 153: "Maltese",
    154: "Pekingese", 155: "Shih Tzu", 156: "Blenheim Spaniel",
    157: "Papillon", 158: "Toy Terrier", 159: "Rhodesian Ridgeback",
    160: "Afghan Hound", 161: "Basset Hound", 162: "Beagle",
    163: "Bloodhound", 164: "Bluetick Coonhound", 165: "Black & Tan Coonhound",
    166: "Walker Hound", 167: "English Foxhound", 168: "Redbone Coonhound",
    169: "Borzoi", 170: "Irish Wolfhound", 171: "Italian Greyhound",
    172: "Whippet", 173: "Ibizan Hound", 174: "Norwegian Elkhound",
    175: "Otterhound", 176: "Saluki", 177: "Scottish Deerhound",
    178: "Weimaraner", 179: "Staffordshire Bull Terrier",
    180: "American Staffordshire Terrier", 181: "Bedlington Terrier",
    182: "Border Terrier", 183: "Kerry Blue Terrier", 184: "Irish Terrier",
    185: "Norfolk Terrier", 186: "Norwich Terrier", 187: "Yorkshire Terrier",
    188: "Wire Fox Terrier", 189: "Lakeland Terrier", 190: "Sealyham Terrier",
    191: "Airedale Terrier", 192: "Cairn Terrier", 193: "Australian Terrier",
    194: "Dandie Dinmont Terrier", 195: "Boston Terrier",
    196: "Miniature Schnauzer", 197: "Giant Schnauzer",
    198: "Standard Schnauzer", 199: "Scottish Terrier",
    200: "Tibetan Terrier", 201: "Silky Terrier",
    202: "Soft-coated Wheaten Terrier", 203: "West Highland White Terrier",
    204: "Lhasa Apso", 205: "Flat-coated Retriever",
    206: "Curly-coated Retriever", 207: "Golden Retriever",
    208: "Labrador Retriever", 209: "Chesapeake Bay Retriever",
    210: "German Shorthaired Pointer", 211: "Vizsla",
    212: "English Setter", 213: "Irish Setter", 214: "Gordon Setter",
    215: "Brittany Spaniel", 216: "Clumber Spaniel",
    217: "English Springer Spaniel", 218: "Welsh Springer Spaniel",
    219: "Cocker Spaniel", 220: "Sussex Spaniel", 221: "Irish Water Spaniel",
    222: "Kuvasz", 223: "Schipperke", 224: "Belgian Groenendael",
    225: "Belgian Malinois", 226: "Briard", 227: "Australian Kelpie",
    228: "Komondor", 229: "Old English Sheepdog",
    230: "Shetland Sheepdog", 231: "Collie", 232: "Border Collie",
    233: "Bouvier des Flandres", 234: "Rottweiler",
    235: "German Shepherd", 236: "Doberman Pinscher",
    237: "Miniature Pinscher", 238: "Greater Swiss Mountain Dog",
    239: "Bernese Mountain Dog", 240: "Appenzeller",
    241: "Entlebucher Mountain Dog", 242: "Boxer", 243: "Bullmastiff",
    244: "Tibetan Mastiff", 245: "French Bulldog", 246: "Great Dane",
    247: "Saint Bernard", 248: "Husky (Eskimo Dog)", 249: "Malamute",
    250: "Siberian Husky", 251: "Dalmatian", 252: "Affenpinscher",
    253: "Basenji", 254: "Pug", 255: "Leonberger", 256: "Newfoundland",
    257: "Great Pyrenees", 258: "Samoyed", 259: "Pomeranian",
    260: "Chow Chow", 261: "Keeshond", 262: "Griffon Bruxellois",
    263: "Pembroke Welsh Corgi", 264: "Cardigan Welsh Corgi",
    265: "Toy Poodle", 266: "Miniature Poodle", 267: "Standard Poodle",
    268: "Mexican Hairless",
}

CAT_INDICES = {281, 282, 283, 284, 285}
DOG_INDICES = set(range(151, 269))
PET_INDICES = CAT_INDICES | DOG_INDICES

_breed_model = None


def _get_breed_model(device: str = "cpu"):
    """Lazy-load the pretrained ImageNet model (shared across requests)."""
    global _breed_model
    if _breed_model is None:
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.to(device).eval()
        _breed_model = m
    return _breed_model


def detect_breeds(pil_image: Image.Image, predicted_class: str, device: str = "cpu", top_k: int = 3):
    """
    Returns top_k breed suggestions using ImageNet pretrained MobileNetV2.
    Filters results to match the predicted class (cats → cat classes, dogs → dog classes).
    """
    model = _get_breed_model(device)
    tensor = _imagenet_transforms(pil_image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    relevant_indices = CAT_INDICES if predicted_class == "cat" else DOG_INDICES

    breed_scores = [
        (IMAGENET_PET_CLASSES[idx], probs[idx].item())
        for idx in relevant_indices
    ]
    breed_scores.sort(key=lambda x: x[1], reverse=True)

    total = sum(s for _, s in breed_scores[:top_k]) or 1.0
    return [
        {
            "breed": name,
            "probability": round(score, 4),
            "relative_pct": round(score / total * 100, 1),
        }
        for name, score in breed_scores[:top_k]
    ]


def analyze_image(pil_image: Image.Image) -> dict:
    """
    Extracts visual properties from the image:
    brightness, contrast, sharpness, dominant color, dimensions, aspect ratio.
    """
    img_rgb   = pil_image.convert("RGB")
    img_gray  = pil_image.convert("L")
    arr       = np.array(img_rgb).astype(np.float32)
    arr_gray  = np.array(img_gray).astype(np.float32)

    # Brightness: mean luminance (0–1)
    brightness = float(arr_gray.mean() / 255)

    # Contrast: std of luminance (0–1, higher = more contrast)
    contrast = float(min(arr_gray.std() / 127.0, 1.0))

    # Sharpness: Laplacian variance (higher = sharper)
    edges     = img_gray.filter(ImageFilter.FIND_EDGES)
    sharpness = float(min(np.array(edges).astype(np.float32).std() / 60.0, 1.0))

    # Dominant color: average RGB clamped to hex
    avg_r = int(np.clip(arr[:, :, 0].mean(), 0, 255))
    avg_g = int(np.clip(arr[:, :, 1].mean(), 0, 255))
    avg_b = int(np.clip(arr[:, :, 2].mean(), 0, 255))
    dominant_color = f"#{avg_r:02x}{avg_g:02x}{avg_b:02x}"

    w, h = pil_image.size
    gcd  = _gcd(w, h)
    aspect = f"{w // gcd}:{h // gcd}"

    quality = "High" if min(w, h) >= 512 else ("Medium" if min(w, h) >= 224 else "Low")

    return {
        "width":           w,
        "height":          h,
        "aspect_ratio":    aspect,
        "quality":         quality,
        "brightness":      round(brightness, 3),
        "contrast":        round(contrast, 3),
        "sharpness":       round(sharpness, 3),
        "dominant_color":  dominant_color,
        "color_r":         avg_r,
        "color_g":         avg_g,
        "color_b":         avg_b,
    }


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a