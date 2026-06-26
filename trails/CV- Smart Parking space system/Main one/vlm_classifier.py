"""
VLM Parking Classifier — Qwen2-VL-2B-Instruct (accurate & lightweight)
-----------------------------------------------------------------------
Uses a 2 B-parameter Qwen vision-language model to count occupied vs vacant
parking spaces.  RAM requirement: ~4 GB (CPU fp32) vs ~2 GB (GPU fp16).

Classes:
  ParkingSlotClassifier  – per-slot binary (Occupied / Empty)
  ParkingLotCounter      – whole-image count (total / occupied / vacant)
"""
from __future__ import annotations

import os
import re

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image

# Maximum image dimension fed to the VLM (fewer visual tokens → less RAM)
_MAX_IMG_PX = 384


def _safe_image(image: Image.Image) -> Image.Image:
    """Resize so neither side exceeds _MAX_IMG_PX, preserving aspect ratio."""
    w, h = image.size
    if max(w, h) <= _MAX_IMG_PX:
        return image
    scale = _MAX_IMG_PX / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ── Model IDs ─────────────────────────────────────────────────────────────────
#  Qwen2-VL-2B: ~4 GB on CPU (fp32) / ~2 GB (GPU fp16) — accurate & compact
LOT_MODEL_ID  = "Qwen/Qwen2-VL-2B-Instruct"
SLOT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


# ── Prompts ───────────────────────────────────────────────────────────────────
SLOT_SYSTEM = (
    "You are a parking space occupancy sensor. "
    "You receive a cropped image of exactly ONE parking space. "
    "Answer with a single word only."
)

SLOT_PROMPT = (
    "Is there any part of a vehicle (car, truck, motorcycle, van) visible "
    "in this parking space? Answer with exactly one word: Occupied or Empty."
)

LOT_SYSTEM = (
    "You are an expert aerial parking lot analyst. "
    "You count parking spaces and vehicles with precision. "
    "Follow the user's format exactly."
)

LOT_PROMPT = """\
Carefully analyze this parking lot image.

STEP 1 — Find all parking spaces:
  Look for white/yellow painted lines on the ground that form rectangular spaces.
  Count every distinct parking space you can see.

STEP 2 — Check each space:
  For each space, decide: does it have a vehicle parked in it?

STEP 3 — Report results ONLY in this format (fill in the numbers):
Total: <number>
Occupied: <number>
Vacant: <number>

Do NOT include any other text. Only the three lines above."""


# ── Helper: extract counts from model text ────────────────────────────────────
def _parse_counts(text: str) -> tuple[int | None, int | None, int | None]:
    """
    Returns (total, occupied, vacant) from model output.
    Tries multiple strategies in order of preference.
    """
    # Strategy 1: Exact format  "Total: X\nOccupied: Y\nVacant: Z"
    m_total = re.search(r"[Tt]otal\s*[:\-]\s*(\d+)", text)
    m_occ   = re.search(r"[Oo]ccup\w*\s*[:\-]\s*(\d+)", text)
    m_vac   = re.search(r"[Vv]ac\w*\s*[:\-]\s*(\d+)", text)

    if m_total and m_occ and m_vac:
        total    = int(m_total.group(1))
        occupied = int(m_occ.group(1))
        vacant   = int(m_vac.group(1))
        if occupied + vacant != total:
            total = occupied + vacant
        return total, occupied, vacant

    if m_occ and m_vac:
        occupied = int(m_occ.group(1))
        vacant   = int(m_vac.group(1))
        return occupied + vacant, occupied, vacant

    if m_total and m_occ:
        total    = int(m_total.group(1))
        occupied = int(m_occ.group(1))
        return total, occupied, max(0, total - occupied)

    nums = re.findall(r"^\s*(\d+)\s*$", text, re.MULTILINE)
    if len(nums) >= 2:
        occupied = int(nums[0])
        vacant   = int(nums[1])
        return occupied + vacant, occupied, vacant

    return None, None, None


# ── Model loader ──────────────────────────────────────────────────────────────
def _load_vlm(model_id: str, requested_device: str) -> tuple:
    """
    Load Qwen2-VL-2B model + processor.

    Tries the best available model class:
      1. Qwen2VLForConditionalGeneration  (exact class, best option)
      2. AutoModelForVision2Seq           (transformers >= 4.39 generic)
      3. AutoModel                        (last-resort generic loader)
    """
    from transformers import AutoProcessor

    # ── Pick best available model class ──────────────────────────────────────
    ModelClass = None
    for _cls_name in (
        "Qwen2VLForConditionalGeneration",
        "AutoModelForVision2Seq",
        "AutoModel",
    ):
        try:
            import transformers as _tf
            ModelClass = getattr(_tf, _cls_name)
            print(f"[VLM] Using model class: {_cls_name}")
            break
        except AttributeError:
            continue

    if ModelClass is None:
        raise ImportError(
            "No suitable vision model class found in your transformers installation. "
            "Please run: pip install -U transformers"
        )

    device = requested_device
    dtype  = torch.float16 if device == "cuda" else torch.float32

    print(f"[VLM] Loading {model_id} on {device} ({dtype}) …")

    model = ModelClass.from_pretrained(
        model_id,
        torch_dtype=dtype,
        _attn_implementation="eager",   # avoids flash-attn requirement
        low_cpu_mem_usage=True,         # stream weights to reduce peak RAM
    )
    model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor, device


def _run_inference(model, processor, system: str, prompt: str,
                   image: Image.Image, device: str,
                   max_new_tokens: int = 80) -> str:
    """Build chat, run model, return only the generated answer text."""
    image = _safe_image(image)

    # SmolVLM2 uses the standard chat-template with image tokens
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": f"{system}\n\n{prompt}"},
            ],
        }
    ]

    # apply_chat_template + processor
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if device == "cuda":
        torch.cuda.empty_cache()

    inputs = processor(
        text=text_input,
        images=[image],
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_tokens = output_ids[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


# ── Public API ─────────────────────────────────────────────────────────────────
class ParkingSlotClassifier:
    """Per-slot binary classifier: Occupied / Empty."""

    def __init__(self, model_id: str = SLOT_MODEL_ID, device: str | None = None):
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor, self.device = _load_vlm(model_id, requested)

    def classify(self, slot_image: Image.Image) -> dict:
        raw = _run_inference(
            self.model, self.processor,
            SLOT_SYSTEM, SLOT_PROMPT,
            slot_image, self.device,
            max_new_tokens=8,
        )
        label = "Occupied" if "occup" in raw.lower() else "Empty"
        return {"label": label, "raw": raw}

    def count_occupancy(self, image: Image.Image) -> dict:
        """Whole-image count — delegates to ParkingLotCounter logic."""
        counter = ParkingLotCounter.__new__(ParkingLotCounter)
        counter.device    = self.device
        counter.model     = self.model
        counter.processor = self.processor
        return counter.count(image)


class ParkingLotCounter:
    """Whole-image parking lot occupancy counter."""

    def __init__(self, model_id: str = LOT_MODEL_ID, device: str | None = None):
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor, self.device = _load_vlm(model_id, requested)

    def count(self, image: Image.Image) -> dict:
        """
        Returns dict:
          total, occupied, vacant, occupancy_pct, raw
        """
        raw = _run_inference(
            self.model, self.processor,
            LOT_SYSTEM, LOT_PROMPT,
            image, self.device,
            max_new_tokens=80,
        )

        total, occupied, vacant = _parse_counts(raw)

        pct = round((occupied / total) * 100, 1) if total else None
        return {
            "total":        total,
            "occupied":     occupied,
            "vacant":       vacant,
            "occupancy_pct": pct,
            "raw":          raw,
        }