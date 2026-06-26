"""FR-06: Occupancy classification via Qwen2.5-VL. Falls back to Qwen2.5-VL-2B if 3B OOMs."""
import re
import json
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

PROMPT = (
    "You are looking at a single cropped parking space. "
    "Reply with exactly one word: 'Occupied' if any part of a vehicle is visible in the space, "
    "otherwise 'Empty'. Answer:"
)

COUNT_PROMPT = (
    "Look at this parking lot image. Count how many parking spaces are occupied "
    "(a vehicle is parked there) and how many are vacant (empty, no vehicle). "
    "Respond ONLY with compact JSON, nothing else, in exactly this format: "
    '{"occupied": <integer>, "vacant": <integer>}'
)


class ParkingSlotClassifier:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

        # Small pixel budget keeps each slot crop fast and under ~4GB VRAM (NFR target).
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
        )

    @torch.inference_mode()
    def classify(self, slot_image: Image.Image) -> dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": slot_image},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[slot_image], padding=True, return_tensors="pt"
        ).to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        new_tokens = output_ids[:, inputs.input_ids.shape[1]:]
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        label = "Occupied" if "occup" in decoded.lower() else "Empty"
        return {"label": label, "raw": decoded}

    @torch.inference_mode()
    def count_occupancy(self, image: Image.Image) -> dict:
        """Whole-image quick mode: no slot boxes needed, model counts directly."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": COUNT_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        new_tokens = output_ids[:, inputs.input_ids.shape[1]:]
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        match = re.search(r"\{.*\}", decoded, re.DOTALL)
        occupied, vacant = None, None
        if match:
            try:
                parsed = json.loads(match.group(0))
                occupied = int(parsed.get("occupied"))
                vacant = int(parsed.get("vacant"))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        total = (occupied + vacant) if occupied is not None and vacant is not None else None
        pct = round((occupied / total) * 100, 1) if total else None
        return {
            "occupied": occupied,
            "vacant": vacant,
            "total": total,
            "occupancy_pct": pct,
            "raw": decoded,
        }
