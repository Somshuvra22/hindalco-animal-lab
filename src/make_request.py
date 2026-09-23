import argparse
import base64
import io
import json
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--out", default="sample-request.json")
args = parser.parse_args()
with Image.open(args.image) as image:
    image = image.convert("RGB")
    image.thumbnail((512, 512))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
body = {"image": base64.b64encode(buffer.getvalue()).decode("ascii")}
with open(args.out, "w", encoding="utf-8") as file:
    json.dump(body, file)
print(f"Wrote {args.out} ({len(buffer.getvalue())} JPEG bytes)")