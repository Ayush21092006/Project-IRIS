import json
import os
from datetime import datetime

LOG_FILE = "history/detection_log.json"

# Ensure folder exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def load_history():
    if not os.path.exists(LOG_FILE):
        return {}
    with open(LOG_FILE, "r") as f:
        return json.load(f)

def save_to_history(objects=None, outfits=None, texts=None):
    data = load_history()
    timestamp = datetime.now().isoformat()

    data[timestamp] = {
        "objects": list(set(objects)) if objects else [],
        "outfits": list(set(outfits)) if outfits else [],
        "texts": list(set(texts)) if texts else []
    }

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)

def get_recent_detections(n=5):
    history = load_history()
    return dict(sorted(history.items(), reverse=True)[:n])
