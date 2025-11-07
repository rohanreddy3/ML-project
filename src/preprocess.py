import re
import json
import pandas as pd

def basic_text_clean(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_tags_field(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return [t.lower() for t in val]
    s = str(val).strip()
    if s.startswith('[') and s.endswith(']'):
        try:
            tags = json.loads(s)
        except Exception:
            tags = [t.strip() for t in s.strip('[]').split(',') if t.strip()]
    else:
        tags = [t.strip() for t in s.split(',') if t.strip()]
    return [t.lower() for t in tags if t]
