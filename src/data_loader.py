# src/data_loader.py
import pandas as pd
import json

def load_csv(path, nrows=None):
    """Load CSV containing 'text' and 'tags' columns; parse tags into lists."""
    df = pd.read_csv(path, nrows=nrows)
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column")
    if 'tags' in df.columns:
        def parse_tags(x):
            try:
                if isinstance(x, str) and x.strip().startswith('['):
                    return json.loads(x)
                elif isinstance(x, str) and ',' in x:
                    return [t.strip() for t in x.split(',') if t.strip()]
                elif isinstance(x, str) and x.strip():
                    return [x.strip()]
                else:
                    return []
            except Exception:
                return []
        df['tags_list'] = df['tags'].apply(parse_tags)
    else:
        df['tags_list'] = [[] for _ in range(len(df))]
    print(f"Loaded {len(df)} rows from {path}")
    return df
