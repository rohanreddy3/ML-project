import pandas as pd, os
p = 'data/raw/ecommerce_text.csv'
out = 'data/raw/ecommerce_text_fixed.csv'
os.makedirs('data/raw', exist_ok=True)

# load without header
df = pd.read_csv(p, header=None, dtype=str, keep_default_na=False, engine='python')
# first column = tag
tags = df.iloc[:,0].astype(str).str.strip()
# join rest into text (handles rows split into multiple columns)
texts = df.iloc[:,1:].astype(str).apply(lambda r: ','.join([c for c in r if c and c!='nan']), axis=1)
fixed = pd.DataFrame({'text': texts, 'tags': tags})
fixed.to_csv(out, index=False)
print("WROTE:", out)
print(fixed.head(3))
