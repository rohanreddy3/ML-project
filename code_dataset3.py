import pandas as pd
import os

# Step 1: Define input and output paths
QUESTIONS = "data/raw/Questions3.csv"
TAGS = "data/raw/Tags3.csv"
ANSWERS = "data/raw/Answers3.csv"
OUT = "data/raw/dataset3.csv"

# Step 2: Load the CSV files
print("🔹 Loading files...")
q = pd.read_csv(QUESTIONS, encoding='latin1')
t = pd.read_csv(TAGS, encoding='latin1')
a = pd.read_csv(ANSWERS, encoding='latin1')

# Step 3: Merge Questions and Tags using ID (if exists)
print("🔹 Merging datasets...")
if 'Id' in q.columns and 'Id' in t.columns:
    df = pd.merge(q, t, on='Id', how='inner')
else:
    df = pd.concat([q, t], axis=1)

# Step 4: Combine title + body + answer (if columns exist)
if 'Title' in df.columns and 'Body' in df.columns:
    df['text'] = df['Title'].fillna('') + ' ' + df['Body'].fillna('')
elif 'Body' in df.columns:
    df['text'] = df['Body'].fillna('')
else:
    df['text'] = df.iloc[:, 1].astype(str)  # fallback: 2nd column as text

# Step 5: Select relevant columns
if 'Tag' in df.columns:
    df = df[['text', 'Tag']]
else:
    df = df.iloc[:, :2]
    df.columns = ['text', 'tags']

# Step 6: Save output
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)
print(f"✅ Saved combined dataset to {OUT}")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Sample data:")
print(df.head(3))
