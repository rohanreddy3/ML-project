# prepare_stackoverflow_dataset_aggregate.py
import pandas as pd, os, json

QUESTIONS = "data/raw/Questions.csv"
TAGS = "data/raw/Tags.csv"
OUT = "data/raw/dataset1.csv"

def main():
    q = pd.read_csv(QUESTIONS, usecols=['Id','Title','Body'], encoding='latin1')
    t = pd.read_csv(TAGS, usecols=['Id','Tag'], encoding='latin1')
    tag_grp = t.groupby('Id')['Tag'].apply(list).reset_index(name='tags_list')
    df = pd.merge(q, tag_grp, on='Id', how='left')
    df['tags_list'] = df['tags_list'].apply(lambda x: x if isinstance(x,list) else [])
    df['text'] = df['Title'].fillna('') + " " + df['Body'].fillna('')
    out = df[['Id','text','tags_list']].copy()
    out['tags'] = out['tags_list'].apply(lambda x: json.dumps(x))
    out = out[['text','tags']]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    out.head(20000).to_csv('data/raw/dataset1_sample.csv', index=False)
    print("Saved aggregated dataset and sample")

if __name__ == "__main__":
    main()
