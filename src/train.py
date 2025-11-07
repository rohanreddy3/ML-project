import os, argparse, traceback, pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_csv
from src.features import LSA_Tag_Generator
from src.evaluate import precision_at_k, recall_at_k, f1_at_k, average_precision_at_k, ndcg_at_k, exact_match_at_k
from src.modeling import save_model

def run_experiments(df, lsa_params_list, splits=[0.5,0.6,0.7,0.8], k_list=[1,3,5], output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    corpus = df['text_clean'].tolist()
    for lsa_params in lsa_params_list:
        print(f"Fitting LSA with params: {lsa_params}", flush=True)
        tagger = LSA_Tag_Generator(**lsa_params)
        tagger.fit(corpus)
        model_name = f"LSA_{lsa_params['n_components']}_{lsa_params['ngram_range'][0]}-{lsa_params['ngram_range'][1]}"
        save_model(tagger, os.path.join('models', f"{model_name}.joblib"))
        for split in splits:
            test_size = 1 - split
            strat = df['tags_list'].apply(lambda x: 1 if len(x)>0 else 0)
            X_train, X_test, y_train, y_test = train_test_split(
                df['text_clean'], df['tags_list'],
                test_size=test_size, random_state=42, stratify=strat
            )
            print(f"  Split {int(split*100)}-{int(test_size*100)}: train {len(X_train)} test {len(X_test)}", flush=True)
            for k in k_list:
                P=R=F=AP=NDCG=EM=0.0; n=0
                for text, true_tags in zip(X_test, y_test):
                    pred = tagger.generate_tags_for_doc(text, top_k=k)
                    P += precision_at_k(pred,true_tags,k)
                    R += recall_at_k(pred,true_tags,k)
                    F += f1_at_k(pred,true_tags,k)
                    AP += average_precision_at_k(pred,true_tags,k)
                    NDCG += ndcg_at_k(pred,true_tags,k)
                    EM += exact_match_at_k(pred,true_tags,k)
                    n+=1
                if n==0:
                    print("    WARNING: n_test == 0 for this split/k (skipping)", flush=True)
                    continue
                results.append({
                    'model': model_name,
                    'split': f"{int(split*100)}-{int(test_size*100)}",
                    'K': k,
                    'precision_at_k': P/n,
                    'recall_at_k': R/n,
                    'f1_at_k': F/n,
                    'map_at_k': AP/n,
                    'ndcg_at_k': NDCG/n,
                    'exact_match_at_k': EM/n,
                    'n_test': n
                })
    dfres = pd.DataFrame(results)
    out_path = os.path.join(output_dir, 'tagging_split_results.csv')
    dfres.to_csv(out_path, index=False)
    print("Saved results to", out_path, flush=True)

if __name__ == '__main__':
    try:
        print(">>> STARTING train.py", flush=True)
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', required=True, help='path to dataset csv')
        parser.add_argument('--output', default='results')
        parser.add_argument('--rows', type=int, default=None, help='(optional) read only first N rows')
        args = parser.parse_args()
        print("Args:", args, flush=True)

        df = load_csv(args.dataset, nrows=args.rows)
        if 'text_clean' not in df.columns:
            df['text_clean'] = df['text'].astype(str)
        if 'tags_list' not in df.columns:
            df['tags_list'] = df['tags'].apply(lambda s: [] if pd.isna(s) else (eval(s) if isinstance(s,str) and s.strip().startswith('[') else [s]))

        print("Loaded dataset rows:", len(df))
        print("Example row (first):", df[['text_clean','tags_list']].head(1).to_dict(orient='records'), flush=True)

        lsa_param_grid = [
            {'n_components':50, 'max_features':5000, 'ngram_range':(1,1), 'min_df':2},
            {'n_components':100, 'max_features':8000, 'ngram_range':(1,2), 'min_df':2}
        ]
        run_experiments(df, lsa_param_grid, output_dir=args.output)
        print(">>> FINISHED", flush=True)
    except Exception as e:
        print("ERROR during run:", str(e), flush=True)
        traceback.print_exc()
        raise
