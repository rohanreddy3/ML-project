import argparse
import os
import re
import json
import pickle
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np

COMMON_TEXT_COLS = ['text','body','description','title','post','question']
COMMON_TAG_COLS = ['tags','tag','labels','label','categories']
METRIC_KEYS = [
    'precision@1','precision@3','precision@5','precision@10',
    'recall@1','recall@3','recall@5','recall@10',
    'P@1','P@3','P@5','MAP','map','mean_average_precision',
    'micro_f1','micro-f1','micro f1','microf1','f1_micro','microF1'
]
PICKLE_TFIDF_KEYWORDS = ['tfidf','vectorizer','tf-idf','tfidfvectorizer']
PICKLE_SVD_KEYWORDS = ['svd','lsa','truncatedsvd']

def find_files(repo_path: str, exts: List[str]=None) -> List[str]:
    matches = []
    for root, dirs, files in os.walk(repo_path):
        for f in files:
            if exts is None or any(f.lower().endswith(ext) for ext in exts):
                matches.append(os.path.join(root, f))
    return matches

def guess_text_and_tag_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    text_col = None
    tag_col = None
    cols = [c.lower() for c in df.columns]
    for cand in COMMON_TEXT_COLS:
        if cand in cols:
            text_col = df.columns[cols.index(cand)]
            break
    for cand in COMMON_TAG_COLS:
        if cand in cols:
            tag_col = df.columns[cols.index(cand)]
            break
    # if no explicit tag col, try to find a column with comma-separated tags heuristically
    if tag_col is None:
        for c in df.columns:
            sample = df[c].dropna().astype(str).head(50)
            if sample.apply(lambda s: ',' in s and 1 <= len(s.split(',')) <= 10).mean() > 0.6:
                tag_col = c
                break
    # text fallback: try to combine title+body if present
    if text_col is None:
        if 'title' in cols and 'body' in cols:
            text_col = df.columns[cols.index('body')]
    return text_col, tag_col

def analyze_dataset_csv(path: str) -> Dict:
    info = {'file': path, 'n_docs': None, 'avg_tokens': None, 'unique_tags': None,
            'train': None, 'val': None, 'test': None}
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        try:
            df = pd.read_csv(path, encoding='latin1', low_memory=False)
        except Exception as e2:
            return info
    n = len(df)
    info['n_docs'] = int(n)
    text_col, tag_col = guess_text_and_tag_columns(df)
    # tokens
    texts = None
    if text_col:
        texts = df[text_col].fillna('').astype(str)
    else:
        # find a reasonable text-like column by size of content
        text_like = max(df.columns, key=lambda c: df[c].astype(str).str.len().median())
        texts = df[text_like].fillna('').astype(str)
    token_counts = texts.apply(lambda s: len(re.findall(r'\w+', s)))
    info['avg_tokens'] = float(token_counts.mean())
    # tags
    if tag_col:
        tag_series = df[tag_col].fillna('').astype(str)
        # split common delimiters
        tags = tag_series.apply(lambda s: [t.strip().lower() for t in re.split(r'[,\|;]+', s) if t.strip()!=''])
        all_tags = [t for row in tags for t in row]
        info['unique_tags'] = int(len(set(all_tags)))
        # top tags (optional)
        info['top_tags'] = Counter(all_tags).most_common(10)
    else:
        info['unique_tags'] = None
    # detect simple splits
    split_cols = [c for c in df.columns if 'split' in c.lower() or 'set' in c.lower()]
    if split_cols:
        sc = split_cols[0]
        counts = df[sc].astype(str).value_counts().to_dict()
        info['train'] = counts.get('train') or counts.get('Train') or counts.get('TRAIN') or counts.get('training')
        info['val'] = counts.get('val') or counts.get('validation')
        info['test'] = counts.get('test')
    else:
        # try filenames like train.csv, test.csv in same folder
        folder = os.path.dirname(path)
        for name in ['train.csv','train.tsv','training.csv','val.csv','validation.csv','test.csv']:
            p = os.path.join(folder, name)
            if os.path.exists(p):
                try:
                    d = pd.read_csv(p, low_memory=False)
                    if 'train' in name:
                        info['train'] = len(d)
                    elif 'val' in name or 'validation' in name:
                        info['val'] = len(d)
                    elif 'test' in name:
                        info['test'] = len(d)
                except:
                    pass
    return info

def extract_metrics_from_csv(path: str) -> Dict:
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, encoding='latin1')
        except Exception:
            return {}
    # search for metric columns
    metrics = {}
    cols = [c.lower() for c in df.columns]
    for mk in METRIC_KEYS:
        for c in df.columns:
            cl = c.lower()
            if cl == mk.lower() or cl.replace(' ','') == mk.lower().replace(' ','') or mk.lower().replace('@','_at_') in cl or mk.lower().replace('@','') in cl:
                try:
                    val = float(df[c].iloc[-1]) if len(df[c].dropna())>0 else None
                    metrics[mk] = val
                except:
                    pass
    # fallback: try to read a single-row CSV with key,value
    if not metrics and len(df.columns)==2 and df.shape[0]>0:
        for _, row in df.iterrows():
            k = str(row[df.columns[0]]).strip().lower()
            v = row[df.columns[1]]
            for mk in METRIC_KEYS:
                if mk.replace('@','').replace('_','') in k.replace('@','').replace('_',''):
                    try:
                        metrics[mk] = float(v)
                    except:
                        pass
    return metrics

def extract_metrics_from_json(path: str) -> Dict:
    try:
        with open(path,'r') as f:
            j = json.load(f)
    except Exception:
        return {}
    metrics = {}
    def search(d, prefix=''):
        if isinstance(d, dict):
            for k,v in d.items():
                key = (prefix + '.' + k) if prefix else k
                if isinstance(v, (dict,list)):
                    search(v, key)
                else:
                    lk = k.lower()
                    for mk in METRIC_KEYS:
                        if mk.replace('@','').replace('_','') in lk.replace('@','').replace('_',''):
                            try:
                                metrics[mk] = float(v)
                            except:
                                pass
                    # also check full key
                    lk_full = key.lower()
                    for mk in METRIC_KEYS:
                        if mk.replace('@','').replace('_','') in lk_full.replace('@','').replace('_',''):
                            try:
                                metrics[mk] = float(v)
                            except:
                                pass
        elif isinstance(d, list):
            for item in d:
                search(item, prefix)
    search(j, '')
    return metrics

def inspect_pickle(path: str) -> Dict:
    info = {}
    try:
        with open(path,'rb') as f:
            obj = pickle.load(f)
    except Exception:
        return info
    # heuristics:
    oname = getattr(obj, '__class__', type(obj)).__name__.lower()
    if 'vectorizer' in oname or any(k in path.lower() for k in PICKLE_TFIDF_KEYWORDS) or 'tfidf' in oname:
        # try to get vocabulary
        vocab = None
        if hasattr(obj, 'vocabulary_'):
            vocab = getattr(obj, 'vocabulary_')
            info['tfidf_vocab_size'] = int(len(vocab))
        elif hasattr(obj, 'get_feature_names_out'):
            try:
                info['tfidf_vocab_size'] = int(len(obj.get_feature_names_out()))
            except:
                pass
    if 'truncatedsvd' in oname or 'svd' in oname or any(k in path.lower() for k in PICKLE_SVD_KEYWORDS):
        if hasattr(obj, 'n_components'):
            info['lsa_components'] = int(getattr(obj,'n_components'))
        elif hasattr(obj, 'components_'):
            info['lsa_components'] = int(getattr(obj,'components_').shape[0])
    # sometimes sklearn pipelines saved; attempt to inspect steps
    if 'pipeline' in oname:
        try:
            steps = obj.named_steps
            for name, step in steps.items():
                sname = getattr(step,'__class__', type(step)).__name__.lower()
                if 'vectorizer' in sname or 'tfidf' in sname:
                    if hasattr(step,'vocabulary_'):
                        info['tfidf_vocab_size'] = int(len(step.vocabulary_))
                if 'svd' in sname or 'truncatedsvd' in sname:
                    if hasattr(step,'n_components'):
                        info['lsa_components'] = int(step.n_components)
        except Exception:
            pass
    return info

def aggregate_findings(repo_path: str) -> Dict:
    findings = {'datasets': [], 'metrics_files': [], 'pickles': [], 'metrics': {}}
    # find candidate CSV dataset files
    csv_files = find_files(repo_path, exts=['.csv','.tsv'])
    # prefer files under data/ or data/processed/, data/raw/
    candidate_datasets = []
    for p in csv_files:
        pl = p.lower()
        if '/data/' in pl or '/dataset' in pl or 'stackoverflow' in pl or 'ecom' in pl or 'product' in pl:
            candidate_datasets.append(p)
    # fallback: top 10 largest csvs (by file size)
    if not candidate_datasets:
        csv_files_sorted = sorted(csv_files, key=lambda x: os.path.getsize(x), reverse=True)
        candidate_datasets = csv_files_sorted[:10]
    # analyze candidates
    for p in candidate_datasets:
        try:
            info = analyze_dataset_csv(p)
            findings['datasets'].append(info)
        except Exception as e:
            pass
    # find metrics files
    metric_candidates = []
    for root, dirs, files in os.walk(repo_path):
        for f in files:
            lf = f.lower()
            if any(x in lf for x in ['metric','metrics','result','results','evaluation','eval']):
                if lf.endswith('.csv') or lf.endswith('.json') or lf.endswith('.txt'):
                    metric_candidates.append(os.path.join(root,f))
    metric_candidates = sorted(set(metric_candidates))
    findings['metrics_files'] = metric_candidates
    # extract metrics
    metrics_agg = {}
    for p in metric_candidates:
        if p.lower().endswith('.csv') or p.lower().endswith('.tsv'):
            m = extract_metrics_from_csv(p)
        elif p.lower().endswith('.json'):
            m = extract_metrics_from_json(p)
        else:
            m = {}
        if m:
            metrics_agg[p] = m
            for k,v in m.items():
                metrics_agg[k] = v
    findings['metrics'] = metrics_agg
    # find pickles
    pickles = find_files(repo_path, exts=['.pkl','.pickle','.joblib'])
    pickles_info = {}
    for p in pickles:
        info = inspect_pickle(p)
        if info:
            pickles_info[p] = info
    findings['pickles'] = pickles_info
    return findings

def build_latex_table(findings: Dict, out_path: Optional[str]=None) -> str:
    # Build a LaTeX tabular with columns:
    # Dataset | #Docs | Train | Val | Test | AvgTokens | UniqueTags | TFIDF Vocab | LSA comps | P@3 | R@3 | MAP | Micro-F1
    rows = []
    headers = ['Dataset','#Docs','Train','Val','Test','AvgTokens','UniqueTags','TFIDF\_Vocab','LSA\_Comps',
               'P@3','R@3','MAP','Micro-F1']
    for d in findings.get('datasets', []):
        row = {}
        row['Dataset'] = os.path.basename(d.get('file',''))
        row['#Docs'] = d.get('n_docs') or 'N/A'
        row['Train'] = d.get('train') or 'N/A'
        row['Val'] = d.get('val') or 'N/A'
        row['Test'] = d.get('test') or 'N/A'
        row['AvgTokens'] = f"{d.get('avg_tokens'):.1f}" if d.get('avg_tokens') is not None else 'N/A'
        row['UniqueTags'] = d.get('unique_tags') or 'N/A'
        # attach TFIDF/LSA info if available from pickles (match by folder)
        folder = os.path.dirname(d.get('file',''))
        tfidf_vocab = 'N/A'
        lsa_comps = 'N/A'
        for p,info in findings.get('pickles', {}).items():
            if os.path.commonpath([os.path.abspath(folder), os.path.abspath(p)]) == os.path.abspath(folder) or os.path.commonpath([os.path.abspath(folder), os.path.abspath(p)]) == os.path.abspath(os.path.dirname(p)):
                if 'tfidf_vocab_size' in info:
                    tfidf_vocab = info['tfidf_vocab_size']
                if 'lsa_components' in info:
                    lsa_comps = info['lsa_components']
        # fallback: take any TFIDF/LSA info
        if tfidf_vocab == 'N/A' or lsa_comps == 'N/A':
            for p,info in findings.get('pickles', {}).items():
                if 'tfidf_vocab_size' in info and tfidf_vocab=='N/A':
                    tfidf_vocab = info['tfidf_vocab_size']
                if 'lsa_components' in info and lsa_comps=='N/A':
                    lsa_comps = info['lsa_components']
        row['TFIDF_Vocab'] = tfidf_vocab
        row['LSA_Comps'] = lsa_comps
        # metrics: try to locate P@3 etc in findings['metrics'] (choose global if dataset-specific not found)
        p_at_3 = 'N/A'
        r_at_3 = 'N/A'
        mapv = 'N/A'
        microf1 = 'N/A'
        # first check metrics files that include dataset name
        for mf, mvals in findings.get('metrics', {}).items():
            if isinstance(mvals, dict):
                for k in mvals:
                    if re.search(r'precision.*3|p@3|p_3', k.lower()):
                        p_at_3 = mvals.get(k, p_at_3)
                    if re.search(r'recall.*3|r@3|r_3', k.lower()):
                        r_at_3 = mvals.get(k, r_at_3)
                    if 'map' == k.lower() or 'mean_average_precision' in k.lower():
                        mapv = mvals.get(k, mapv)
                    if 'micro' in k.lower() and ('f1' in k.lower() or 'f-1' in k.lower()):
                        microf1 = mvals.get(k, microf1)
        # convert floats
        def fmt(x):
            if isinstance(x, (int, np.integer)):
                return str(int(x))
            try:
                return f"{float(x):.3f}"
            except:
                return str(x)
        row['P@3'] = fmt(p_at_3)
        row['R@3'] = fmt(r_at_3)
        row['MAP'] = fmt(mapv)
        row['Micro-F1'] = fmt(microf1)
        rows.append(row)
    # if no datasets found but metrics exist, create a single summary row
    if not rows and findings.get('metrics'):
        row = {'Dataset': 'Summary', '#Docs':'N/A','Train':'N/A','Val':'N/A','Test':'N/A','AvgTokens':'N/A','UniqueTags':'N/A',
               'TFIDF_Vocab':'N/A','LSA_Comps':'N/A','P@3':'N/A','R@3':'N/A','MAP':'N/A','Micro-F1':'N/A'}
        # aggregate from metrics
        for mf,mvals in findings['metrics'].items():
            for k,v in mvals.items():
                if re.search(r'precision.*3|p@3|p_3', k.lower()):
                    row['P@3'] = f"{v:.3f}" if isinstance(v,(int,float)) else v
                if re.search(r'recall.*3|r@3|r_3', k.lower()):
                    row['R@3'] = f"{v:.3f}" if isinstance(v,(int,float)) else v
                if 'map' == k.lower() or 'mean_average_precision' in k.lower():
                    row['MAP'] = f"{v:.3f}" if isinstance(v,(int,float)) else v
                if 'micro' in k.lower() and ('f1' in k.lower() or 'f-1' in k.lower()):
                    row['Micro-F1'] = f"{v:.3f}" if isinstance(v,(int,float)) else v
        rows = [row]

    # construct LaTeX table
    colspec = "lrrrrrrrrrrrr"
    header_line = " & ".join(headers) + " \\\\ \\midrule"
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Quantitative summary extracted from project}")
    lines.append("\\begin{tabular}{" + colspec + "}")
    lines.append("\\toprule")
    lines.append(header_line)
    for r in rows:
        vals = [str(r[h]) for h in headers]
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    latex = "\n".join(lines)
    if out_path:
        with open(out_path,'w') as f:
            f.write(latex)
    return latex

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX quantitative summary table from project repo")
    parser.add_argument('--repo-path', type=str, default='.', help='Path to repository root')
    parser.add_argument('--out', type=str, default=None, help='Optional output .tex file for the table')
    args = parser.parse_args()

    findings = aggregate_findings(args.repo_path)
    latex = build_latex_table(findings, out_path=args.out)
    print("=== LaTeX table ===")
    print(latex)
    if args.out:
        print(f"\nSaved LaTeX table to: {args.out}")

if __name__ == '__main__':
    main()
