# scripts/plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('results/dataset1_results/tagging_split_results.csv')

# example: plot F1 vs K for each model & split
for model, g in df.groupby('model'):
    plt.figure(figsize=(6,4))
    for split, sub in g.groupby('split'):
        plt.plot(sub['K'], sub['f1_at_k'], marker='o', label=split)
    plt.title(f'F1 vs K — {model}')
    plt.xlabel('K')
    plt.ylabel('F1@K')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/{model}_f1_vs_k.png')
    print('Saved results/'+f'{model}_f1_vs_k.png')
