"""Generate raw CSV from TEST SET metrics (phase='test' or test_ar_* columns)."""
import os, csv

LOG_DIR = 'logs'
SEEDS = ['42', '123', '618', '2024', '9999']
ALL_MODELS = [
    'FA-LSTM', 'FA-LSTM_A1', 'FA-LSTM_A2', 'FA-LSTM_A3',
    'FA-LSTM_A4', 'FA-LSTM_A5', 'FA-LSTM_A6', 'FA-LSTM_A8',
    'LSTM', 'LSTM_AR', 'GRU', 'TCN', 'Transformer', 'DA-RNN'
]

def detect(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return 'falstm' if ('phase' in f.readline() and 'phys_r2' in f.readline()) else 'baseline'

def extract_falstm(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        for r in csv.DictReader(f):
            if r.get('phase','') == 'test':
                g = lambda k: float(r.get(k,0))
                return {
                    'seq_mse': g('phys_mse'), 'seq_rmse': g('phys_rmse'), 'seq_mae': g('phys_mae'), 'seq_r2': g('phys_r2'),
                    'buck_load_mae': g('buckling_load_mae'), 'buck_load_rmse': g('buckling_load_rmse'),
                    'buck_load_r2': g('buckling_load_r2'), 'buck_load_mape': g('buckling_load_mape'),
                    'buck_load_nrmse': g('buckling_load_nrmse'), 'buck_load_pcc': g('buckling_load_pcc'),
                    'buck_disp_mae': g('buckling_disp_mae'), 'buck_disp_rmse': g('buckling_disp_rmse'),
                }
    return None

def extract_baseline(path):
    last = None
    with open(path, encoding='utf-8', errors='replace') as f:
        for r in csv.DictReader(f): last = r
    if not last: return None
    g = lambda *ks: next((float(last[k]) for k in ks if k in last and last[k] and last[k].strip()), 0.0)
    return {
        'seq_rmse': g('test_ar_seq_rmse','val_ar_seq_rmse'), 'seq_mae': g('test_ar_seq_mae','val_ar_seq_mae'), 'seq_r2': g('test_ar_seq_r2','val_ar_seq_r2'),
        'buck_load_mae': g('test_ar_buckling_load_mae','val_ar_buckling_load_mae'), 'buck_load_rmse': g('test_ar_buckling_load_rmse','val_ar_buckling_load_rmse'),
        'buck_load_r2': g('test_ar_buckling_load_r2','val_ar_buckling_load_r2'), 'buck_load_mape': g('test_ar_buckling_load_mape','val_ar_buckling_load_mape'),
        'buck_load_nrmse': g('test_ar_buckling_load_nrmse','val_ar_buckling_load_nrmse'), 'buck_load_pcc': g('test_ar_buckling_load_pcc','val_ar_buckling_load_pcc'),
        'buck_disp_mae': g('test_ar_buckling_disp_mae','val_ar_buckling_disp_mae'), 'buck_disp_rmse': g('test_ar_buckling_disp_rmse','val_ar_buckling_disp_rmse'),
    }

rows = []
for m in ALL_MODELS:
    ld = os.path.join(LOG_DIR, m)
    if not os.path.isdir(ld): continue
    for s in SEEDS:
        p = os.path.join(ld, f'seed_{s}', 'training_metrics.csv')
        if not os.path.isfile(p): continue
        d = extract_falstm(p) if detect(p) == 'falstm' else extract_baseline(p)
        if d is None: continue
        d['model'], d['seed'], d['source'] = m, s, 'test_set'
        rows.append(d)

COLS = ['model','seed','source','seq_r2','seq_mae','seq_rmse',
        'buck_load_r2','buck_load_mae','buck_load_rmse','buck_load_mape','buck_load_nrmse','buck_load_pcc',
        'buck_disp_mae','buck_disp_rmse']

out = 'results/all_models_test_data.csv'
os.makedirs('results', exist_ok=True)
with open(out, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
    w.writeheader()
    for r in sorted(rows, key=lambda x: (ALL_MODELS.index(x['model']), x['seed'])):
        w.writerow(r)

print(f"{out}: {len(rows)} rows, {len(set(r['model'] for r in rows))} models")
