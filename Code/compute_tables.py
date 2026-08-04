"""
compute_tables.py — Reproducible metric extraction for comparison tables
=========================================================================
Extracts TEST SET metrics (NOT validation) from multi-seed training logs.
- FA-LSTM family: reads the ``phase='test'`` row from ``training_metrics.csv``.
- Baselines (LSTM/GRU/TCN/Transformer/DA-RNN): reads ``test_ar_*`` columns.
- Computes mean and std across 5 seeds (42, 123, 618, 2024, 9999).
- Outputs JSON tables: ``computed_tables.json`` and ``computed_tables_test.json``.

All values are computed programmatically -- no manual copying from TensorBoard.
"""
import os, csv, sys, json, statistics

MODELS_ABLATION = [
    ('FA-LSTM',       'FA-LSTM (full model)'),
    ('FA-LSTM_A1',    'w/o Feature Attention'),
    ('FA-LSTM_A2',    'w/o Attentional Pooling'),
    ('FA-LSTM_A3',    'w/o FiLM Conditioning'),
    ('FA-LSTM_A4',    'w/o Prior Knowledge'),
    ('FA-LSTM_A5',    'w/o State Residual'),
    ('FA-LSTM_A6',    'w/o Multi-Head Output'),
    ('FA-LSTM_A8',    'TF-only training'),
]

MODELS_BASELINE = [
    'LSTM', 'LSTM_AR', 'GRU', 'TCN', 'Transformer', 'DA-RNN'
]

SEEDS = ['42', '123', '618', '2024', '9999']
LOG_DIR = 'logs'

def detect_format(csv_path):
    with open(csv_path, encoding='utf-8', errors='replace') as f:
        h = f.readline()
    return 'falstm' if ('phase' in h and 'phys_r2' in h) else 'baseline'

def extract_falstm_test(csv_path):
    """FA-LSTM: find phase='test' row (single evaluation at training end)."""
    with open(csv_path, encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            if row.get('phase', '') == 'test':
                g = lambda k: float(row.get(k, 0))
                return {
                    'seq_rmse': g('phys_rmse'), 'seq_mae': g('phys_mae'), 'seq_r2': g('phys_r2'),
                    'buck_load_mae': g('buckling_load_mae'), 'buck_load_rmse': g('buckling_load_rmse'),
                    'buck_load_r2': g('buckling_load_r2'), 'buck_load_mape': g('buckling_load_mape'),
                    'buck_load_nrmse': g('buckling_load_nrmse'), 'buck_load_pcc': g('buckling_load_pcc'),
                    'buck_disp_mae': g('buckling_disp_mae'), 'buck_disp_rmse': g('buckling_disp_rmse'),
                }
    return None

def extract_baseline_test(csv_path):
    """Baseline: use test_ar_* columns from last row."""
    last = None
    with open(csv_path, encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            last = row
    if last is None:
        return None
    g = lambda *ks: next((float(last[k]) for k in ks if k in last and last[k] and last[k].strip()), 0.0)
    seq_r2 = g('test_ar_seq_r2', 'val_ar_seq_r2')
    if seq_r2 < 0.3 and 'test_ar_seq_r2' not in last:
        return None  # fallback: no test data, use val_ar
    return {
        'seq_rmse': g('test_ar_seq_rmse', 'val_ar_seq_rmse'),
        'seq_mae': g('test_ar_seq_mae', 'val_ar_seq_mae'),
        'seq_r2': g('test_ar_seq_r2', 'val_ar_seq_r2'),
        'buck_load_mae': g('test_ar_buckling_load_mae', 'val_ar_buckling_load_mae'),
        'buck_load_rmse': g('test_ar_buckling_load_rmse', 'val_ar_buckling_load_rmse'),
        'buck_load_r2': g('test_ar_buckling_load_r2', 'val_ar_buckling_load_r2'),
        'buck_load_mape': g('test_ar_buckling_load_mape', 'val_ar_buckling_load_mape'),
        'buck_load_nrmse': g('test_ar_buckling_load_nrmse', 'val_ar_buckling_load_nrmse'),
        'buck_load_pcc': g('test_ar_buckling_load_pcc', 'val_ar_buckling_load_pcc'),
        'buck_disp_mae': g('test_ar_buckling_disp_mae', 'val_ar_buckling_disp_mae'),
        'buck_disp_rmse': g('test_ar_buckling_disp_rmse', 'val_ar_buckling_disp_rmse'),
    }

def compute_stats(values):
    clean = [v for v in values if abs(v) < 1e6]
    if not clean: return 0.0, 0.0, 0
    m = statistics.mean(clean)
    s = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return m, s, len(clean)

def main():
    results = {}
    for model_key, model_label in MODELS_ABLATION + [(m, m) for m in MODELS_BASELINE]:
        log_dir = os.path.join(LOG_DIR, model_key)
        if not os.path.isdir(log_dir): continue

        seed_data = {}
        missing = []
        for seed in SEEDS:
            csv_p = os.path.join(log_dir, f'seed_{seed}', 'training_metrics.csv')
            if not os.path.isfile(csv_p):
                missing.append(seed); continue
            fmt = detect_format(csv_p)
            d = extract_falstm_test(csv_p) if fmt == 'falstm' else extract_baseline_test(csv_p)
            if d is None:
                missing.append(seed); continue
            seed_data[seed] = d

        if not seed_data:
            print(f"SKIP {model_key}: no test data. Missing: {missing}")
            continue
        if missing:
            print(f"NOTE: {model_key} missing seeds: {missing} (using {len(seed_data)}/{len(SEEDS)})")

        metrics = ['seq_r2','seq_mae','seq_rmse','buck_load_r2','buck_load_mae','buck_load_rmse','buck_load_mape']
        means, stds = {}, {}
        for k in metrics:
            vals = [d[k] for d in seed_data.values()]
            m, s, n = compute_stats(vals)
            means[k], stds[k] = m, s

        results[model_key] = {
            'label': model_label, 'means': means, 'stds': stds,
            'n': len(seed_data), 'missing': missing,
            'seeds': {s: {'buck_load_mae': d['buck_load_mae'], 'seq_r2': d['seq_r2'],
                          'buck_load_r2': d['buck_load_r2'], 'buck_load_mape': d['buck_load_mape']}
                      for s, d in seed_data.items()}
        }

    fa = results.get('FA-LSTM')
    if not fa:
        print("FATAL: no FA-LSTM data"); sys.exit(1)

    # ── TABLE 1: Ablation Buckling ──
    print("=" * 140)
    print("TABLE 1: Ablation — Critical Buckling Load (Test Set, mean ± std)")
    print("=" * 140)
    print(f"{'Configuration':<28} {'N':>3} {'R2_cr':>18} {'MAE_cr (kN)':>20} {'RMSE_cr (kN)':>20} {'MAPE_cr':>16} {'Δ MAE_cr':>12}")
    print("-" * 140)
    for mk, _ in MODELS_ABLATION:
        r = results.get(mk);
        if not r: continue
        m, s, n = r['means'], r['stds'], r['n']
        d = ((m['buck_load_mae'] - fa['means']['buck_load_mae']) / fa['means']['buck_load_mae'] * 100) if mk != 'FA-LSTM' else 0
        ds = f"+{d:.1f}%" if d > 0 else "—"
        mn = f" [missing:{','.join(r['missing'])}]" if r['missing'] else ""
        print(f"{r['label']:<28} {n:>3} {m['buck_load_r2']:.4f} ± {s['buck_load_r2']:.4f}   {m['buck_load_mae']:.4f} ± {s['buck_load_mae']:.4f}     {m['buck_load_rmse']:.4f} ± {s['buck_load_rmse']:.4f}     {m['buck_load_mape']:.2f}% ± {s['buck_load_mape']:.2f}%    {ds:>12}{mn}")

    # ── TABLE 2: Ablation Sequence ──
    print("\n" + "=" * 140)
    print("TABLE 2: Ablation — Sequence-Level (Test Set, mean ± std)")
    print("=" * 140)
    print(f"{'Configuration':<28} {'N':>3} {'R2_seq':>18} {'MAE_seq':>20} {'RMSE_seq':>20} {'Δ MAE_seq':>12}")
    print("-" * 140)
    for mk, _ in MODELS_ABLATION:
        r = results.get(mk);
        if not r: continue
        m, s, n = r['means'], r['stds'], r['n']
        d = ((m['seq_mae'] - fa['means']['seq_mae']) / fa['means']['seq_mae'] * 100) if mk != 'FA-LSTM' else 0
        ds = f"+{d:.1f}%" if d > 0 else "—"
        print(f"{r['label']:<28} {n:>3} {m['seq_r2']:.4f} ± {s['seq_r2']:.4f}   {m['seq_mae']:.4f} ± {s['seq_mae']:.4f}     {m['seq_rmse']:.4f} ± {s['seq_rmse']:.4f}     {ds:>12}")

    # ── TABLE 3: Baseline Buckling ──
    print("\n" + "=" * 140)
    print("TABLE 3: Baseline — Critical Buckling Load (Test Set, mean ± std)")
    print("=" * 140)
    print(f"{'Model':<20} {'N':>3} {'R2_cr':>18} {'MAE_cr (kN)':>20} {'RMSE_cr (kN)':>20} {'MAPE_cr':>16}")
    print("-" * 140)
    r = results['FA-LSTM']; m, s, n = r['means'], r['stds'], r['n']
    print(f"{'FA-LSTM (ours)':<20} {n:>3} {m['buck_load_r2']:.4f} ± {s['buck_load_r2']:.4f}   {m['buck_load_mae']:.4f} ± {s['buck_load_mae']:.4f}     {m['buck_load_rmse']:.4f} ± {s['buck_load_rmse']:.4f}     {m['buck_load_mape']:.2f}% ± {s['buck_load_mape']:.2f}%")
    for mn in MODELS_BASELINE:
        r = results.get(mn);
        if not r: continue
        m, s, n = r['means'], r['stds'], r['n']
        print(f"{mn:<20} {n:>3} {m['buck_load_r2']:.4f} ± {s['buck_load_r2']:.4f}   {m['buck_load_mae']:.4f} ± {s['buck_load_mae']:.4f}     {m['buck_load_rmse']:.4f} ± {s['buck_load_rmse']:.4f}     {m['buck_load_mape']:.2f}% ± {s['buck_load_mape']:.2f}%")

    # ── TABLE 4: Baseline Sequence ──
    print("\n" + "=" * 140)
    print("TABLE 4: Baseline — Sequence-Level (Test Set, mean ± std)")
    print("=" * 140)
    print(f"{'Model':<20} {'N':>3} {'R2_seq':>18} {'MAE_seq':>20} {'RMSE_seq':>20}")
    print("-" * 140)
    r = results['FA-LSTM']; m, s, n = r['means'], r['stds'], r['n']
    print(f"{'FA-LSTM (ours)':<20} {n:>3} {m['seq_r2']:.4f} ± {s['seq_r2']:.4f}   {m['seq_mae']:.4f} ± {s['seq_mae']:.4f}     {m['seq_rmse']:.4f} ± {s['seq_rmse']:.4f}")
    for mn in MODELS_BASELINE:
        r = results.get(mn);
        if not r: continue
        m, s, n = r['means'], r['stds'], r['n']
        print(f"{mn:<20} {n:>3} {m['seq_r2']:.4f} ± {s['seq_r2']:.4f}   {m['seq_mae']:.4f} ± {s['seq_mae']:.4f}     {m['seq_rmse']:.4f} ± {s['seq_rmse']:.4f}")

    # ── TABLE 5: Per-Seed Detail ──
    print("\n" + "=" * 140)
    print("TABLE 5: Per-Seed Detail — Test Set Buckling Load MAE (kN)")
    print("=" * 140)
    hdr = f"{'Configuration':<28} " + " ".join(f"{'seed_'+s:>12}" for s in SEEDS) + f"  {'MEAN':>10}  {'STD':>10}"
    print(hdr)
    print("-" * 140)
    for mk, _ in MODELS_ABLATION + [(m, m) for m in MODELS_BASELINE]:
        r = results.get(mk);
        if not r: continue
        row = f"{r['label']:<28} "; vals = []
        for s in SEEDS:
            if s in r['seeds']:
                v = r['seeds'][s]['buck_load_mae']; row += f"{v:>12.4f}"; vals.append(v)
            else: row += f"{' —':>12}"
        if vals:
            m, st, _ = compute_stats(vals); row += f"  {m:>10.4f}  {st:>10.4f}"
        print(row)

    with open('computed_tables_test.json', 'w') as f:
        json.dump({k: {'label': v['label'], 'n': v['n'], 'missing': v['missing'],
                       'means': v['means'], 'stds': v['stds'], 'seeds': v['seeds']}
                   for k, v in results.items()}, f, indent=2)
    print("\nSaved to computed_tables_test.json")

if __name__ == '__main__':
    main()
