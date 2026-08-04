"""Fixed: extract best VAL_AR epoch for baselines, using AR metrics as selection criterion."""
import os, csv, sys, json, statistics

MODELS = [
    'FA-LSTM', 'FA-LSTM_A1', 'FA-LSTM_A2', 'FA-LSTM_A3',
    'FA-LSTM_A4', 'FA-LSTM_A5', 'FA-LSTM_A8',
    'LSTM', 'LSTM_AR', 'GRU', 'TCN', 'Transformer'
]
SEEDS = ['42', '123', '618', '2024', '9999']

def extract_falstm_format(csv_path):
    """FA-LSTM: find epoch with lowest val_ar total_loss."""
    best = None
    best_loss = float('inf')
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('phase', '') != 'val_ar':
                continue
            loss = float(row['total_loss'])
            if loss < best_loss:
                best_loss = loss
                best = row
    if best is None:
        return None
    def f(key, default=0.0):
        try: return float(best.get(key, default))
        except: return default
    return {
        'epoch': best.get('epoch', '?'),
        'total_loss': f('total_loss'),
        'seq_rmse': f('phys_rmse'),
        'seq_mae': f('phys_mae'),
        'seq_r2': f('phys_r2'),
        'buckling_load_mae': f('buckling_load_mae'),
        'buckling_load_rmse': f('buckling_load_rmse'),
        'buckling_load_r2': f('buckling_load_r2'),
        'buckling_load_mape': f('buckling_load_mape'),
    }

def extract_baseline_format(csv_path):
    """Baseline: find epoch with lowest val_ar_seq_rmse (AR mode), report val_ar_* metrics."""
    best = None
    best_rmse = float('inf')
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use val_ar_seq_rmse as selection criterion
            val_ar_rmse = row.get('val_ar_seq_rmse', '')
            if not val_ar_rmse or val_ar_rmse == '':
                continue
            try:
                rmse = float(val_ar_rmse)
            except:
                continue
            # Filter out obviously bad/unrealistic values (0 or near-0 means not populated)
            if rmse < 0.01:
                continue
            if rmse < best_rmse:
                best_rmse = rmse
                best = row
    if best is None:
        return None

    def g(*keys):
        for k in keys:
            try:
                v = float(best.get(k, ''))
                if v == v and abs(v) < 1e8:
                    return v
            except:
                pass
        return 0.0

    return {
        'epoch': best.get('epoch', '?'),
        'total_loss': g('val_ar_seq_rmse'),  # use RMSE as loss proxy
        'seq_rmse': g('val_ar_seq_rmse'),
        'seq_mae': g('val_ar_seq_mae'),
        'seq_r2': g('val_ar_seq_r2'),
        'buckling_load_mae': g('val_ar_buckling_load_mae'),
        'buckling_load_rmse': g('val_ar_buckling_load_rmse'),
        'buckling_load_r2': g('val_ar_buckling_load_r2'),
        'buckling_load_mape': g('val_ar_buckling_load_mape'),
    }

def detect_format(csv_path):
    with open(csv_path) as f:
        header = f.readline()
    if 'phase' in header and 'phys_r2' in header:
        return 'falstm'
    return 'baseline'

def main():
    base = 'logs'
    print("=" * 130)
    hdr = f"{'Model':<22} {'Seed':<6} {'Ep':<5} {'Seq R2':>9} {'Seq MAE':>10} {'Seq RMSE':>10} {'Buck R2':>10} {'Buck MAE':>12} {'Buck RMSE':>12} {'Buck MAPE':>10}"
    print(hdr)
    print("-" * 130)

    all_results = {}

    for model in MODELS:
        log_dir = os.path.join(base, model)
        if not os.path.exists(log_dir):
            continue

        seeds_data = []
        for seed in SEEDS:
            csv_path = os.path.join(log_dir, f'seed_{seed}', 'training_metrics.csv')
            if not os.path.exists(csv_path):
                print(f"  {model:<22} {seed:<6} MISSING FILE")
                continue

            fmt = detect_format(csv_path)
            d = extract_falstm_format(csv_path) if fmt == 'falstm' else extract_baseline_format(csv_path)

            if d is None:
                print(f"  {model:<22} {seed:<6} NO VALID DATA")
                continue

            seeds_data.append(d)
            print(f"  {model:<22} {seed:<6} {str(d['epoch']):<5} "
                  f"{d['seq_r2']:>9.4f} {d['seq_mae']:>10.4f} {d['seq_rmse']:>10.4f} "
                  f"{d['buckling_load_r2']:>10.4f} {d['buckling_load_mae']:>12.4f} "
                  f"{d['buckling_load_rmse']:>12.4f} {d['buckling_load_mape']:>10.2f}%")

        if len(seeds_data) < 2:
            continue

        metrics = ['seq_r2', 'seq_mae', 'seq_rmse', 'buckling_load_r2',
                   'buckling_load_mae', 'buckling_load_rmse', 'buckling_load_mape']

        means = {}
        stds = {}
        for k in metrics:
            vals = [d[k] for d in seeds_data if abs(d[k]) < 100]  # filter extreme outliers
            if vals:
                means[k] = statistics.mean(vals)
                stds[k] = statistics.stdev(vals) if len(vals) > 1 else 0.0

        all_results[model] = {'means': means, 'stds': stds, 'n': len(seeds_data)}

    # Summary
    print("\n" + "=" * 130)
    print("SUMMARY: Mean +/- Std (Best AR epoch per seed)")
    print("-" * 130)
    print(f"{'Model':<22} {'N':<3} {'Seq R2':>20} {'Seq MAE':>20} {'Buck R2':>20} {'Buck MAE':>20} {'Buck MAPE':>18}")
    print("-" * 130)

    for model in MODELS:
        if model not in all_results:
            continue
        r = all_results[model]
        m, s, n = r['means'], r['stds'], r['n']
        print(f"{model:<22} {n:<3} "
              f"{m['seq_r2']:.4f} +/- {s['seq_r2']:.4f}   "
              f"{m['seq_mae']:.4f} +/- {s['seq_mae']:.4f}   "
              f"{m['buckling_load_r2']:.4f} +/- {s['buckling_load_r2']:.4f}   "
              f"{m['buckling_load_mae']:.4f} +/- {s['buckling_load_mae']:.4f}   "
              f"{m['buckling_load_mape']:.2f}% +/- {s['buckling_load_mape']:.2f}%")

    with open('best_metrics_per_seed.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to best_metrics_per_seed.json")

if __name__ == '__main__':
    main()
