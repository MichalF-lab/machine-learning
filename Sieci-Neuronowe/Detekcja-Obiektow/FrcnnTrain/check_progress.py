import json
d = json.load(open(r'D:\CLCO\FrcnnTrain\Runs\bccd\training_metrics.json'))

# Show trajectory in chunks
chunk = 50
print(f"Trajectory by {chunk}-epoch buckets:")
print(f"{'Epochs':<15} {'avg_loss':>10} {'avg_mAP50':>10} {'avg_prec':>10} {'avg_rec':>10}")
for i in range(0, len(d), chunk):
    bucket = d[i:i+chunk]
    n = len(bucket)
    al = sum(e['loss'] for e in bucket)/n
    am = sum(e['mAP50'] for e in bucket)/n
    ap = sum(e['precision'] for e in bucket)/n
    ar = sum(e['recall'] for e in bucket)/n
    print(f"{bucket[0]['epoch']:>3}-{bucket[-1]['epoch']:<10} {al:>10.4f} {am:>10.4f} {ap:>10.4f} {ar:>10.4f}")

# Find peak val mAP and check if it declined after
mAPs = [e['mAP50'] for e in d]
peak_idx = mAPs.index(max(mAPs))
print(f"\nPeak mAP50: ep{d[peak_idx]['epoch']} = {mAPs[peak_idx]:.4f}")
print(f"After peak: {len(d) - peak_idx - 1} more epochs")
if peak_idx < len(d) - 10:
    after = mAPs[peak_idx+1:]
    print(f"Avg mAP after peak: {sum(after)/len(after):.4f}  (decline if < {mAPs[peak_idx]:.4f})")
