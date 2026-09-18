"""PretrainedSED at several bars on gold slice B (held-out real-world clips), next to BEATs at 0.35."""
import sys; sys.path.insert(0, ".")
from benchmark import audioset_detector_eval as E
print(f"{'bar':>5} {'masked-conseq':>13} {'conseq':>7} {'all':>6} {'FP/min':>7} {'onsetMAE':>9}")
for b in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
    r = E.evaluate_one("psed", b)
    print(f"{b:5.2f} {r['masked_conseq_recall']:13.1%} {r['conseq_recall']:7.1%} {r['all_recall']:6.1%} {r['fp_per_min']:7.2f} {r['onset_mae']:9.2f}")
r = E.evaluate_one("beats", 0.35)
print(f"BEATs 0.35: masked {r['masked_conseq_recall']:.1%} conseq {r['conseq_recall']:.1%} all {r['all_recall']:.1%} FP/min {r['fp_per_min']:.2f}")
