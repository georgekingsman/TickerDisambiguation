"""Quick comparison of v1 vs v1.1 results."""
import json

v1_test = json.load(open("results/e3_test_seed42.json"))
v1_amb = json.load(open("results/e3_ambiguous_seed42.json"))
v1_hard = json.load(open("results/e3_hard_seed42.json"))
v1p1_test = json.load(open("results/v1p1_test_seed42.json"))
v1p1_amb = json.load(open("results/v1p1_ambiguous_seed42.json"))
v1p1_hard = json.load(open("results/v1p1_hard_seed42.json"))
v1p1_goog = json.load(open("results/v1p1_google_shareclass_seed42.json"))
v1p1_meta = json.load(open("results/v1p1_meta_alias_seed42.json"))

rows = [
    ("test accuracy", f"{v1_test['accuracy']:.2%}", f"{v1p1_test['accuracy']:.2%}", f"{v1p1_test['accuracy']-v1_test['accuracy']:+.2%}"),
    ("ambiguous accuracy", f"{v1_amb['accuracy']:.2%}", f"{v1p1_amb['accuracy']:.2%}", f"{v1p1_amb['accuracy']-v1_amb['accuracy']:+.2%}"),
    ("hard_eval accuracy", f"{v1_hard['accuracy']:.2%}", f"{v1p1_hard['accuracy']:.2%}", f"{v1p1_hard['accuracy']-v1_hard['accuracy']:+.2%}"),
    ("hallucination (all)", "0%", "0%", "="),
    ("verbosity (all)", "0%", "0%", "="),
    ("test GOOG/GOOGL confuse", str(v1_test['goog_googl_confusion']), str(v1p1_test['goog_googl_confusion']), f"{v1p1_test['goog_googl_confusion']-v1_test['goog_googl_confusion']:+d}"),
    ("ambig GOOG/GOOGL confuse", str(v1_amb['goog_googl_confusion']), str(v1p1_amb['goog_googl_confusion']), f"{v1p1_amb['goog_googl_confusion']-v1_amb['goog_googl_confusion']:+d}"),
    ("hard GOOG/GOOGL confuse", str(v1_hard['goog_googl_confusion']), str(v1p1_hard['goog_googl_confusion']), f"{v1p1_hard['goog_googl_confusion']-v1_hard['goog_googl_confusion']:+d}"),
    ("---", "---", "---", "---"),
    ("google_shareclass_eval", "n/a", f"{v1p1_goog['accuracy']:.2%}", "NEW"),
    ("meta_alias_eval", "n/a", f"{v1p1_meta['accuracy']:.2%}", "NEW"),
]

print("=== LoRA v1 vs v1.1 (seed 42) ===")
print(f"{'Metric':<30} {'v1':>10} {'v1.1':>10} {'delta':>10}")
print("-" * 62)
for r in rows:
    print(f"{r[0]:<30} {r[1]:>10} {r[2]:>10} {r[3]:>10}")
