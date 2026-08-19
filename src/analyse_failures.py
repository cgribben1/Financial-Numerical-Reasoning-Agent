"""Analyse failure modes from an eval results JSON file."""

import json
import sys
from collections import defaultdict


def classify_failure(predicted, gold) -> str:
    """Assign a failure mode label to a wrong answer."""
    try:
        p = float(str(predicted).replace(",", "").replace("$", "").strip())
        g = float(str(gold).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        # String mismatch
        return "string_mismatch"

    if g == 0.0:
        return "other_numeric"

    rel_err = abs(p - g) / (abs(g) + 1e-9)

    # Sign flip: same magnitude, opposite sign
    if abs(p + g) / (abs(g) + 1e-9) < 0.05:
        return "sign_error"

    # Scale errors
    for factor, label in [
        (100, "pct_vs_decimal"),   # 14.1 vs 0.141
        (0.01, "pct_vs_decimal"),
        (1_000_000, "unit_scale"),
        (1_000, "unit_scale"),
        (1_000_000_000, "unit_scale"),
    ]:
        if abs(p / factor - g) / (abs(g) + 1e-9) < 0.01:
            return label

    # Off by one row / adjacent value
    if rel_err < 0.5:
        return "close_miss"

    # Empty / blank answer
    if str(predicted).strip() in {"", "None", "nan"}:
        return "no_answer"

    return "wrong_value"


def detect_cascade(results: list[dict]) -> set[str]:
    """Return record_ids where turn 0 was wrong (cascade risk)."""
    turn0_wrong = set()
    for r in results:
        if r["turn_index"] == 0 and not r["is_correct"]:
            turn0_wrong.add(r["record_id"])
    return turn0_wrong


def main(path: str = "results_200.json") -> None:
    data = json.load(open(path))
    turns = data["turns"]
    summary = data["summary"]

    failures = [t for t in turns if not t["is_correct"]]
    total = len(turns)
    n_fail = len(failures)

    print(f"\n{'='*60}")
    print(f"EVAL SUMMARY  —  {summary['execution_accuracy']:.1%}  ({summary['correct']}/{total})")
    print(f"{'='*60}")

    # --- Failure mode table ---
    mode_counts: dict[str, int] = defaultdict(int)
    cascade_ids = detect_cascade(turns)
    cascade_failures = 0

    for f in failures:
        mode = classify_failure(f["predicted_answer"], f["gold_answer"])
        # Override: if turn > 0 and record had a bad turn 0, flag as cascade
        if f["turn_index"] > 0 and f["record_id"] in cascade_ids:
            cascade_failures += 1
            mode_counts["cascade_from_t0"] += 1
        else:
            mode_counts[mode] += 1

    print(f"\n{'FAILURE MODE':<25} {'COUNT':>6} {'% OF FAILS':>11} {'% OF TOTAL':>11}")
    print("-" * 56)
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f"{mode:<25} {count:>6} {count/n_fail:>10.1%} {count/total:>10.1%}")
    print(f"\n{'TOTAL FAILURES':<25} {n_fail:>6} {'100.0%':>11} {n_fail/total:>10.1%}")

    # --- By turn ---
    print(f"\n{'ACCURACY BY TURN':}")
    print(f"{'Turn':<6} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print("-" * 34)
    by_turn: dict[int, list] = defaultdict(list)
    for t in turns:
        by_turn[t["turn_index"]].append(t["is_correct"])
    for ti in sorted(by_turn):
        vals = by_turn[ti]
        acc = sum(vals) / len(vals)
        print(f"{ti:<6} {sum(vals):>8} {len(vals):>7} {acc:>9.1%}")

    # --- By conversation type ---
    print(f"\n{'ACCURACY BY CONV TYPE':}")
    by_type: dict[str, list] = defaultdict(list)
    for t in turns:
        by_type[t["conversation_type"]].append(t["is_correct"])
    for ct, vals in sorted(by_type.items()):
        print(f"  {ct:<20} {sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.1%}")

    # --- By answer source ---
    print(f"\n{'ACCURACY BY ANSWER SOURCE':}")
    by_src: dict[str, list] = defaultdict(list)
    for t in turns:
        src = t.get("answer_source") or "unknown"
        by_src[src].append(t["is_correct"])
    for src, vals in sorted(by_src.items(), key=lambda x: -len(x[1])):
        print(f"  {src:<25} {sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.1%}")

    # --- Worst conversations ---
    conv_results: dict[str, list] = defaultdict(list)
    for t in turns:
        conv_results[t["record_id"]].append(t["is_correct"])
    worst = sorted(conv_results.items(), key=lambda x: (sum(x[1]), len(x[1])))[:10]

    print(f"\n{'TOP 10 WORST CONVERSATIONS':}")
    print(f"{'Record ID':<45} {'Score':>8}")
    print("-" * 55)
    for rec_id, vals in worst:
        score = f"{sum(vals)}/{len(vals)}"
        cascade = " [cascade]" if rec_id in cascade_ids and sum(vals) < len(vals) else ""
        print(f"{rec_id:<45} {score:>8}{cascade}")

    # --- Sample of wrong answers ---
    print(f"\n{'SAMPLE FAILURES (first 15, non-cascade):':}")
    print(f"{'Record':<35} {'T':>2}  {'Predicted':>14}  {'Gold':>14}  {'Mode':<20}")
    print("-" * 90)
    shown = 0
    for f in failures:
        if f["turn_index"] > 0 and f["record_id"] in cascade_ids:
            continue
        mode = classify_failure(f["predicted_answer"], f["gold_answer"])
        rec = f["record_id"][:34]
        print(f"{rec:<35} {f['turn_index']:>2}  {str(f['predicted_answer']):>14}  {str(f['gold_answer']):>14}  {mode:<20}")
        shown += 1
        if shown >= 15:
            break


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results_200.json"
    main(path)
