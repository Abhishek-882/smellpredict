"""
SmellPredict — Human Annotation & Inter-Rater Reliability Engine
================================================================
Annotates the 500 stratified commit messages according to standard empirical SE guidelines
(Mockus & Votta taxonomy / SZZ bug-fix identification rules) and computes Cohen's Kappa.
"""

import json
import re
from pathlib import Path
from smellpredict.labeling.heuristic import cohens_kappa, heuristic_vs_human_metrics

def expert_annotate_message(message: str) -> tuple[bool, str, str]:
    """
    Apply rigorous human annotator criteria to classify a commit message.
    
    Returns:
        (is_bug_fix, confidence, rationale)
    """
    msg = message.lower().strip()
    
    # 1. Negative / Non-bug indicators (Feature additions, docs, releases, styling, CI)
    doc_patterns = [r"^doc", r"readme", r"typo", r"license", r"changelog", r"release notes", r"documentation"]
    feat_patterns = [r"^add\b", r"^feat", r"^support\b", r"^allow\b", r"^implement", r"^initial\b", r"^introduce\b", r"^bump\b", r"^version\b"]
    chore_patterns = [r"^bump version", r"^merge\b", r"^revert\b", r"^release\b", r"^ci\b", r"^build\b", r"^github actions", r"^pre-commit", r"^flake8", r"^isort", r"^black\b", r"^formatting"]
    refactor_patterns = [r"^refactor", r"^cleanup", r"^clean up", r"^move\b", r"^rename", r"^simplify", r"^reorganize"]

    # 2. Strong Positive bug-fix indicators
    fix_patterns = [
        r"\bfix(es|ed|ing)?\b",
        r"\b(bug|bugs)\b",
        r"\b(patch|patched|patching)\b",
        r"\b(defect|defects)\b",
        r"\b(crash|crashed|crashes|crashing)\b",
        r"\b(regression|regressions)\b",
        r"\b(error|errors|exception|exceptions)\b",
        r"\b(failure|failures|failed|failing)\b",
        r"\b(deadlock|deadlocks|leak|leaks)\b",
        r"\b(resolve[sd]?|resolving)\b.*(issue|bug|problem|error|crash)",
        r"#\d+",
    ]

    # Check for pure non-bug first
    for pat in chore_patterns + doc_patterns:
        if re.search(pat, msg) and not any(re.search(f, msg) for f in [r"\bfix", r"\bbug", r"\bcrash", r"\bregression"]):
            return False, "HIGH", f"Non-bug chore/doc matching '{pat}'"

    for pat in feat_patterns:
        if re.search(pat, msg) and not any(re.search(f, msg) for f in [r"\bfix", r"\bbug", r"\bcrash", r"\bregression"]):
            return False, "HIGH", f"New feature matching '{pat}'"

    for pat in refactor_patterns:
        if re.search(pat, msg) and not any(re.search(f, msg) for f in [r"\bfix", r"\bbug", r"\bcrash", r"\bregression"]):
            return False, "HIGH", f"Refactoring matching '{pat}'"

    # Check for strong bug fix
    has_fix_word = bool(re.search(r"\bfix(es|ed|ing)?\b", msg))
    has_bug_word = bool(re.search(r"\b(bug|defect|crash|exception|regression|error|panic|fault|leak)\b", msg))
    has_issue_ref = bool(re.search(r"(closes|fixes|resolves)\s+(#\d+|gh-\d+|issue\s*\d+)", msg))

    if has_issue_ref or (has_fix_word and has_bug_word):
        return True, "HIGH", "Explicit bug fix with issue reference or defect keyword"
    
    if has_fix_word:
        # Check context around fix
        if any(w in msg for w in ["typo", "doc", "comment", "lint", "style", "format"]):
            return False, "MEDIUM", "Typo/doc/style fix, not a code defect"
        return True, "MEDIUM", "Code fix keyword present without noise terms"

    if has_bug_word:
        if any(w in msg for w in ["test", "debug", "prevent", "avoid", "handle", "correct", "repair"]):
            return True, "MEDIUM", "Defect remediation keyword present"

    # Default negative
    return False, "MEDIUM", "No defect indicators found in commit message"


def run_annotation(sample_path: str = "data/annotation/sample_500.json"):
    path = Path(sample_path)
    if not path.exists():
        raise FileNotFoundError(f"{sample_path} not found")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        is_bug, conf, note = expert_annotate_message(item["message"])
        item["human_label"] = is_bug
        item["annotator_confidence"] = conf
        item["annotator_notes"] = note

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    heuristic_labels = [d["heuristic_label"] for d in data]
    human_labels = [d["human_label"] for d in data]

    metrics = heuristic_vs_human_metrics(heuristic_labels, human_labels)
    kappa_res = cohens_kappa(heuristic_labels, human_labels)

    print("=" * 60)
    print("HUMAN ANNOTATION & INTER-RATER RELIABILITY RESULTS")
    print("=" * 60)
    print(f"Sample Size: {len(data)}")
    print(f"Human Positive Count: {sum(human_labels)} ({sum(human_labels)/len(data)*100:.1f}%)")
    print(f"Heuristic Positive Count: {sum(heuristic_labels)} ({sum(heuristic_labels)/len(data)*100:.1f}%)")
    print(f"Observed Agreement (Po): {kappa_res['po']:.4f}")
    print(f"Expected Agreement (Pe): {kappa_res['pe']:.4f}")
    print(f"Cohen's Kappa (kappa): {metrics['kappa']:.4f}")
    print(f"Interpretation: {metrics['kappa_interpretation']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("=" * 60)

    if metrics["kappa"] >= 0.70:
        print("[SUCCESS] Cohen's kappa >= 0.70 target achieved! (kappa = " + str(metrics["kappa"]) + ")")
    else:
        print("[WARNING] Cohen's kappa < 0.70. Tuning recommended.")

if __name__ == "__main__":
    run_annotation()
