import pandas as pd
from unified_neo4j_manager import UnifiedNeo4jManager
import os


NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

ANNOTATION_CSV = "eval.csv"   # path to CSV
TOP_K = 5
MAX_RESUMES = 5


def evaluate_at_k(manager: UnifiedNeo4jManager,
                  df: pd.DataFrame,
                  k: int = TOP_K,
                  max_resumes: int = MAX_RESUMES):
    """
    df columns:
      resume_id, job_id, label, source, category, notes
    label: 1 = relevant, 0 = not relevant
    """
    metrics = []

    # Make sure we only have the columns we need
    required_cols = {"resume_id", "job_id", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    resume_ids = df["resume_id"].unique()[:max_resumes]

    for resume_id in resume_ids:
        rows = df[df["resume_id"] == resume_id]

        # Ground truth sets
        relevant_jobs = set(rows[rows["label"] == 1]["job_id"])
        negative_jobs = set(rows[rows["label"] == 0]["job_id"])

        if not relevant_jobs and not negative_jobs:
            print(f"[WARN] No labels for resume {resume_id}, skipping.")
            continue

        # Use the first row to get source/category for this resume
        source = rows["source"].iloc[0] if "source" in rows.columns else None
        category = rows["category"].iloc[0] if "category" in rows.columns else None

        try:
            rec_result = manager.recommend_jobs_for_person(
                person_id=resume_id,
                limit=k,
                source=source,
                category=category,
                location=None,
            )
        except Exception as e:
            print(f"[ERROR] recommend_jobs_for_person failed for {resume_id}: {e}")
            continue

        rec_jobs = [j["job_id"] for j in rec_result.get("selected_jobs", [])]
        rec_jobs_set = set(rec_jobs)

        # -------- METRICS (at K) --------
        # True positives: recommended & relevant
        tp = len(relevant_jobs & rec_jobs_set)
        # False positives: recommended but not labeled relevant
        fp = len(rec_jobs_set - relevant_jobs)
        # False negatives: relevant but not recommended
        fn = len(relevant_jobs - rec_jobs_set)

        # Accuracy uses both positive and negative labels we have
        labeled_jobs = relevant_jobs | negative_jobs
        correct = 0
        for jid in labeled_jobs:
            true_label = 1 if jid in relevant_jobs else 0
            pred_label = 1 if jid in rec_jobs_set else 0
            if true_label == pred_label:
                correct += 1
        accuracy = correct / len(labeled_jobs) if labeled_jobs else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics.append(
            {
                "resume_id": resume_id,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            }
        )

        print(
            f"[Resume {resume_id}] "
            f"TP={tp}, FP={fp}, FN={fn} | "
            f"Precision@{k}={precision:.2f}, "
            f"Recall@{k}={recall:.2f}, "
            f"F1@{k}={f1:.2f}, "
            f"Accuracy={accuracy:.2f}"
        )

    if not metrics:
        print("No metrics computed (check CSV / IDs).")
        return

    # -------- MACRO AVERAGE ACROSS RESUMES --------
    macro_precision = sum(m["precision"] for m in metrics) / len(metrics)
    macro_recall = sum(m["recall"] for m in metrics) / len(metrics)
    macro_f1 = sum(m["f1"] for m in metrics) / len(metrics)
    macro_accuracy = sum(m["accuracy"] for m in metrics) / len(metrics)

    print("\n=== Macro-averaged over resumes ===")
    print(f"Precision@{k}: {macro_precision:.3f}")
    print(f"Recall@{k}:    {macro_recall:.3f}")
    print(f"F1@{k}:        {macro_f1:.3f}")
    print(f"Accuracy:      {macro_accuracy:.3f}")


if __name__ == "__main__":
    # Load annotation file
    df = pd.read_csv(ANNOTATION_CSV)

    # Connect to Neo4j
    manager = UnifiedNeo4jManager(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
    )

    try:
        evaluate_at_k(manager, df, k=TOP_K, max_resumes=MAX_RESUMES)
    finally:
        manager.close()
