"""Run the whole preparation from the raw Excel file and place every customer in a segment.

Usage, from the project folder:

    python -m src.run_pipeline

It uses the model saved by the notebook, so run the notebook once first. If the notebook's own
segment list is present, the script also reports how closely the two agree.
"""
import json

import pandas as pd

from .config import MODEL_DIR, PROCESSED_DIR
from .data_prep import build_customer_features, clean_transactions, load_raw
from .segment import assign_segments


def main():
    raw = load_raw()
    sales, returns = clean_transactions(raw)
    reference_date = json.loads((MODEL_DIR / "preprocessing_config.json").read_text())["reference_date"]
    customers = build_customer_features(sales, returns, today=reference_date)
    segmented = assign_segments(customers)

    print(f"{len(raw):,} raw rows -> {len(sales):,} sales lines -> {len(segmented):,} customers")
    summary = segmented.groupby("Segment").agg(Customers=("Monetary", "size"), Sales=("Monetary", "sum"))
    summary["% of customers"] = (summary["Customers"] / summary["Customers"].sum() * 100).round(1)
    summary["% of sales"] = (summary["Sales"] / summary["Sales"].sum() * 100).round(1)
    print(summary.drop(columns="Sales").sort_values("% of sales", ascending=False).to_string())

    output = PROCESSED_DIR / "customer_segments_from_script.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    segmented.to_csv(output)
    print(f"\nSaved {output.name}")

    notebook_file = PROCESSED_DIR / "customer_segments.csv"
    if notebook_file.exists():
        from_notebook = pd.read_csv(notebook_file, index_col=0)["Segment"]
        match = (segmented["Segment"] == from_notebook.reindex(segmented.index)).mean()
        print(f"Agreement with the notebook's segments: {match:.2%}")


if __name__ == "__main__":
    main()
