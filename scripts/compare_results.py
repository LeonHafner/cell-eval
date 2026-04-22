"""
Compare two cell-eval result directories (e.g. buggy vs fixed).

Each directory should contain per-celltype output from run_eval_only.py:
    <ct>_results.csv          (per-perturbation rows)
    <ct>_agg_results.csv      (describe()-style summary)

Produces:
    - a per-metric mean diff table
    - a list of moved vs unchanged metrics
    - per-perturbation diffs for the top-moving metrics
"""

import argparse
import glob
import os
import sys

import polars as pl


def _load_agg(results_dir: str) -> pl.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_agg_results.csv"))):
        ct = os.path.basename(path).replace("_agg_results.csv", "")
        df = pl.read_csv(path).with_columns(pl.lit(ct).alias("celltype"))
        frames.append(df)
    if not frames:
        print(f"no agg CSVs found in {results_dir}", file=sys.stderr)
        sys.exit(1)
    return pl.concat(frames, how="diagonal_relaxed")


def _load_perpert(results_dir: str) -> pl.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_results.csv"))):
        fname = os.path.basename(path)
        if fname.endswith("_agg_results.csv"):
            continue
        ct = fname.replace("_results.csv", "")
        df = pl.read_csv(path).with_columns(pl.lit(ct).alias("celltype"))
        frames.append(df)
    if not frames:
        print(f"no per-pert CSVs found in {results_dir}", file=sys.stderr)
        sys.exit(1)
    return pl.concat(frames, how="diagonal_relaxed")


def compare_means(agg_a: pl.DataFrame, agg_b: pl.DataFrame,
                  label_a: str, label_b: str) -> pl.DataFrame:
    """Melt the `mean` row of each agg frame and diff per metric per celltype."""
    stat_col = "statistic" if "statistic" in agg_a.columns else agg_a.columns[0]
    keep = ["celltype"]

    def melt_mean(df: pl.DataFrame, label: str) -> pl.DataFrame:
        mean_rows = df.filter(pl.col(stat_col) == "mean").drop(stat_col)
        val_cols = [c for c in mean_rows.columns if c not in keep]
        return (
            mean_rows.unpivot(index=keep, on=val_cols,
                              variable_name="metric", value_name=label)
                     .with_columns(pl.col(label).cast(pl.Float64, strict=False))
        )

    a = melt_mean(agg_a, label_a)
    b = melt_mean(agg_b, label_b)
    return (
        a.join(b, on=["celltype", "metric"], how="inner")
         .with_columns(
             abs_diff=(pl.col(label_b) - pl.col(label_a)).abs(),
             rel_diff=(pl.col(label_b) - pl.col(label_a)) / pl.col(label_a).abs(),
         )
         .sort("abs_diff", descending=True, nulls_last=True)
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="first results dir (e.g. buggy)")
    p.add_argument("--b", required=True, help="second results dir (e.g. fixed)")
    p.add_argument("--label-a", default="buggy")
    p.add_argument("--label-b", default="fixed")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="abs_diff below this is treated as unchanged")
    p.add_argument("--top", type=int, default=10,
                   help="how many biggest-moving metrics to drill into")
    args = p.parse_args()

    agg_a = _load_agg(args.a)
    agg_b = _load_agg(args.b)

    comp = compare_means(agg_a, agg_b, args.label_a, args.label_b)
    moved = comp.filter(pl.col("abs_diff") > args.tol)
    unchanged = comp.filter(pl.col("abs_diff") <= args.tol)

    print("=" * 70)
    print(f"Per-metric mean-over-perturbations comparison  ({args.label_a} vs {args.label_b})")
    print("=" * 70)
    with pl.Config(tbl_rows=60, tbl_width_chars=140):
        print(comp)

    print()
    print(f"Metrics that moved     (|diff| > {args.tol}):  {moved.height}")
    print(f"Metrics unchanged      (|diff| <={args.tol}):  {unchanged.height}")
    print()
    if unchanged.height:
        unchanged_list = sorted(set(unchanged["metric"].to_list()))
        print("Unchanged metrics (should be fdr-only + anndata):")
        for m in unchanged_list:
            print(f"  - {m}")

    # Per-perturbation drill-down
    if moved.height:
        top_metrics = moved["metric"].unique().to_list()[: args.top]
        print()
        print(f"Per-perturbation values for top {len(top_metrics)} moving metrics:")
        pert_a = _load_perpert(args.a)
        pert_b = _load_perpert(args.b)
        join_key = [c for c in pert_a.columns
                    if c.lower() in ("perturbation", "target", "target_gene_name") or c == "celltype"]
        for m in top_metrics:
            if m not in pert_a.columns or m not in pert_b.columns:
                continue
            side = (
                pert_a.select([*join_key, m]).rename({m: f"{m}__{args.label_a}"})
                      .join(pert_b.select([*join_key, m]).rename({m: f"{m}__{args.label_b}"}),
                            on=join_key, how="inner")
            )
            print()
            print(f"--- {m} ---")
            with pl.Config(tbl_rows=40, tbl_width_chars=140):
                print(side)


if __name__ == "__main__":
    main()
