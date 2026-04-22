"""
Replicates exactly what `state tx predict` does after prediction, on a pair of
already-written adata_pred.h5ad + adata_real.h5ad. Skips the model-inference
step so we can iterate on cell-eval behaviour without re-predicting.

Writes one `<ct>_results.csv` + `<ct>_agg_results.csv` per celltype into the
given output dir.
"""

import argparse
import logging
import os

import anndata as ad

from cell_eval import MetricsEvaluator
from cell_eval.utils import split_anndata_on_celltype


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adata-pred", required=True, help="Path to adata_pred.h5ad")
    p.add_argument("--adata-real", required=True, help="Path to adata_real.h5ad")
    p.add_argument("--output-dir", required=True, help="Where to write metric CSVs")
    p.add_argument("--celltype-col", default="celltype")
    p.add_argument("--pert-col", default="target_gene_name")
    p.add_argument("--control-pert", default="non-targeting")
    p.add_argument("--profile", default="de",
                   choices=["full", "minimal", "de", "anndata"])
    p.add_argument("--num-threads", type=int, default=4)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading real from {args.adata_real}")
    real = ad.read_h5ad(args.adata_real)
    logger.info(f"Loading pred from {args.adata_pred}")
    pred = ad.read_h5ad(args.adata_pred)

    ct_split_real = split_anndata_on_celltype(adata=real, celltype_col=args.celltype_col)
    ct_split_pred = split_anndata_on_celltype(adata=pred, celltype_col=args.celltype_col)
    assert set(ct_split_real.keys()) == set(ct_split_pred.keys()), (
        f"Celltype mismatch: {set(ct_split_real) ^ set(ct_split_pred)}"
    )

    # Matches state's pdex_kwargs / skip_metrics exactly (see state _predict.py).
    pdex_kwargs = dict(exp_post_agg=True, is_log1p=True)
    skip_metrics = ["pearson_edistance", "clustering_agreement"]

    for ct in ct_split_real.keys():
        logger.info(f"--- celltype: {ct} ---")
        evaluator = MetricsEvaluator(
            adata_pred=ct_split_pred[ct],
            adata_real=ct_split_real[ct],
            control_pert=args.control_pert,
            pert_col=args.pert_col,
            outdir=args.output_dir,
            prefix=ct,
            num_threads=args.num_threads,
            pdex_kwargs=pdex_kwargs,
        )
        evaluator.compute(
            profile=args.profile,
            skip_metrics=skip_metrics,
        )

    logger.info(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
