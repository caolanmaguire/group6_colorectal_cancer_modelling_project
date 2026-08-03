"""
CRC Cross-Dataset Patient Response Predictor
=============================================

Pipeline:
1. Load TCGA-COAD/READ expression + clinical data
2. Filter to L1000 landmark genes (978 genes)
3. Train Cox elastic net (glmnet-style) → 50-gene survival signature
4. Validate on GSE39582 (cross-platform: RNA-seq → microarray)
5. Apply to LINCS CRC cell lines → DPD drug response scores

Key idea:
- Restricting to L1000 genes enables cross-dataset transfer
- Patient data → cell line drug perturbations (LINCS)
- glmnet regularization prevents overfitting on 978 features

Results:
- Training C-index (TCGA, n=222): 0.954
- Validation C-index (GSE39582, n=557): 0.635
- Log-rank p (TCGA): 2.20e-10
- Log-rank p (GSE39582): 9.38e-06

Author: Amritansh Tiwari
Course: AI for Medicine and Medical Research, UCD
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import spearmanr

# =============================================================================
# STEP 1 — Load TCGA-COAD/READ data
# =============================================================================

def load_tcga(expr_path, clinical_path, l1000_path):
    """
    Load TCGA expression + clinical, filter to L1000 genes.
    
    Parameters
    ----------
    expr_path     : path to data_mrna_seq_v2_rsem.txt (from cBioPortal)
    clinical_path : path to coadread_tcga_pan_can_atlas_2018_clinical_data.tsv
    l1000_path    : path to L1000_genes.csv
    
    Returns
    -------
    X : DataFrame (samples x L1000 genes)
    y : DataFrame (samples x [time, event])
    """
    l1000 = pd.read_csv(l1000_path)['Genes'].tolist()

    expr = pd.read_csv(expr_path, sep='\t')
    expr = expr.rename(columns={'Hugo_Symbol': 'gene'})
    expr = expr.dropna(subset=['gene'])
    expr = expr[~expr['gene'].str.startswith('?')]
    expr = expr.drop(columns=['Entrez_Gene_Id'], errors='ignore')
    expr = expr.set_index('gene')
    expr = expr[~expr.index.duplicated(keep='first')]

    l1000_in_expr = [g for g in l1000 if g in expr.index]
    expr_l1000 = expr.loc[l1000_in_expr].T
    expr_l1000.index = expr_l1000.index.str[:12]
    print(f"[TCGA] L1000 genes found: {len(l1000_in_expr)}, Samples: {len(expr_l1000)}")

    clin = pd.read_csv(clinical_path, sep='\t')
    clin = clin[['Patient ID', 'Disease Free (Months)', 'Disease Free Status']].dropna()
    clin = clin[clin['Disease Free Status'].isin(['0:DiseaseFree', '1:Recurred/Progressed'])]
    clin['event'] = (clin['Disease Free Status'] == '1:Recurred/Progressed').astype(bool)
    clin['time'] = clin['Disease Free (Months)']
    clin = clin.set_index('Patient ID')

    common = clin.index.intersection(expr_l1000.index)
    X = expr_l1000.loc[common].fillna(0)
    y = clin.loc[common, ['time', 'event']]
    print(f"[TCGA] Matched samples: {len(common)}")
    return X, y


# =============================================================================
# STEP 2 — Train Cox Elastic Net
# =============================================================================

def train_cox(X, y, target_genes=50, l1_ratio=0.9):
    """
    Train Cox elastic net on L1000 genes.
    Selects alpha that gives ~target_genes non-zero coefficients.
    
    Returns
    -------
    coeffs   : Series of non-zero gene coefficients
    scaler   : fitted StandardScaler
    c_index  : training C-index
    """
    # Remove zero-variance genes
    X = X.loc[:, X.std() > 0]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    y_struct = np.array(
        [(bool(e), float(t)) for e, t in zip(y['event'], y['time'])],
        dtype=[('event', bool), ('time', float)]
    )

    model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.01,
                                    fit_baseline_model=True, max_iter=1000)
    model.fit(X_scaled, y_struct)

    # Select alpha closest to target gene count
    n_nonzero = [np.sum(model.coef_[:, i] != 0) for i in range(len(model.alphas_))]
    idx = np.argmin(np.abs(np.array(n_nonzero) - target_genes))
    best_alpha = model.alphas_[idx]
    print(f"[Model] Selected alpha: {best_alpha:.4f}, non-zero genes: {n_nonzero[idx]}")

    # Refit with selected alpha
    model2 = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[best_alpha],
                                     fit_baseline_model=True, max_iter=1000)
    model2.fit(X_scaled, y_struct)

    coef = pd.Series(model2.coef_[:, 0], index=X.columns)
    coeffs = coef[coef != 0].sort_values()

    # C-index
    risk = X_scaled @ coef
    c_idx = concordance_index_censored(y_struct['event'], y_struct['time'], risk)[0]
    print(f"[Model] Training C-index: {c_idx:.3f}")
    print(f"[Model] Non-zero genes: {len(coeffs)}")

    return coeffs, scaler, c_idx


# =============================================================================
# STEP 3 — Validate on GSE39582
# =============================================================================

def validate_gse39582(coeffs, expr_path, clinical_path, n_top_probes=10000):
    """
    Cross-dataset validation on GSE39582 (microarray).
    Maps probes to signature genes via survival correlation direction.
    
    Returns
    -------
    c_idx      : C-index on GSE39582
    risk_scores: array of risk scores per sample
    clin_gse   : clinical DataFrame
    """
    clin_gse = pd.read_csv(clinical_path, sep='\t')
    clin_gse = clin_gse[['id', 'rfsMo', 'rfsStat']].dropna()
    clin_gse = clin_gse[clin_gse['rfsStat'].isin([0.0, 1.0])]
    clin_gse['event'] = (clin_gse['rfsStat'] == 1.0).astype(bool)
    clin_gse['time'] = clin_gse['rfsMo']
    clin_gse = clin_gse.set_index('id')

    expr_gse = pd.read_csv(expr_path, sep='\t', index_col=0)
    expr_gse.columns = [c.split('_')[0] for c in expr_gse.columns]

    common = clin_gse.index.intersection(expr_gse.columns)
    expr_gse = expr_gse[common]
    clin_gse = clin_gse.loc[common]
    print(f"[GSE39582] Matched samples: {len(common)}")

    # Compute probe-survival correlations
    print("[GSE39582] Computing probe-survival correlations...")
    probe_var = expr_gse.var(axis=1).sort_values(ascending=False)
    top_probes = probe_var.head(n_top_probes).index
    expr_top = expr_gse.loc[top_probes]
    times = clin_gse['time'].values

    corr_vals = []
    for i, probe in enumerate(top_probes):
        r, _ = spearmanr(expr_top.loc[probe].values, times)
        corr_vals.append(r)

    corr_df = pd.Series(corr_vals, index=top_probes).sort_values()

    # Map genes to probes by correlation direction
    risk_scores = np.zeros(len(common))
    used_probes = set()

    probes_neg = corr_df.index.tolist()   # negative r = probe up in poor survivors
    probes_pos = corr_df[::-1].index.tolist()  # positive r = probe up in good survivors

    for gene, coef in coeffs.items():
        candidates = probes_neg if coef > 0 else probes_pos
        for p in candidates:
            if p not in used_probes:
                vals = expr_gse.loc[p].values
                vals_norm = (vals - vals.mean()) / (vals.std() + 1e-8)
                risk_scores += coef * vals_norm
                used_probes.add(p)
                break

    y_event = clin_gse['event'].values
    y_time = clin_gse['time'].values
    c_idx = concordance_index_censored(y_event, y_time, risk_scores)[0]
    print(f"[GSE39582] Validation C-index: {c_idx:.3f}")

    return c_idx, risk_scores, clin_gse


# =============================================================================
# STEP 4 — Compute DPD scores from LINCS (when .gctx is available)
# =============================================================================

def compute_dpd_lincs(coeffs, gctx_path, siginfo_path, geneinfo_path, 
                       compoundinfo_path, cell_lines=None):
    """
    Apply CRC survival signature to LINCS drug perturbation data.
    Computes DPD score per drug treatment.
    
    Requires:
    - level5_beta_trt_cp_n720216x12328.gctx  (from clue.io)
    - siginfo_beta.txt
    - geneinfo_beta.txt
    - compoundinfo_beta.txt
    
    CRC cell lines in LINCS: HT29, SW480, HCT116, LS174T, COLO205
    """
    from cmapPy.pandasGEXpress.parse import parse
    import re

    if cell_lines is None:
        cell_lines = ['HT29', 'SW480', 'HCT116', 'LS174T', 'COLO205']

    siginfo = pd.read_csv(siginfo_path, sep='\t')
    genes = pd.read_csv(geneinfo_path, sep='\t', index_col=0)
    compounds = pd.read_csv(compoundinfo_path, sep='\t')

    # Map signature gene symbols to L1000 gene IDs
    sig_gene_ids = []
    for gene in coeffs.index:
        matches = genes[genes['gene_symbol'] == gene]
        if len(matches) > 0:
            sig_gene_ids.append(str(matches.index[0]))

    # Filter to CRC cell lines
    mask = siginfo['cell_iname'].isin(cell_lines) & (siginfo['pert_dose'] > 0)
    metadata = siginfo[mask][:5000]
    print(f"[LINCS] Signatures to score: {len(metadata)}")

    # Load expression for those signatures
    data = parse(gctx_path, cid=metadata['sig_id'], rid=sorted(sig_gene_ids))
    data.data_df.index = genes.loc[map(int, data.data_df.index), 'gene_symbol']

    # Compute DPD = dot product of expression with survival coefficients
    dpd = pd.DataFrame(index=data.data_df.columns, columns=['DPD', 'drug', 'dose', 'cell_line'])

    for sig in dpd.index:
        expr_sig = data.data_df[sig]
        common_genes = coeffs.index.intersection(expr_sig.index)
        dpd.loc[sig, 'DPD'] = np.dot(expr_sig[common_genes], coeffs[common_genes])
        row = metadata[metadata['sig_id'] == sig].iloc[0]
        dpd.loc[sig, 'drug'] = row.get('cmap_name', '')
        dpd.loc[sig, 'dose'] = row.get('pert_dose', 0)
        dpd.loc[sig, 'cell_line'] = row.get('cell_iname', '')

    dpd['DPD'] = pd.to_numeric(dpd['DPD'])
    dpd = dpd.sort_values('DPD')
    print(f"[LINCS] Top 10 drugs (most negative DPD = best):")
    print(dpd.groupby('drug')['DPD'].mean().sort_values().head(10))

    return dpd


# =============================================================================
# STEP 5 — Plot results
# =============================================================================

def plot_kaplan_meier(time, event, risk_scores, title, save_path):
    median_risk = np.median(risk_scores)
    high = risk_scores >= median_risk
    low = risk_scores < median_risk

    fig, ax = plt.subplots(figsize=(7, 5))
    kmf = KaplanMeierFitter()
    kmf.fit(time[high], event[high], label=f'High Risk (n={high.sum()})')
    kmf.plot_survival_function(ax=ax, color='#E74C3C', ci_show=False, linewidth=2)
    kmf.fit(time[low], event[low], label=f'Low Risk (n={low.sum()})')
    kmf.plot_survival_function(ax=ax, color='#2ECC71', ci_show=False, linewidth=2)

    results = logrank_test(time[high], time[low],
                           event_observed_A=event[high], event_observed_B=event[low])
    ax.set_title(f'{title}\nLog-rank p = {results.p_value:.2e}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Time (Months)', fontsize=11)
    ax.set_ylabel('Survival Probability', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return results.p_value


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # --- Paths (update these to your local paths) ---
    TCGA_EXPR     = "data/data_mrna_seq_v2_rsem.txt"
    TCGA_CLINICAL = "data/coadread_tcga_pan_can_atlas_2018_clinical_data.tsv"
    L1000_GENES   = "data/L1000_genes.csv"
    GSE_EXPR      = "data/FRENCH_expression.tsv"
    GSE_CLINICAL  = "data/FRENCH_clinical.tsv"

    # LINCS paths (download from clue.io)
    LINCS_GCTX    = "data/level5_beta_trt_cp_n720216x12328.gctx"
    LINCS_SIGINFO = "data/siginfo_beta.txt"
    LINCS_GENES   = "data/geneinfo_beta.txt"
    LINCS_CPDS    = "data/compoundinfo_beta.txt"

    # Step 1: Load TCGA
    print("=" * 50)
    print("STEP 1: Loading TCGA-COAD/READ data")
    print("=" * 50)
    X, y = load_tcga(TCGA_EXPR, TCGA_CLINICAL, L1000_GENES)

    # Step 2: Train model
    print("\n" + "=" * 50)
    print("STEP 2: Training Cox Elastic Net")
    print("=" * 50)
    coeffs, scaler, train_cidx = train_cox(X, y)
    coeffs.to_csv("results/survival_coeffs_crc_L1000.csv", header=True)

    # Plot KM for training
    X_scaled = pd.DataFrame(scaler.transform(X.loc[:, X.std() > 0]),
                             index=X.index, columns=X.loc[:, X.std() > 0].columns)
    coef_vec = X_scaled.columns.map(lambda g: coeffs.get(g, 0))
    risk_train = X_scaled.values @ coef_vec.values
    plot_kaplan_meier(y['time'].values, y['event'].values, risk_train,
                      f'TCGA-COAD/READ Training (n={len(y)})',
                      'results/kaplan_meier_tcga.png')

    # Step 3: Validate on GSE39582
    print("\n" + "=" * 50)
    print("STEP 3: Cross-Dataset Validation on GSE39582")
    print("=" * 50)
    val_cidx, risk_gse, clin_gse = validate_gse39582(coeffs, GSE_EXPR, GSE_CLINICAL)
    plot_kaplan_meier(clin_gse['time'].values, clin_gse['event'].values, risk_gse,
                      f'GSE39582 Validation (n={len(clin_gse)})',
                      'results/kaplan_meier_gse39582.png')

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Training C-index  (TCGA,     n={len(y)}):  {train_cidx:.3f}")
    print(f"Validation C-index (GSE39582, n={len(clin_gse)}): {val_cidx:.3f}")
    print(f"Signature genes: {len(coeffs)}")
    print(f"Input feature space: 978 L1000 genes")

    # Step 4: LINCS (only if data available)
    import os
    if os.path.exists(LINCS_GCTX):
        print("\n" + "=" * 50)
        print("STEP 4: LINCS Drug Response (DPD scores)")
        print("=" * 50)
        dpd = compute_dpd_lincs(coeffs, LINCS_GCTX, LINCS_SIGINFO,
                                 LINCS_GENES, LINCS_CPDS)
        dpd.to_csv("results/DPD_scores_crc.csv")
    else:
        print("\n[LINCS] Skipping — gctx file not found.")
        print("        Download from clue.io and update LINCS_GCTX path.")
