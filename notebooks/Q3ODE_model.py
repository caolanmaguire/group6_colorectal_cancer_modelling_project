"""
ODE-Based Survival Model for Colorectal Cancer (TCGA COAD/READ)
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


# 0. Covariate list (shared by fetch, fit, predict)

COVARIATES = [
    "age", "stage", "msi_score", "sex_male", "site_rectum", "subtype_msi",
    "expr_pc1", "expr_pc2",
]

CRC_GENE_PANEL = [
    "APC", "TP53", "KRAS", "NRAS", "BRAF", "PIK3CA", "PTEN", "SMAD4",
    "SMAD2", "FBXW7", "CTNNB1", "TCF7L2", "MYC", "EGFR", "ERBB2", "MET",
    "VEGFA", "CDKN2A", "TGFBR2", "MLH1", "MSH2", "MSH6", "PMS2",
]


# ----------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------

def fetch_tcga_coadread_clinical(out_csv="tcga_coadread_clinical.csv", study_id="coadread_tcga_pan_can_atlas_2018"):
   
    import requests

    base = "https://www.cbioportal.org/api"

    print(f"Fetching patient list for study '{study_id}' ...")
    r = requests.get(f"{base}/studies/{study_id}/patients", timeout=60)
    if r.status_code != 200:
        raise RuntimeError(
            f"Could not list patients for study '{study_id}' "
            f"(HTTP {r.status_code}): {r.text[:300]}\n"
            f"-> Check the study still exists at "
            f"https://www.cbioportal.org/study?id={study_id}"
        )
    patients = pd.DataFrame(r.json())
    patient_ids = patients["patientId"].unique().tolist()
    print(f"Found {len(patient_ids)} patients.")

   
    r = requests.get(f"{base}/studies/{study_id}/clinical-attributes", timeout=60)
    available_attrs = set()
    if r.status_code == 200:
        available_attrs = {a["clinicalAttributeId"] for a in r.json()}
    else:
        print(f"WARNING: could not list clinical-attributes (HTTP {r.status_code}); "
              f"falling back to a fixed attribute list.")

    def pick_attr(candidates):
        for c in candidates:
            if not available_attrs or c in available_attrs:
                return c
        return None

    attr_map = {
        "OS_MONTHS": ["OS_MONTHS"],
        "OS_STATUS": ["OS_STATUS"],
        "AGE": ["AGE"],
        "STAGE": ["AJCC_PATHOLOGIC_TUMOR_STAGE", "TUMOR_STAGE"],
        "SEX": ["SEX", "GENDER"],
        "MSI": ["MSI_SCORE_MANTIS"],
        "SUBTYPE": ["SUBTYPE"],
        "SITE": ["TUMOR_SITE", "PRIMARY_SITE_PATIENT", "ANATOMIC_ORGAN_SUBDIVISION"],
    }
    resolved = {k: pick_attr(v) for k, v in attr_map.items()}
    print(f"Resolved clinical attribute IDs: {resolved}")
    attribute_ids = [v for v in resolved.values() if v is not None]

    identifiers = [{"entityId": pid, "studyId": study_id} for pid in patient_ids]

    print("Fetching clinical attributes (this can take ~10-30s for ~600 patients)...")
    r = requests.post(
        f"{base}/clinical-data/fetch",
        params={"clinicalDataType": "PATIENT", "projection": "SUMMARY"},
        json={"attributeIds": attribute_ids, "identifiers": identifiers},
        timeout=180,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"clinical-data/fetch failed (HTTP {r.status_code}): {r.text[:500]}\n"
            f"-> The cBioPortal API schema may have changed. Inspect the live "
            f"schema at https://www.cbioportal.org/api/swagger-ui/index.html "
            f"under 'Clinical Data' -> POST /clinical-data/fetch, and adjust "
            f"the payload in fetch_tcga_coadread_clinical() accordingly."
        )
    long_df = pd.DataFrame(r.json())
    if long_df.empty or "clinicalAttributeId" not in long_df.columns:
        raise RuntimeError(
            "clinical-data/fetch returned no usable data. Raw response head:\n"
            f"{r.text[:500]}\n"
            "-> Check available attributes with:\n"
            f"  GET {base}/studies/{study_id}/clinical-attributes"
        )
    wide = long_df.pivot_table(
        index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first"
    ).reset_index()

    def col(resolved_key):
        name = resolved.get(resolved_key)
        if name is not None and name in wide.columns:
            return wide[name]
        return pd.Series(np.nan, index=wide.index)

    df = pd.DataFrame()
    df["patient_id"] = wide["patientId"]
    df["time_days"] = pd.to_numeric(col("OS_MONTHS"), errors="coerce") * 30.44
    df["event"] = col("OS_STATUS").astype(str).str.startswith("1").astype(int)
    df["age"] = pd.to_numeric(col("AGE"), errors="coerce")

    stage_map = {
        "STAGE I": 1, "STAGE IA": 1, "STAGE IB": 1,
        "STAGE II": 2, "STAGE IIA": 2, "STAGE IIB": 2, "STAGE IIC": 2,
        "STAGE III": 3, "STAGE IIIA": 3, "STAGE IIIB": 3, "STAGE IIIC": 3,
        "STAGE IV": 4, "STAGE IVA": 4, "STAGE IVB": 4, "STAGE IVC": 4,
    }
    df["stage"] = col("STAGE").astype(str).str.upper().map(stage_map)
    df["msi_score"] = pd.to_numeric(col("MSI"), errors="coerce")

    sex_raw = col("SEX").astype(str).str.upper()
    df["sex_male"] = np.where(sex_raw.str.startswith("MALE"), 1,
                       np.where(sex_raw.str.startswith("FEMALE"), 0, np.nan))

    site_raw = col("SITE").astype(str).str.upper()
    has_site = site_raw.notna() & (site_raw != "NAN")
    df["site_rectum"] = np.where(has_site, site_raw.str.contains("RECTU").astype(float), np.nan)

    subtype_raw = col("SUBTYPE").astype(str).str.upper()
    has_subtype = subtype_raw.notna() & (subtype_raw != "NAN")
    df["subtype_msi"] = np.where(has_subtype, subtype_raw.str.contains("MSI").astype(float), np.nan)

    df = df.dropna(subset=["time_days", "event", "age", "stage"])
    df = df[df["time_days"] > 0]
    if df.empty:
        raise RuntimeError(
            "After parsing, 0 patients had complete data (time_days, event, age, "
            "stage). Inspect the raw 'wide' dataframe columns/values above to see "
            "what the API actually returned, and adjust attribute IDs or the "
            "stage_map as needed."
        )

    # Expression-derived PCA covariate (best-effort; skipped gracefully on failure)
    expr_df = fetch_expression_pca(study_id, df["patient_id"].tolist(), base)
    if expr_df is not None:
        df = df.merge(expr_df, on="patient_id", how="left")
    else:
        print("NOTE: proceeding without expression PCA covariates.")

    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} patients to {out_csv}")
    return df


def fetch_expression_pca(study_id, patient_ids, base, n_components=2):
   
    import requests

    try:
        # 1. Find an mRNA expression (z-score, if available) molecular profile
        r = requests.get(f"{base}/studies/{study_id}/molecular-profiles", timeout=60)
        r.raise_for_status()
        profiles = pd.DataFrame(r.json())
        mrna = profiles[profiles["molecularAlterationType"] == "MRNA_EXPRESSION"]
        if mrna.empty:
            print("NOTE: no MRNA_EXPRESSION molecular profile found for this study; "
                  "skipping expression PCA covariate.")
            return None
        zscore = mrna[mrna["molecularProfileId"].str.contains("Zscores", case=False, na=False)]
        profile_id = (zscore.iloc[0] if not zscore.empty else mrna.iloc[0])["molecularProfileId"]
        print(f"Using expression profile: {profile_id}")

        # 2. Map gene symbols -> Entrez IDs
        r = requests.post(
            f"{base}/genes/fetch",
            params={"geneIdType": "HUGO_GENE_SYMBOL", "projection": "SUMMARY"},
            json=CRC_GENE_PANEL, timeout=60,
        )
        r.raise_for_status()
        genes = pd.DataFrame(r.json())
        if genes.empty:
            print("NOTE: gene symbol lookup returned nothing; skipping expression PCA.")
            return None
        entrez_ids = genes["entrezGeneId"].dropna().astype(int).unique().tolist()

        # 3. Map patients -> primary-tumor sample IDs
        r = requests.get(f"{base}/studies/{study_id}/samples", timeout=60)
        r.raise_for_status()
        samples = pd.DataFrame(r.json())
        if "sampleType" in samples.columns:
            primary = samples[samples["sampleType"].str.contains("Primary", case=False, na=False)]
            primary = primary if not primary.empty else samples
        else:
            primary = samples
        sample_map = primary.drop_duplicates("patientId").set_index("patientId")["sampleId"].to_dict()
        sample_ids = [sample_map[p] for p in patient_ids if p in sample_map]
        if len(sample_ids) < 20:
            print(f"NOTE: only {len(sample_ids)} samples with expression data found; "
                  f"skipping expression PCA (too few for a stable PCA).")
            return None

        # 4. Fetch expression values
        print(f"Fetching expression data for {len(entrez_ids)} genes x "
              f"{len(sample_ids)} samples...")
        r = requests.post(
            f"{base}/molecular-profiles/{profile_id}/molecular-data/fetch",
            params={"projection": "SUMMARY"},
            json={"entrezGeneIds": entrez_ids, "sampleIds": sample_ids},
            timeout=180,
        )
        r.raise_for_status()
        records = pd.DataFrame(r.json())
        if records.empty:
            print("NOTE: expression fetch returned no data; skipping expression PCA.")
            return None

        wide = records.pivot_table(index="patientId", columns="entrezGeneId", values="value", aggfunc="first")
        wide = wide.dropna(axis=1, thresh=int(0.5 * len(wide)))  # drop genes >50% missing
        wide = wide.fillna(wide.median())
        if wide.shape[1] < 2 or wide.shape[0] < 20:
            print("NOTE: insufficient expression data after cleaning; skipping expression PCA.")
            return None

        # 5. PCA via SVD (avoids an extra sklearn dependency)
        X = wide.to_numpy(dtype=float)
        Xc = X - X.mean(axis=0)
        Xc = Xc / (X.std(axis=0) + 1e-8)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        n_comp = min(n_components, S.shape[0])
        scores = U[:, :n_comp] * S[:n_comp]
        explained = (S ** 2) / np.sum(S ** 2)
        print(f"Expression PCA explained variance: "
              f"{[f'{v:.1%}' for v in explained[:n_comp]]}")

        pc_df = pd.DataFrame(
            scores, columns=[f"expr_pc{i + 1}" for i in range(n_comp)], index=wide.index
        ).reset_index().rename(columns={"patientId": "patient_id"})
        return pc_df

    except Exception as e:
        print(f"NOTE: expression PCA fetch failed ({type(e).__name__}: {e}); "
              f"continuing without it.")
        return None


def make_demo_data(n=400, seed=0):
    """Synthetic COAD-like dataset so the pipeline can be tested with no internet."""
    rng = np.random.default_rng(seed)
    age = rng.normal(66, 12, n).clip(25, 95)
    stage = rng.choice([1, 2, 3, 4], size=n, p=[0.25, 0.30, 0.25, 0.20])
    msi_score = rng.normal(0.3, 0.15, n).clip(0, 1)
    sex_male = rng.choice([0, 1], size=n, p=[0.47, 0.53])
    site_rectum = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    subtype_msi = (msi_score > 0.4).astype(float)
    expr_pc1 = rng.normal(0, 1, n)
    expr_pc2 = rng.normal(0, 1, n)

    # ground-truth Weibull-Cox hazard used to generate survival times
    beta_true = np.array([0.03, 0.55, -0.8, 0.1, 0.15, -0.3, 0.25, -0.05])
    X = np.column_stack([
        age - 60, stage - 1, msi_score, sex_male, site_rectum,
        subtype_msi, expr_pc1, expr_pc2,
    ])
    lp = X @ beta_true
    lambda0, k = 0.0008, 1.3
    u = rng.uniform(0, 1, n)
    true_time = (-np.log(u) / (lambda0 * np.exp(lp))) ** (1 / k)

    censor_time = rng.uniform(200, 3000, n)
    time_days = np.minimum(true_time, censor_time)
    event = (true_time <= censor_time).astype(int)

    return pd.DataFrame({
        "time_days": time_days, "event": event,
        "age": age, "stage": stage, "msi_score": msi_score,
        "sex_male": sex_male, "site_rectum": site_rectum,
        "subtype_msi": subtype_msi, "expr_pc1": expr_pc1, "expr_pc2": expr_pc2,
    })


# ----------------------------------------------------------------------
# 2. ODE survival model
# ----------------------------------------------------------------------

def hazard(t, lin_pred, lambda0, k):
    """Weibull baseline hazard modulated by covariates (proportional hazards)."""
    t = max(t, 1e-8)
    return lambda0 * k * (t ** (k - 1)) * np.exp(lin_pred)


def survival_ode_rhs(t, S, lin_pred, lambda0, k):
    h = hazard(t, lin_pred, lambda0, k)
    return [-h * S[0]]


def solve_survival(t_max, lin_pred, lambda0, k, t_eval=None):
    """Integrate dS/dt = -h(t) S(t) from 0 to t_max, return S(t_eval)."""
    if t_eval is None:
        t_eval = np.array([t_max])
    sol = solve_ivp(
        survival_ode_rhs, (0, t_max), y0=[1.0],
        args=(lin_pred, lambda0, k),
        t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-9,
    )
    return sol.y[0]


def negative_log_likelihood(theta, X, time, event):
    
    log_lambda0, log_k, *beta = theta
    lambda0, k = np.exp(log_lambda0), np.exp(log_k)
    beta = np.array(beta)

    t = np.clip(time, 1e-6, None)
    lin_preds = X @ beta
    cum_hazard = lambda0 * np.exp(lin_preds) * t ** k          # H(t) = -log S(t)
    log_S = -cum_hazard
    log_h = np.log(lambda0) + np.log(k) + (k - 1) * np.log(t) + lin_preds

    log_lik = event * log_h + log_S
    return -np.sum(log_lik)


def sanity_check_ode(n_checks=5, seed=1):
    """Confirm the closed-form S(t) used for fast fitting matches the
    numerical ODE solver (solve_survival) to within tight tolerance."""
    rng = np.random.default_rng(seed)
    lambda0, k = 5e-4, 1.2
    for _ in range(n_checks):
        lp = rng.normal(0, 1)
        t = rng.uniform(50, 2000)
        S_ode = solve_survival(t, lp, lambda0, k)[-1]
        S_closed = np.exp(-lambda0 * np.exp(lp) * t ** k)
        assert abs(S_ode - S_closed) < 1e-4, (S_ode, S_closed)
    print(f"ODE solver vs closed-form check passed ({n_checks} random cases).")


def fit_ode_survival_model(df, covariates=COVARIATES, verbose=True):
    X = df[covariates].to_numpy(dtype=float)
    X = (X - X.mean(0)) / X.std(0)  # standardize for stable optimization
    time = df["time_days"].to_numpy(dtype=float)
    event = df["event"].to_numpy(dtype=int)

    theta0 = np.array([np.log(1e-4), np.log(1.1)] + [0.0] * X.shape[1])

    result = minimize(
        negative_log_likelihood, theta0, args=(X, time, event),
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-5, "adaptive": True},
    )
    if verbose:
        print("Optimization success:", result.success, "| NLL:", result.fun)

    log_lambda0, log_k, *beta = result.x
    fitted = {
        "lambda0": np.exp(log_lambda0),
        "k": np.exp(log_k),
        "beta": np.array(beta),
        "covariates": covariates,
        "X_mean": df[covariates].mean().to_numpy(),
        "X_std": df[covariates].std().to_numpy(),
        "raw_result": result,
    }
    return fitted


# ----------------------------------------------------------------------
# 3. Prediction / evaluation
# ----------------------------------------------------------------------

def predict_survival_curve(fitted, patient_row, t_grid):
    x = np.array([patient_row[c] for c in fitted["covariates"]], dtype=float)
    x = (x - fitted["X_mean"]) / fitted["X_std"]
    lin_pred = x @ fitted["beta"]
    S = solve_survival(t_grid[-1], lin_pred, fitted["lambda0"], fitted["k"], t_eval=t_grid)
    return S


def predict_risk_score(fitted, df):
    """Linear predictor (higher = higher hazard = worse prognosis)."""
    X = df[fitted["covariates"]].to_numpy(dtype=float)
    X = (X - fitted["X_mean"]) / fitted["X_std"]
    return X @ fitted["beta"]


def concordance_index_manual(time, event, risk_score):
    """C-index: fraction of comparable pairs correctly ordered by risk."""
    n = len(time)
    concordant, comparable = 0, 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if event[i] == 1 and time[i] < time[j]:
                comparable += 1
                if risk_score[i] > risk_score[j]:
                    concordant += 1
                elif risk_score[i] == risk_score[j]:
                    concordant += 0.5
    return concordant / comparable if comparable else np.nan


def fit_and_report_coxph(train_df, test_df, covariates):
    
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        print("\n(Skipping CoxPH baseline comparison -- run 'pip install lifelines' to enable it.)")
        return

    cols = covariates + ["time_days", "event"]
    cph = CoxPHFitter(penalizer=0.1)  
    cph.fit(train_df[cols], duration_col="time_days", event_col="event")

    print("\n--- CoxPH baseline (lifelines) for comparison ---")
    summary = cph.summary[["coef", "exp(coef)", "p"]]
    for name, row in summary.iterrows():
        print(f"  {name}: coef={row['coef']:.4f}  HR={row['exp(coef)']:.3f}  p={row['p']:.4f}")

    c_index = cph.score(test_df[cols], scoring_method="concordance_index")
    print(f"  CoxPH held-out C-index: {c_index:.3f}")
    print("  (compare to the ODE model's C-index printed above -- they should be similar)")


# ----------------------------------------------------------------------
# 4. Main pipeline
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ODE survival model for TCGA COAD/READ")
    parser.add_argument("--csv", type=str, default=None, help="path to clinical CSV")
    parser.add_argument("--fetch", action="store_true", help="fetch data from cBioPortal")
    parser.add_argument("--demo", action="store_true", help="use synthetic demo data")
    args = parser.parse_args()

    sanity_check_ode()

    if args.fetch:
        df = fetch_tcga_coadread_clinical()
    elif args.csv:
        df = pd.read_csv(args.csv)
    else:
        print("No --csv/--fetch given, using synthetic demo data (--demo).")
        df = make_demo_data()

    print("\nMissing-value counts per column:")
    for c in ["time_days", "event"] + COVARIATES:
        n_missing = df[c].isna().sum() if c in df.columns else len(df)
        print(f"  {c}: {n_missing} / {len(df)} missing"
              + ("  <- column not found in data!" if c not in df.columns else ""))

    # time_days and event are essential -- drop rows missing those.
    df = df.dropna(subset=["time_days", "event"]).reset_index(drop=True)


    usable_covariates = []
    for c in COVARIATES:
        if c not in df.columns or df[c].isna().all():
            print(f"WARNING: dropping covariate '{c}' -- entirely missing in this dataset.")
            continue
        if df[c].isna().any():
            med = df[c].median()
            n_imputed = df[c].isna().sum()
            df[c] = df[c].fillna(med)
            print(f"NOTE: imputed {n_imputed} missing '{c}' values with median ({med:.3g}).")
        usable_covariates.append(c)

    if not usable_covariates:
        raise RuntimeError(
            "No usable covariates remain after checking for missing data. "
            "Inspect the CSV directly (e.g. tcga_coadread_clinical.csv) and "
            "update COVARIATES near the top of this script to match columns "
            "that are actually populated."
        )
    if usable_covariates != COVARIATES:
        print(f"Using covariates: {usable_covariates} (COVARIATES list trimmed due to missing data)")

    df = df.dropna(subset=usable_covariates).reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError(
            "0 patients remain after cleaning. Check tcga_coadread_clinical.csv "
            "directly to see what was actually returned by the API."
        )
    print(f"\nLoaded {len(df)} patients | events: {df['event'].sum()} "
          f"({100*df['event'].mean():.1f}% observed deaths)")

    # train / test split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_train = int(0.75 * len(df))
    train_df = df.iloc[idx[:n_train]].reset_index(drop=True)
    test_df = df.iloc[idx[n_train:]].reset_index(drop=True)

    print("\nFitting ODE survival model on training set...")
    fitted = fit_ode_survival_model(train_df, covariates=usable_covariates)
    print(f"\nFitted parameters:")
    print(f"  lambda0 = {fitted['lambda0']:.6g}")
    print(f"  k       = {fitted['k']:.4f}")
    for name, b in zip(fitted["covariates"], fitted["beta"]):
        print(f"  beta[{name}] = {b:.4f}  (hazard ratio per +1 SD = {np.exp(b):.3f})")

    # evaluate discrimination on held-out patients
    risk_test = predict_risk_score(fitted, test_df)
    c_index = concordance_index_manual(
        test_df["time_days"].to_numpy(), test_df["event"].to_numpy(), risk_test
    )
    print(f"\nHeld-out concordance index (C-index): {c_index:.3f}")
    print("(0.5 = no better than chance, 1.0 = perfect ranking of risk)")

    fit_and_report_coxph(train_df, test_df, usable_covariates)

    if len(test_df) == 0:
        print("\nNo held-out patients available -- skipping example prediction/plot.")
        return
    # example: predict a single patient's survival curve
    example_patient = test_df.iloc[0]
    t_grid = np.linspace(1, 3000, 200)
    S_curve = predict_survival_curve(fitted, example_patient, t_grid)
    print(f"\nExample patient predicted 5-year (1825-day) survival probability: "
          f"{np.interp(1825, t_grid, S_curve):.3f}")

    # optional plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        for stage_val, color in zip([1, 2, 3, 4], ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]):
            row = example_patient.copy()
            if "stage" in fitted["covariates"]:
                row["stage"] = stage_val
            S = predict_survival_curve(fitted, row, t_grid)
            plt.plot(t_grid / 30.44, S, label=f"Stage {stage_val}", color=color)
        plt.xlabel("Months since diagnosis")
        plt.ylabel("Predicted survival probability S(t)")
        plt.title("ODE-model predicted survival by stage (other covariates fixed)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("ode_survival_curves.png", dpi=150)
        print("\nSaved plot to ode_survival_curves.png")
    except ImportError:
        pass


if __name__ == "__main__":
    sys.exit(main())
