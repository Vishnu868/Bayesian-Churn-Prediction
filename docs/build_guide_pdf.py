"""Build docs/Project_Guide.pdf: the whole project explained, in simple words and in technical detail.

    python scripts/make_figures.py && python scripts/make_guide_figures.py && python docs/build_guide_pdf.py

Every number is read from results/*/metrics.json, so the guide always matches the committed results.
"""
import json
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, Spacer, Table,
                                TableStyle)
from reportlab.platypus.doctemplate import BaseDocTemplate, Frame, PageTemplate
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "docs", "Project_Guide.pdf")

# ------------------------------------------------------------------ fonts & styles
F, FF = "/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/truetype/freefont/"
pdfmetrics.registerFont(TTFont("DV", F + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", F + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", FF + "FreeSansOblique.ttf"))
pdfmetrics.registerFont(TTFont("DV-BI", FF + "FreeSansBoldOblique.ttf"))
pdfmetrics.registerFont(TTFont("DVM", F + "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK, MUTED = colors.HexColor("#1F2328"), colors.HexColor("#57606A")
BLUE, GREEN = colors.HexColor("#2F5D9E"), colors.HexColor("#2E7D4F")
SIMPLE_BG, TECH_BG, NOTE_BG = colors.HexColor("#EAF5EE"), colors.HexColor("#EEF2FA"), colors.HexColor("#FFF6E0")

S = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=24, leading=30, alignment=TA_CENTER, textColor=INK),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=12.5, leading=18, alignment=TA_CENTER, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=17, leading=22, spaceBefore=4, spaceAfter=8, textColor=BLUE),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=12.5, leading=17, spaceBefore=10, spaceAfter=4, textColor=INK,
                        keepWithNext=1),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.6, leading=14.2, spaceAfter=5, textColor=INK),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9.6, leading=14, leftIndent=14, bulletIndent=3,
                             spaceAfter=2, textColor=INK),
    "box": ParagraphStyle("box", fontName="DV", fontSize=9.4, leading=13.8, textColor=INK),
    "boxh": ParagraphStyle("boxh", fontName="DV-B", fontSize=9.4, leading=13, spaceAfter=2),
    "eq": ParagraphStyle("eq", fontName="DV", fontSize=10.2, leading=15, alignment=TA_CENTER, spaceBefore=3,
                         spaceAfter=6, textColor=INK),
    "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=8.5, leading=11.5, alignment=TA_CENTER, textColor=MUTED,
                          spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.4, leading=11.2, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=8.4, leading=11.2, textColor=INK),
    "code": ParagraphStyle("code", fontName="DVM", fontSize=7.9, leading=10.4, textColor=INK),
    "toc1": ParagraphStyle("toc1", fontName="DV", fontSize=10.5, leading=17, leftIndent=6),
}

story = []
_h1 = [0]


def P(text, style="body"):
    story.append(Paragraph(text, S[style]))


def H1(text):
    _h1[0] += 1
    story.append(PageBreak())
    p = Paragraph(f"{_h1[0]}. {text}", S["h1"])
    p._toc = f"{_h1[0]}. {text}"
    story.append(p)


def H2(text):
    story.append(Paragraph(text, S["h2"]))


def BUL(items):
    for it in items:
        story.append(Paragraph(it, S["bullet"], bulletText="•"))


def _callout(title, body, bg, edge, title_color):
    paras = [Paragraph(title, ParagraphStyle("t", parent=S["boxh"], textColor=title_color))]
    for b in (body if isinstance(body, list) else [body]):
        paras.append(Paragraph(b, S["box"]))
    t = Table([[paras]], colWidths=[16.6 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(t)
    story.append(Spacer(1, 7))


def SIMPLE(body):
    _callout("In simple words", body, SIMPLE_BG, GREEN, GREEN)


def TECH(body):
    _callout("Technical detail", body, TECH_BG, BLUE, BLUE)


def NOTE(body, title="Important"):
    _callout(title, body, NOTE_BG, colors.HexColor("#C9971C"), colors.HexColor("#8A6510"))


def EQ(text):
    P(text, "eq")


def CODE(text):
    t = Table([[Preformatted(text, S["code"])]], colWidths=[16.6 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6F8FA")),
                           ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(t)
    story.append(Spacer(1, 7))


def TABLE(rows, widths, header=True, bold_rows=()):
    data = [[Paragraph(str(c), S["cellb"] if (header and r == 0) or r in bold_rows else S["cell"])
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D7DE")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7ECF3")))
    for r in bold_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#EEF2FA")))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 8))


COMPACT = os.environ.get("GUIDE_COMPACT")      # e.g. GUIDE_COMPACT=1: JPEG figures for a < 3 MB copy


def _compact(path):
    """Re-encode a PNG figure as a JPEG (white background) in a temp dir, for a small PDF."""
    import tempfile

    from PIL import Image as PILImage
    out = os.path.join(tempfile.gettempdir(), "guide_compact_" + os.path.basename(path).replace(".png", ".jpg"))
    im = PILImage.open(path).convert("RGBA")
    bg = PILImage.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[3])
    if bg.width > 1800:
        bg = bg.resize((1800, round(bg.height * 1800 / bg.width)), PILImage.LANCZOS)
    bg.save(out, "JPEG", quality=80, optimize=True)
    return out


def FIGURE(rel, caption, width=16.4):
    path = os.path.join(ROOT, rel)
    img = Image(_compact(path) if COMPACT else path)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width * cm, width * cm * ratio
    story.append(KeepTogether([img, Spacer(1, 3), Paragraph(caption, S["cap"])]))


def J(rel):
    with open(os.path.join(ROOT, "results", rel)) as f:
        return json.load(f)


# ------------------------------------------------------------------ results
MAIN, LEG = J("main/metrics.json"), J("legacy_with_customerID/metrics.json")
CV, PRI = J("cv/metrics.json"), J("prior_sensitivity/metrics.json")["rows"]
LOO = J("prior_sensitivity/metrics.json")["loo_compare"]
DSZ, SYN = J("data_size/metrics.json")["runs"], J("synthetic_sanity_check/metrics.json")["summary"]
EDA = J("dataset_summary.json")
POST = pd.read_csv(os.path.join(ROOT, "results/main/posterior_summary.csv"), index_col=0)
POH = pd.read_csv(os.path.join(ROOT, "results/main/posterior_summary_onehot.csv"), index_col=0)
TM, TF, CI = MAIN["test_metrics"], MAIN["test_metrics_no_pca"], MAIN["bootstrap_95ci"]
INFO, DIAG, UNC = MAIN["info"], MAIN["bayes_diagnostics"], MAIN["uncertainty"]
B, LR, RF = "Bayesian LR (HMC)", "Logistic Regression", "Random Forest"
BOH, LRF, RFF = "Bayesian LR (one-hot, no PCA)", "Logistic Regression (full, no PCA)", "Random Forest (full, no PCA)"
CVS = CV["summary"]


def f4(x):
    return f"{x:.4f}"


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def cvpm(model, met):
    return f"{CVS[model][met]['mean']:.4f} ± {CVS[model][met]['std']:.4f}"


def ci(model, met):
    lo, hi = CI[model][met]
    return f"[{lo:.3f}, {hi:.3f}]"


DS = pd.DataFrame([{"n": r["n"], "b_auc": r["bayes"]["ROC-AUC"], "m_auc": r["mle_lr"]["ROC-AUC"],
                    "b_ll": r["bayes"]["LogLoss"], "m_ll": r["mle_lr"]["LogLoss"], "w": r["hdi_width_mean"]}
                   for r in DSZ]).groupby("n").mean()
PRI_PCA = [r for r in PRI if r["features"].startswith("PCA")]
PRI_OH = [r for r in PRI if r["features"].startswith("one-hot")][0]


def ohrow(name):
    r = POH.loc[f"beta[{name}]"]
    return r["mean"], r["hdi_3%"], r["hdi_97%"]


# ================================================================== COVER
story.append(Spacer(1, 3.0 * cm))
P("Bayesian Churn Prediction", "title")
P("Customer churn with uncertainty: Bayesian logistic regression (HMC / NUTS), PCA and classical baselines", "sub")
story.append(Spacer(1, 0.8 * cm))
P("<b>The Complete Project Guide</b><br/>Every part of the project from start to end, explained in simple words "
  "and in technical detail", "sub")
story.append(Spacer(1, 1.2 * cm))
P("Team 12 · Department of Artificial Intelligence<br/>Amrita Vishwa Vidyapeetham, Coimbatore", "sub")
story.append(Spacer(1, 0.4 * cm))
P("Dodla Nithya Sai · M. Vishnu Vardhan · P. Sreekar Reddy · S. Beena Reddy", "sub")
story.append(Spacer(1, 1.6 * cm))
TABLE([["", ""],
       ["Data", f"<b>Real</b> IBM Telco Customer Churn: {INFO['n_customers']:,} customers, {INFO['n_features']} "
                f"features, {pct(INFO['churn_rate'])} churn"],
       ["Main model", f"Bayesian logistic regression on {INFO['pca_components']} PCA components, NUTS, "
                      f"{DIAG['chains']} chains × {DIAG['draws_per_chain']} draws"],
       ["Headline result", f"Test ROC-AUC {f4(TM[B]['ROC-AUC'])} (95% CI {ci(B, 'ROC-AUC')}), accuracy "
                           f"{pct(TM[B]['Accuracy'])} vs {pct(TM['Majority class']['Accuracy'])} for always "
                           f"predicting 'stay'"],
       ["Convergence", f"max R-hat {DIAG['max_rhat']:.3f}, min bulk ESS {DIAG['min_ess_bulk']:.0f}, "
                       f"{DIAG['divergences']} divergences"],
       ["How to read this guide", "Each topic has a green <b>In simple words</b> box (analogies, no maths) and a "
                                  "blue <b>Technical detail</b> box (equations, settings, code)."]],
      [3.6, 13.0], header=False)

story.append(PageBreak())
P("Contents", "h1")
toc = TableOfContents()
toc.levelStyles = [S["toc1"]]
story.append(toc)

# ================================================================== 1. At a glance
H1("The project at a glance")
SIMPLE([
    "A phone and internet company loses customers every month. This is called <b>churn</b>. Keeping a customer "
    "is much cheaper than winning a new one, so the company wants to know <i>who is about to leave</i> and offer "
    "them something (a discount, a better contract) before they go.",
    "We built a program that reads each customer's details (contract type, how long they have been a customer, "
    "monthly bill, which services they use...) and outputs a <b>probability of leaving</b>. The twist: our main "
    "model is <b>Bayesian</b>. A normal model says <i>\"72% chance of leaving\"</i>. Ours says <i>\"72%, and I'm "
    "fairly sure it's between 66% and 78%\"</i>. That range tells the retention team how much to trust each "
    "number.",
])
TECH([
    "Binary classification on the IBM Telco Customer Churn data (7,043 rows → 7,032 after cleaning). Pipeline: "
    "label encoding → z-score standardisation → PCA keeping ≥95% variance (16 components) → Bayesian logistic "
    "regression with N(0, 1) priors, sampled with the No-U-Turn Sampler (PyMC 5, 4 chains × 1,000 draws after "
    "1,000 tuning steps). Baselines: majority class, base-rate prior, scikit-learn logistic regression, random "
    "forest (300 trees). Evaluation: stratified 80/20 split plus 5-fold cross-validation, bootstrap 95% CIs, "
    "calibration (Brier, ECE, reliability), posterior diagnostics (R-hat, ESS, divergences, BFMI, trace, rank, "
    "energy), posterior predictive checks, and per-customer 94% highest-density intervals.",
])
FIGURE("docs/figures/guide_pipeline.png", "Figure 1.1 The full pipeline, from the raw CSV to the report.")
H2("Key findings in one table")
TABLE([["Question", "Answer (from the real data)"],
       ["Does the model beat chance?", f"Yes. ROC-AUC {f4(TM[B]['ROC-AUC'])} vs 0.5; accuracy {pct(TM[B]['Accuracy'])} "
                                       f"vs {pct(TM['Majority class']['Accuracy'])} for 'nobody churns'."],
       ["Is Bayesian better than logistic regression?",
        f"Not in accuracy: they are practically identical (AUC {f4(TM[B]['ROC-AUC'])} vs {f4(TM[LR]['ROC-AUC'])}). "
        "With 5,625 customers the N(0,1) prior barely matters. The Bayesian model <i>adds</i> uncertainty, "
        "it does not add accuracy."],
       ["Is the uncertainty useful?",
        f"Yes. On the most confident half of test customers accuracy is {pct(UNC['acc_most_confident_half'])}; on the "
        f"least confident half it is {pct(UNC['acc_least_confident_half'])}."],
       ["Does PCA help?", f"No. It costs accuracy: logistic regression on all features gets AUC "
                          f"{f4(TF[LRF]['ROC-AUC'])} vs {f4(TM[LR]['ROC-AUC'])} on PCA components."],
       ["Best model overall", f"Bayesian LR on one-hot features (no PCA): AUC {f4(TF[BOH]['ROC-AUC'])}, F1 "
                              f"{f4(TF[BOH]['F1'])}, with readable coefficients."],
       ["Do the original README numbers reproduce?",
        "Yes, but only when customerID is (wrongly) used as a feature. See Chapter 15."]],
      [5.0, 11.6])

# ================================================================== 2. Problem
H1("The business problem: customer churn")
SIMPLE([
    "Think of a gym. Every month some members stop coming and cancel. If the gym could spot the people who are "
    "drifting away, a trainer could call them, offer a free session, and keep them. Churn prediction is that "
    "early-warning system for a telecom company.",
    "Two kinds of mistakes are possible. A <b>false alarm</b> (we think someone will leave but they would have "
    "stayed) wastes a small discount. A <b>miss</b> (we think someone will stay but they leave) loses the whole "
    "customer. Because the costs are different, the company wants a <i>probability</i>, not just yes/no, and it "
    "helps to know <i>how sure</i> that probability is.",
])
TECH([
    "Given features x<sub>i</sub> of customer i, estimate p<sub>i</sub> = P(churn<sub>i</sub> = 1 | x<sub>i</sub>, D). "
    "The decision \"offer an incentive\" is made when the expected saving exceeds its cost. A point estimate "
    "p̂<sub>i</sub> ignores estimation error. A Bayesian posterior p(p<sub>i</sub> | D) also gives that error, so "
    "risk-aware rules become possible (for example \"act only if the lower end of the 94% HDI is above t\").",
    "Problem statement (from the original project): traditional classifiers give a single probability. This project "
    "gives a full posterior distribution over every parameter and every prediction, so the retention team can make "
    "risk-aware decisions.",
])
H2("Objectives")
BUL(["Clean and encode the Telco churn data and compress the features with PCA (≥95% variance).",
     "Build a Bayesian logistic regression and learn its parameters with Hamiltonian Monte Carlo (NUTS).",
     "Benchmark it against logistic regression and a random forest (plus trivial baselines).",
     "Quantify the uncertainty of every prediction (posterior mean and 94% HDI) and check that it is meaningful.",
     "Verify that the sampler converged (R-hat, ESS, divergences, trace, rank and energy plots).",
     "Generate an automated HTML report and make the whole project reproducible from one script."])

# ================================================================== 3. Dataset
H1("The dataset: IBM Telco Customer Churn")
SIMPLE([
    "The data is a spreadsheet with one row per customer of a fictional California telecom company. "
    "There are 7,043 customers and 21 columns. The last column, <b>Churn</b>, says whether the customer left in the "
    "last month (Yes/No). About 1 in 4 customers left.",
    "It is real, public data. IBM published it as a sample, and it is the standard teaching and benchmarking "
    "dataset for churn prediction (Kaggle: <i>blastchar/telco-customer-churn</i>). We did <b>not</b> make up any data "
    "for the main results.",
])
TECH([
    "File <font face='DVM'>WA_Fn-UseC_-Telco-Customer-Churn.csv</font> (970,457 bytes, 7,043 rows × 21 columns). "
    "Downloaded by <font face='DVM'>scripts/download_data.py</font> from IBM's public GitHub mirror. The download "
    "resumes after an interruption (HTTP Range), retries with back-off, writes to a <font face='DVM'>.part</font> "
    "file, and checks the header and row count before the file is moved into place. "
    "sha256 16320c9c1ec7…5e91.",
])
TABLE([["Group", "Columns", "Values"],
       ["Identifier", "customerID", "unique string. Not a feature (see Ch. 15)"],
       ["Demographics", "gender, SeniorCitizen, Partner, Dependents", "Male/Female, 0/1, Yes/No, Yes/No"],
       ["Account", "tenure, Contract, PaperlessBilling, PaymentMethod",
        "0–72 months; month-to-month / one year / two year; Yes/No; 4 payment methods"],
       ["Charges", "MonthlyCharges, TotalCharges", "$18.25–$118.75; total billed so far"],
       ["Phone", "PhoneService, MultipleLines", "Yes/No; Yes/No/No phone service"],
       ["Internet", "InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, "
                    "StreamingMovies", "DSL / Fiber optic / No; each add-on Yes/No/No internet service"],
       ["Target", "Churn", "Yes (left in the last month) / No"]], [2.6, 7.4, 6.6])
FIGURE("docs/figures/dataset_overview.png",
       "Figure 3.1 Real data: class balance, churn rate by contract, internet service and payment method, and the "
       "tenure and monthly-charge distributions of stayers vs churners (dashed line = overall churn rate).")
H2("What the data already tells us")
cb = EDA["churn_by_contract"]
ib = EDA["churn_by_internet"]
BUL([f"<b>Contract matters most:</b> month-to-month customers churn {pct(cb['Month-to-month'], 0)}, one-year "
     f"{pct(cb['One year'], 0)}, two-year only {pct(cb['Two year'], 0)}.",
     f"<b>New customers leave:</b> median tenure is {EDA['tenure_median_churn_vs_stay'][0]:.0f} months for churners "
     f"vs {EDA['tenure_median_churn_vs_stay'][1]:.0f} for stayers.",
     f"<b>Fiber-optic customers churn more</b> ({pct(ib['Fiber optic'], 0)} vs {pct(ib['DSL'], 0)} for DSL), even "
     "though fibre is the premium product. This is likely a price or quality issue.",
     "<b>Electronic-check payers churn about 45%</b>, roughly three times the rate of automatic payers."])

# ================================================================== 4. Preprocessing
H1("Preprocessing")
H2("4.1 Cleaning")
SIMPLE("Eleven customers have an empty TotalCharges cell. They all have tenure 0: they joined this month and "
       "have not been billed yet. We remove these 11 rows (0.16% of the data), leaving 7,032 customers. We also "
       "turn Churn into a number: No → 0, Yes → 1.")
TECH(["<font face='DVM'>data.clean()</font>: TotalCharges is stored as text, so blank strings become NaN with "
      "<font face='DVM'>pd.to_numeric(errors='coerce')</font>. Those rows are dropped (7,043 → 7,032), Churn is mapped "
      "to {0,1}, and customerID is dropped. <font face='DVM'>keep_id=True</font> keeps it only to reproduce the original "
      "(buggy) run."])
H2("4.2 Encoding categories as numbers")
SIMPLE("Models do arithmetic, so words like \"Fiber optic\" must become numbers. <b>Label encoding</b> gives each "
       "category a code: DSL=0, Fiber optic=1, No=2. It is compact (one column per question) and is what the "
       "original project used. For the extra interpretable model we use <b>one-hot encoding</b>: one yes/no column "
       "per category, like ticking boxes on a form.")
TECH(["Label encoding (sorted categories, same as sklearn's LabelEncoder) gives <b>19 numeric features</b>. "
      "It imposes an artificial order (DSL &lt; Fiber &lt; No), which a linear model on raw codes cannot undo. "
      "PCA mixes the codes linearly, so it inherits this order.",
      "One-hot (drop-first) gives 30 columns, but 6 are exact duplicates: \"No internet service\" in each of the "
      "6 add-on columns equals InternetService_No, and \"No phone service\" equals 1 − PhoneService_Yes. The design "
      "matrix has rank 24 &lt; 30. This made NUTS slow (168 s, tree depth 7.6). We fold those levels into \"No\", "
      "which gives <b>23 columns with full rank</b> (condition number 82). Sampling then takes 54 s with tree depth 6.1."])
H2("4.3 Train/test split and standardisation")
SIMPLE("We hide 20% of customers in a locked drawer (the <b>test set</b>) and only look at them at the very end. "
       "It's like keeping the real exam questions away from a student while they study. We also rescale every "
       "feature so it has average 0 and spread 1. Otherwise \"TotalCharges\" (in thousands) would shout over "
       "\"SeniorCitizen\" (0 or 1).")
TECH(["Stratified split, <font face='DVM'>test_size=0.2, random_state=42</font>: 5,625 train / 1,407 test, both with "
      f"{pct(INFO['train_churn_rate'])} churn. z = (x − μ)/σ, with μ and σ estimated on the <b>training split only</b> "
      "(<font face='DVM'>FeaturePipeline.fit</font>). The test set is transformed with the training statistics, "
      "so no information leaks from test to train. A unit test checks this."])

# ================================================================== 5. PCA
H1("Dimensionality reduction with PCA")
SIMPLE(["Many features say almost the same thing. TotalCharges is roughly tenure × MonthlyCharges, and the "
        "streaming columns move together. PCA is like photographing a 3-D object from the angle that shows the "
        "most: it rotates the data so the first new axis captures the most variation, the second the next most, "
        "and so on. We keep the fewest axes that still hold 95% of the information.",
        f"Result: 19 features → <b>{INFO['pca_components']} components</b> holding "
        f"<b>{pct(INFO['pca_variance_retained'])}</b> of the variance."])
FIGURE("docs/figures/guide_pca.png", "Figure 5.1 PCA on a toy 2-D example: PC1 points along the direction of "
                                     "greatest spread, PC2 is perpendicular to it.", width=7)
TECH(["Given the standardised training matrix Z (n × d), PCA computes the eigen-decomposition of the covariance "
      "C = ZᵀZ/(n−1) = VΛVᵀ. The scores are T = Z V<sub>k</sub>, where V<sub>k</sub> holds the eigenvectors of the "
      "k largest eigenvalues. k is the smallest number with Σ<sub>j≤k</sub> λ<sub>j</sub> / Σ<sub>j</sub> "
      "λ<sub>j</sub> ≥ 0.95 (<font face='DVM'>PCA(n_components=0.95, svd_solver='full')</font>).",
      "Why it is used here: the components are uncorrelated, which makes the posterior geometry nicer for HMC (see "
      "the posterior correlation plot), and the model has fewer parameters. The cost is interpretability (a "
      "coefficient on PC2 has no business meaning) and some accuracy, because the components are chosen to "
      "explain the <i>features</i>, not churn. Loadings are saved in <font face='DVM'>results/main/pca_loadings.csv</font>.",
      "Reading the two most important components from their loadings: <b>PC1 ≈ loyalty</b> (TotalCharges +0.43, "
      "tenure +0.42, Contract +0.32), with posterior coefficient −0.49, so loyal customers churn less. <b>PC2 ≈ "
      "expensive, paperless, senior</b> (MonthlyCharges +0.47, PaperlessBilling +0.38, SeniorCitizen +0.31, "
      "Contract −0.31), with coefficient +0.85, the strongest churn driver."])
FIGURE("results/main/figures/pca_variance.png", "Figure 5.2 Real data: explained variance per component and "
                                                "cumulative. 16 components pass the 95% threshold.", width=13)

# ================================================================== 6. Logistic regression
H1("Logistic regression: the base model")
SIMPLE(["Imagine a scorecard. Each feature adds or subtracts points: a month-to-month contract adds points, a long "
        "tenure subtracts points. The total score can be any number. The <b>sigmoid</b> function squashes that "
        "score into a probability between 0 and 1: a very negative score gives about 0%, zero gives 50%, and a "
        "large positive score gives about 100%.",
        "Learning the model means finding the right number of points for each feature. These numbers are the "
        "<b>coefficients</b> β."])
FIGURE("docs/figures/guide_sigmoid.png", "Figure 6.1 The sigmoid link maps the linear score (log-odds) to a "
                                         "probability. The red dot is the learned intercept.", width=11)
TECH(["η<sub>i</sub> = β<sub>0</sub> + Σ<sub>j</sub> β<sub>j</sub> z<sub>ij</sub>,   "
      "p<sub>i</sub> = σ(η<sub>i</sub>) = 1 / (1 + e<sup>−η<sub>i</sub></sup>),   y<sub>i</sub> ~ Bernoulli(p<sub>i</sub>)",
      "Log-likelihood: ℓ(β) = Σ<sub>i</sub> [ y<sub>i</sub> log p<sub>i</sub> + (1 − y<sub>i</sub>) log(1 − p<sub>i</sub>) ]. "
      "Classical LR maximises ℓ(β) − λ‖β‖²/2 (sklearn default: L2 penalty with C = 1/λ = 1). "
      "e<sup>β<sub>j</sub></sup> is the <b>odds ratio</b> for a one-unit (here one-SD) increase in feature j.",
      "Classical LR gives one best β̂. It reports no uncertainty unless you add standard errors from the Hessian, "
      "and those rely on a large-sample normal approximation."])

# ================================================================== 7. Bayesian thinking
H1("Bayesian thinking: prior, likelihood, posterior")
SIMPLE(["A doctor sees a patient with a cough. <b>Before</b> any test, the doctor has a belief (\"most coughs are "
        "colds\"). This is the <b>prior</b>. A test result arrives; how likely that result would be under each "
        "illness is the <b>likelihood</b>. Combining both gives an updated belief, the <b>posterior</b>. More "
        "evidence makes the belief sharper.",
        "Our model does the same with its coefficients. It starts with a mild belief that every effect is "
        "probably small (between −2 and +2 on the log-odds scale). It then reads the customers and ends up with a "
        "<i>range</i> of plausible values for every coefficient, not one number."])
FIGURE("docs/figures/guide_bayes_update.png", "Figure 7.1 Real data, simplest possible model (churn rate only): "
       "the posterior narrows from the vague prior to a sharp peak at 26.6% as more customers are observed.")
TECH(["Bayes' theorem: p(β | D) = p(D | β) p(β) / p(D), i.e. posterior ∝ likelihood × prior. "
      "The evidence p(D) = ∫ p(D | β) p(β) dβ has no closed form for logistic regression, so we cannot write the "
      "posterior down. We draw samples from it instead (MCMC, next chapter).",
      "Link to classical LR: the maximum a posteriori (MAP) estimate with prior N(0, s²) is exactly L2-penalised "
      "logistic regression with λ = 1/s². With s = 1 and C = 1 the two models optimise the same objective. "
      "That is why their predictions are almost identical here. The Bayesian model differs by keeping the "
      "<b>whole distribution</b> around the MAP, not just the peak."])

# ================================================================== 8. Model
H1("The Bayesian logistic regression model")
SIMPLE("Our model is logistic regression where every coefficient is a random quantity with a bell-curve prior "
       "centred at 0. After seeing the data we have 4,000 plausible versions of the model. To predict a customer "
       "we ask all 4,000 versions and look at the spread of their answers.")
EQ("β<sub>0</sub> ~ N(0, 1),   β<sub>j</sub> ~ N(0, 1)  (j = 1…16)")
EQ("p<sub>i</sub> = σ(β<sub>0</sub> + Σ<sub>j</sub> β<sub>j</sub> z<sub>ij</sub>),   y<sub>i</sub> ~ Bernoulli(p<sub>i</sub>)")
TECH(["Graphical model: β → p<sub>i</sub> ← z<sub>i</sub>, p<sub>i</sub> → y<sub>i</sub>, repeated over the 5,625 "
      "training customers. Only <b>parameter learning</b> is needed, because the structure (features → churn) is "
      "fixed. The 17 unknowns are β<sub>0</sub>…β<sub>16</sub>.",
      "Why N(0, 1)? The inputs are standardised, so a coefficient of ±2 already means the odds change by a factor "
      "of e² ≈ 7.4 per SD. N(0,1) puts 95% of prior mass in [−1.96, 1.96]. It is weakly informative: it rules out "
      "absurd effects but is easily overwhelmed by 5,625 observations. Chapter 15 tests σ ∈ {0.01, 0.1, 1, 10}.",
      "Implementation (<font face='DVM'>src/models.py</font>) uses <font face='DVM'>logit_p=</font> instead of "
      "<font face='DVM'>p=</font> for numerical stability. The original code also stored "
      "<font face='DVM'>p</font> as a Deterministic for every training row, which bloats the trace by "
      "4,000 × 5,625 numbers; this version computes predictions from the coefficient samples instead."])
CODE("""with pm.Model(coords={"feature": names}) as model:
    Xd   = pm.Data("X", X)
    b0   = pm.Normal("intercept", mu=0.0, sigma=1.0)
    beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims="feature")
    pm.Bernoulli("y", logit_p=b0 + pm.math.dot(Xd, beta), observed=y)
    idata = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.9, random_seed=42)""")

# ================================================================== 9. HMC
H1("Sampling the posterior: MCMC, HMC and NUTS")
SIMPLE(["The posterior is a landscape: high ground where coefficient values fit the data well, low ground where "
        "they don't. We want to explore it so that we spend time in each region in proportion to its height.",
        "A <b>random walk</b> is like a blindfolded hiker taking small random steps. It gets there eventually, but "
        "slowly, especially along a narrow ridge. <b>Hamiltonian Monte Carlo</b> is like a skateboarder in a bowl: "
        "give them a random push and physics carries them smoothly along the curved surface to a far-away point. "
        "<b>NUTS</b> (No-U-Turn Sampler) decides automatically how long each ride should be: it stops when the "
        "skateboarder starts coming back."])
FIGURE("docs/figures/guide_hmc.png", "Figure 9.1 Illustration on a correlated 2-D Gaussian. Random-walk "
       "Metropolis barely moves in 60 steps. HMC trajectories cross the whole distribution in 8 iterations.")
TECH(["Treat β as a position and add an auxiliary momentum p ~ N(0, M). The Hamiltonian is",
      "<b>H(β, p) = U(β) + K(p)</b>,  U(β) = −log p(β | D) (up to a constant),  K(p) = ½ pᵀM⁻¹p.",
      "Hamilton's equations dβ/dt = M⁻¹p, dp/dt = −∇U(β) are integrated with the <b>leapfrog</b> scheme "
      "(step size ε): p ← p − (ε/2)∇U(β);  β ← β + εM⁻¹p;  p ← p − (ε/2)∇U(β). The end point is accepted with "
      "probability min(1, exp(H<sub>old</sub> − H<sub>new</sub>)). Energy is nearly conserved, so acceptance is "
      "high even for distant proposals.",
      "<b>NUTS</b> (Hoffman &amp; Gelman, 2014) builds the trajectory by repeated doubling until it makes a U-turn "
      "(the trajectory starts to double back on itself), then samples a point from it. This removes the need "
      "to hand-tune the number of steps L. "
      "PyMC adapts ε (dual averaging to reach <font face='DVM'>target_accept=0.9</font>) and a diagonal mass "
      "matrix M during the 1,000 tuning steps. These are then discarded.",
      f"This run: {DIAG['chains']} chains in parallel, mean step size {DIAG['mean_step_size']:.3f}, mean tree depth "
      f"{DIAG['mean_tree_depth']:.2f} (≈2<sup>3</sup> = 8 leapfrog steps), {DIAG['sampling_time_s']:.0f} s on 4 CPU cores."])

# ================================================================== 10. Diagnostics
H1("Did the sampler work? Convergence diagnostics")
SIMPLE("We run 4 independent explorers (chains) from different random starting points. If they all draw the "
       "same map of the landscape, we trust the map. If one explorer is stuck in a different valley, the map is "
       "wrong. The diagnostics are numerical and visual checks that the 4 maps agree.")
TABLE([["Diagnostic", "What it checks", "Rule of thumb", "This run"],
       ["R-hat (split, rank-normalised)", "between-chain vs within-chain variance", "&lt; 1.01",
        f"max {DIAG['max_rhat']:.4f}"],
       ["Bulk ESS", "effective independent draws (centre of the distribution)", "&gt; 400",
        f"min {DIAG['min_ess_bulk']:.0f} of 4,000"],
       ["Tail ESS", "effective draws in the 5%/95% tails", "&gt; 400", f"min {DIAG['min_ess_tail']:.0f}"],
       ["Divergences", "leapfrog blew up (posterior has regions the sampler cannot handle)", "0",
        f"{DIAG['divergences']}"],
       ["E-BFMI", "is the momentum resampling exploring energy levels well?", "&gt; 0.3", f"min {DIAG['bfmi_min']:.2f}"]],
      [3.6, 6.2, 2.6, 4.2])
TECH(["R̂ = √( V̂ / W ), where W is the mean within-chain variance and V̂ = (n−1)/n · W + B/n mixes in the "
      "between-chain variance B. Values near 1 mean the chains agree. ArviZ uses the rank-normalised split-R̂ "
      "(Vehtari et al., 2021). ESS = S / (1 + 2 Σ<sub>t</sub> ρ<sub>t</sub>), where ρ<sub>t</sub> is the "
      "autocorrelation at lag t. NUTS draws are nearly independent, so ESS ≈ 3,400 out of 4,000.",
      "All checks pass with a wide margin. This is expected: logistic regression with a Gaussian prior has a "
      "log-concave posterior (a single smooth hill), the easiest case for HMC."])
FIGURE("results/main/figures/trace.png", "Figure 10.1 Trace plots. Left: the 4 chains' density estimates "
       "overlap for every coefficient. Right: the chains look like stationary 'fuzzy caterpillars' with no trends.")
FIGURE("results/main/figures/rank.png", "Figure 10.2 Rank plots. If the chains mix well, each chain's ranks are "
       "uniformly spread (points close to the dashed line).")
FIGURE("results/main/figures/energy.png", "Figure 10.3 Energy plot. The marginal energy and the energy transition "
       "distributions overlap, so HMC explores well (E-BFMI ≈ 1).", width=8.5)
FIGURE("results/main/figures/posterior_corr.png", "Figure 10.4 Posterior correlations between coefficients. "
       "Because PCA components are uncorrelated, the coefficients are nearly independent (pale colours).", width=8.5)
FIGURE("results/main/figures/forest.png", "Figure 10.5 Forest plot: posterior mean and 94% HDI of each "
       "coefficient (4 chains combined).", width=8.5)

# ================================================================== 11. Uncertainty
H1("From parameters to predictions: uncertainty")
SIMPLE(["For one customer we ask all 4,000 sampled models for a churn probability. We get 4,000 answers. "
        "Their average is our prediction. The narrowest range that holds 94% of them is the <b>94% HDI</b> "
        "(highest-density interval). A narrow range means \"the data has told us a lot about customers like this\". "
        "A wide range means \"few similar customers, so be careful\".",
        "If the range crosses 50% (for example 42%–58%), the model honestly can't say whether this customer will "
        "leave. Those are the cases where a human should look."])
FIGURE("docs/figures/guide_hdi.png", "Figure 11.1 For a skewed distribution the HDI is shorter than the "
       "equal-tailed interval. Every point inside the HDI is more plausible than every point outside.", width=12)
TECH(["Posterior predictive probability for customer *: p<sub>*</sub><sup>(s)</sup> = σ(β<sub>0</sub><sup>(s)</sup> + "
      "β<sup>(s)</sup>·z<sub>*</sub>), s = 1…4,000. Point prediction E[p<sub>*</sub> | D] ≈ mean over s. "
      "HDI: sort the samples and take the shortest window containing ⌊0.94·S⌋ of them (<font face='DVM'>models.hdi</font>, "
      "vectorised over customers). This is <b>epistemic</b> (parameter) uncertainty. The outcome itself is still a "
      "coin flip with probability p<sub>*</sub>, which is <b>aleatoric</b> uncertainty.",
      f"Real-data results (1,407 test customers): mean HDI width {UNC['hdi_width_mean']:.3f} (≈ ±{UNC['hdi_width_mean'] / 2 * 100:.1f} "
      f"percentage points), median {UNC['hdi_width_median']:.3f}, 90th percentile {UNC['hdi_width_p90']:.3f}, max "
      f"{UNC['hdi_width_max']:.3f}. {pct(UNC['frac_hdi_straddles_0.5'])} of test HDIs straddle 0.5.",
      f"Is the uncertainty meaningful? Ranking customers by |mean − 0.5|/sd: accuracy is "
      f"{pct(UNC['acc_most_confident_half'])} on the most confident half vs {pct(UNC['acc_least_confident_half'])} on the "
      f"least confident half. Where the HDI straddles 0.5, accuracy is {pct(UNC['acc_hdi_straddles_0.5'])} (a coin flip); "
      f"elsewhere it is {pct(UNC['acc_hdi_clear_of_0.5'])}."])
FIGURE("results/main/figures/uncertainty_band.png", "Figure 11.2 Every test customer, sorted by predicted churn "
       "probability, with the 94% HDI band. Red ticks are customers who actually churned.")
FIGURE("results/main/figures/uncertainty_analysis.png", "Figure 11.3 Left: intervals are widest for mid-range "
       "probabilities. Centre: distribution of widths. Right: selective prediction. Keeping only the most "
       "confident customers raises accuracy steadily.")
NOTE("Most of the 'confident vs unsure' signal here comes from |mean − 0.5|, which any probabilistic classifier "
     "also has. With 5,625 training customers the epistemic part is small (±4 points). The Bayesian intervals "
     "matter much more with little data (Chapter 15.4) or for rare customer profiles.", "Honest caveat")
H2("Posterior predictive check")
SIMPLE("A model that has learnt the data should be able to <i>re-create</i> data that looks like the real thing. "
       "We let each of the sampled models simulate churn for the training customers and compare the simulated "
       "churn rate with the real one.")
TECH(f"For 2,000 posterior draws, y<sub>rep</sub> ~ Bernoulli(p<sup>(s)</sup>) for all training customers. The 94% "
     f"interval of the simulated churn rate is [{UNC['ppc_train_interval94'][0]:.3f}, {UNC['ppc_train_interval94'][1]:.3f}] "
     f"and the observed rate is {UNC['ppc_train_observed']:.3f}, well inside it. On the test set: "
     f"[{UNC['ppc_test_interval94'][0]:.3f}, {UNC['ppc_test_interval94'][1]:.3f}] vs observed {UNC['ppc_test_observed']:.3f}.")
FIGURE("results/main/figures/ppc_train.png", "Figure 11.4 Posterior predictive check of the churn rate (training data).",
       width=11)

# ================================================================== 12. Baselines
H1("Baselines: what we compare against")
SIMPLE("A result only means something next to a reference. 'Always say nobody leaves' already gets 73% accuracy, "
       "because 73% of customers stay. So 77% accuracy is not as impressive as it sounds. ROC-AUC is the fairer "
       "score here.")
TABLE([["Baseline", "How it works", "Why include it"],
       ["Majority class", "predicts 'stay' for everyone", "accuracy floor (73.4%)"],
       ["Prior (base rate)", "predicts P = 26.6% for everyone", "no-skill probabilistic floor (Brier 0.195, AUC 0.5)"],
       ["Logistic Regression", "sklearn, L2, C=1, same PCA inputs", "the classical twin of the Bayesian model"],
       ["Random Forest", "300 trees, default depth, same PCA inputs", "non-linear model with feature interactions"],
       ["LR / RF on all features", "no PCA, standardised label codes", "measures what PCA costs"],
       ["Bayesian LR (one-hot)", "N(0,1) prior on 23 one-hot/numeric features, NUTS", "interpretable Bayesian model"]],
      [3.6, 6.4, 6.6])
TECH("Random forest: bootstrap-aggregated decision trees, each split on a random subset of √d features. The "
     "probability is the mean of the trees' leaf class frequencies. It is not calibrated by construction. "
     "All models see exactly the same split and the same fitted scaler/PCA.")

# ================================================================== 13. Metrics
H1("How we measure success: metrics")
SIMPLE(["<b>Accuracy</b>: % of customers classified correctly. <b>Precision</b>: of the people we flagged, how many "
        "really left. <b>Recall</b>: of the people who left, how many we caught. <b>F1</b>: a balance of precision "
        "and recall.",
        "<b>ROC-AUC</b>: pick one churner and one stayer at random. How often does the model give the churner the "
        "higher score? 0.5 = coin flip, 1.0 = perfect.",
        "<b>Calibration</b>: when the model says 30%, do about 30% of those customers actually leave? A weather "
        "forecaster who says '30% rain' should see rain on 3 out of 10 such days."])
TECH(["With TP, FP, TN, FN at threshold 0.5: Accuracy = (TP+TN)/N; Precision = TP/(TP+FP); Recall = TP/(TP+FN); "
      "F1 = 2PR/(P+R). ROC-AUC = P(score<sub>churner</sub> &gt; score<sub>stayer</sub>), threshold-free. PR-AUC "
      "(average precision) is better when positives are rare (chance level = 0.266).",
      "Brier = (1/N) Σ (p<sub>i</sub> − y<sub>i</sub>)². Log-loss = −(1/N) Σ [y log p + (1−y) log(1−p)]. "
      "ECE = Σ<sub>b</sub> (n<sub>b</sub>/N) |ȳ<sub>b</sub> − p̄<sub>b</sub>| over 10 equal-width bins.",
      "Uncertainty of the metrics themselves: (1) 1,000 bootstrap resamples of the test set, with the same indices "
      "for all models so differences are paired; (2) stratified 5-fold cross-validation over all 7,032 customers, "
      "with scaler and PCA re-fitted inside every fold."])

# ================================================================== 14. Results
H1("Results on the real data")
H2("14.1 Hold-out test set (1,407 customers)")
rows = [["Model", "Acc", "Prec", "Rec", "F1", "ROC-AUC", "Brier", "ECE"]]
order = ["Majority class", "Prior (base rate)", LR, RF, B]
for k in order:
    r = TM[k]
    rows.append([k] + [f4(r[c]) for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Brier", "ECE"]])
for k in [LRF, RFF, BOH]:
    r = TF[k]
    rows.append([k] + [f4(r[c]) for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Brier", "ECE"]])
TABLE(rows, [5.0, 1.5, 1.5, 1.5, 1.5, 1.8, 1.6, 1.6], bold_rows=(5, 8))
P(f"95% bootstrap CI of ROC-AUC: Bayesian (PCA) {ci(B, 'ROC-AUC')}, LR {ci(LR, 'ROC-AUC')}, RF {ci(RF, 'ROC-AUC')}, "
  f"Bayesian one-hot {ci(BOH, 'ROC-AUC')}. Paired difference RF − Bayesian: "
  f"{CI[RF]['dAUC_vs_' + B][1]:+.4f} [{CI[RF]['dAUC_vs_' + B][0]:+.4f}, {CI[RF]['dAUC_vs_' + B][2]:+.4f}]; "
  f"one-hot Bayesian − PCA Bayesian: {CI[BOH]['dAUC_vs_' + B][1]:+.4f} "
  f"[{CI[BOH]['dAUC_vs_' + B][0]:+.4f}, {CI[BOH]['dAUC_vs_' + B][2]:+.4f}].")
FIGURE("results/main/figures/roc_pr.png", "Figure 14.1 ROC and precision-recall curves. The LR (orange) curve is "
       "hidden under the Bayesian (blue) curve: they are practically identical.")
FIGURE("results/main/figures/calibration.png", "Figure 14.2 Reliability diagram: LR and Bayesian lie on the "
       "diagonal (well calibrated); the random forest is over-confident at high probabilities.")
FIGURE("results/main/figures/confusion_matrices.png", "Figure 14.3 Confusion matrices at threshold 0.5.")
H2("14.2 Five-fold cross-validation (all 7,032 customers)")
rows = [["Model", "Accuracy", "F1", "ROC-AUC", "Brier"]]
for k in [LR, RF, B, LRF, RFF, BOH]:
    rows.append([k] + [cvpm(k, c) for c in ["Accuracy", "F1", "ROC-AUC", "Brier"]])
TABLE(rows, [5.4, 2.8, 2.8, 2.8, 2.8], bold_rows=(3, 6))
FIGURE("results/cv/cv_boxplot.png", "Figure 14.4 ROC-AUC and F1 across the 5 CV folds (each dot is one fold).")
SIMPLE(["All the reasonable models are close. The Bayesian model and ordinary logistic regression give "
        "the same answers, because with thousands of customers the gentle prior hardly changes anything. The "
        "random forest is a little worse here. Skipping PCA and using the original features helps a bit more "
        "than any choice of model."])
H2("14.3 What drives churn? (interpretable Bayesian model)")
fo, fo_lo, fo_hi = ohrow("InternetService_Fiber optic")
te, te_lo, te_hi = ohrow("tenure")
c2, c2_lo, c2_hi = ohrow("Contract_Two year")
mc, mc_lo, mc_hi = ohrow("MonthlyCharges")
P(f"The PCA coefficients are hard to interpret, so we also fitted the same Bayesian model on the 23 one-hot "
  f"features (standardised). Coefficients are log-odds per one standard deviation of the feature. Tenure: "
  f"{te:+.2f} [94% HDI {te_lo:+.2f}, {te_hi:+.2f}]. Two-year contract: {c2:+.2f} [{c2_lo:+.2f}, {c2_hi:+.2f}]. "
  f"Fiber optic: {fo:+.2f} [{fo_lo:+.2f}, {fo_hi:+.2f}]. MonthlyCharges: {mc:+.2f} [{mc_lo:+.2f}, {mc_hi:+.2f}]. "
  "MonthlyCharges has a very wide interval because it is almost a linear combination of the service columns, and "
  "the posterior honestly reports that the data cannot separate their effects.")
FIGURE("results/main/figures/onehot_coefficients.png", "Figure 14.5 Posterior coefficients of the one-hot Bayesian "
       "model (94% HDI). Red = raises churn, blue = lowers churn, grey = the HDI includes zero.", width=13)

# ================================================================== 15. Further experiments
H1("Further experiments and the original results")
H2("15.1 Results reported in the original project (README)")
NOTE("These numbers are copied from the original README. They were <b>not</b> produced by this rebuild. They "
     "are listed so they can be compared with the reproduced numbers below.", "Original numbers")
TABLE([["Model (original README)", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
       ["Logistic Regression", "0.7939", "0.6265", "0.5561", "0.5892", "0.8342"],
       ["Random Forest", "0.7854", "0.6216", "0.4920", "0.5493", "0.8238"],
       ["Bayesian HMC", "0.7704", "0.5794", "0.4973", "0.5353", "0.8204"]], [5.0, 2.3, 2.3, 2.3, 2.3, 2.3])
P("Original setup as stated: 7,032 customers, '20 features', PCA 20 → 17 components with 97.2% variance, "
  "intercept −1.6266 ± 0.0502, PC1 −0.4907, PC2 0.8556, PC3 0.1181, PC4 0.1733, PC5 −0.2820.")
H2("15.2 Reproducing them: the customerID finding")
LT, LP = LEG["test_metrics"], pd.read_csv(os.path.join(ROOT, "results/legacy_with_customerID/posterior_summary.csv"), index_col=0)
TECH([f"The cleaned data has only <b>19</b> features. The original '20 features / 17 components / 97.2%' is "
      f"reproduced exactly when <font face='DVM'>customerID</font> (a unique string, label-encoded to 0…7031) is left "
      f"in as a feature: we get {LEG['info']['n_features']} features → {LEG['info']['pca_components']} components, "
      f"{pct(LEG['info']['pca_variance_retained'])}. With that setting the Bayesian model gives Acc "
      f"{f4(LT[B]['Accuracy'])}, P {f4(LT[B]['Precision'])}, R {f4(LT[B]['Recall'])}, F1 {f4(LT[B]['F1'])}, AUC "
      f"{f4(LT[B]['ROC-AUC'])}, and the posterior is intercept {LP.loc['intercept', 'mean']:.4f} ± "
      f"{LP.loc['intercept', 'sd']:.4f}, PC1 {LP.loc['beta[PC1]', 'mean']:.4f}, PC2 {LP.loc['beta[PC2]', 'mean']:.4f}, "
      f"PC5 {LP.loc['beta[PC5]', 'mean']:.4f}. These match the README to 2–3 decimals.",
      f"The README's 'Logistic Regression' row (AUC 0.8342) matches LR on all features <i>without</i> PCA "
      f"(rebuild: {f4(TF[LRF]['ROC-AUC'])}), not LR on PCA components. The original comparison therefore probably "
      f"set PCA-based Bayesian against non-PCA LR. The README's RF row (AUC 0.8238) did not reproduce exactly (legacy "
      f"rebuild {f4(LT[RF]['ROC-AUC'])}), most likely because of a different sklearn version or settings.",
      "An ID number carries no information about churn. Using it as a feature adds a noise dimension that PCA "
      "must keep, so the corrected pipeline drops it. The effect on accuracy is negligible (AUC "
      f"{f4(LT[B]['ROC-AUC'])} vs {f4(TM[B]['ROC-AUC'])}), but it is still a bug."])
H2("15.3 Prior sensitivity and LOO model comparison")
rows = [["Features", "prior σ", "ROC-AUC", "Log-loss", "elpd_loo ± SE", "mean HDI width", "mean |β|"]]
for r in PRI:
    rows.append([r["features"], f"{r['prior_sigma']:g}", f4(r["ROC-AUC"]), f4(r["LogLoss"]),
                 f"{r['elpd_loo']:.1f} ± {r['elpd_loo_se']:.1f}", f"{r['hdi_width_mean']:.3f}", f"{r['beta_abs_mean']:.3f}"])
TABLE(rows, [3.4, 1.6, 1.9, 1.9, 3.0, 2.6, 2.2])
SIMPLE("A very strong prior (σ = 0.01: 'all effects are basically zero') squashes the model and hurts it. "
       "Anything from σ = 0.1 to σ = 10 gives the same answers, because the 5,625 customers outvote the prior. "
       "So the results do not depend on our particular choice of prior.")
TECH("elpd_loo is the expected log predictive density estimated by Pareto-smoothed importance-sampling "
     "leave-one-out cross-validation (Vehtari, Gelman &amp; Gabry, 2017). Higher is better. p_loo is the effective "
     "number of parameters. " + f"In a paired comparison (az.compare) the one-hot model beats the PCA model by "
     f"Δelpd = {LOO['PCA (16 comps)']['elpd_diff']:.1f} ± {LOO['PCA (16 comps)']['dse']:.1f} (paired SE), i.e. more "
     "than 5 standard errors, consistent with its better test and CV scores. A strong prior (σ = 0.1) slightly "
     "raises F1 at threshold 0.5 because it pulls probabilities towards the base rate, but it worsens log-loss and "
     "elpd, so it is not a better model.")
H2("15.4 When does Bayes help? Training-set size")
rows = [["Train customers", "Bayes AUC", "≈MLE LR AUC", "Bayes log-loss", "≈MLE log-loss", "mean 94% HDI width"]]
for n, r in DS.iterrows():
    rows.append([f"{n:,}", f4(r.b_auc), f4(r.m_auc), f4(r.b_ll), f4(r.m_ll), f"{r.w:.3f}"])
TABLE(rows, [2.8, 2.3, 2.5, 2.8, 2.8, 3.4])
FIGURE("results/data_size/data_size.png", "Figure 15.1 Sub-sampled training sets (3 repeats each), fixed test "
       "set. With few customers the prior protects the Bayesian model, while unregularised LR over-fits (huge "
       "log-loss). The intervals shrink as data grows.")
SIMPLE("With only 50–100 customers the Bayesian model is much safer: plain maximum-likelihood logistic regression "
       "becomes over-confident and its probabilities become badly wrong. The Bayesian model's interval widths "
       "also show that it knows it is unsure. With thousands of customers the two agree. This is where the "
       "Bayesian approach earns its keep.")
H2("15.5 SYNTHETIC sanity check: recovering known coefficients")
NOTE(f"This section uses <b>synthetic data</b> generated by the program, NOT real customers. It only checks that "
     f"the sampler is implemented correctly: on data with known true coefficients, about 94% of the 94% HDIs "
     f"should contain the truth.", "Synthetic data")
P(f"{SYN['reps']} replicate datasets (n = {SYN['n_train']}, d = {SYN['d']}): the 94% HDIs contained the true "
  f"coefficient in <b>{pct(SYN['param_hdi94_coverage'])}</b> of {SYN['n_params_checked']} cases (target 94%). Mean "
  f"RMSE of the posterior means {SYN['rmse_mean']:.3f}. Test AUC {SYN['test_auc_mean']:.3f} vs oracle (true "
  f"coefficients) {SYN['oracle_auc_mean']:.3f}. Max R-hat {SYN['max_rhat']:.3f}, {SYN['divergences_total']} divergences.")
FIGURE("results/synthetic_sanity_check/synthetic_recovery.png", "Figure 15.2 SYNTHETIC data only: posterior "
       "intervals vs true coefficients (left) and posterior means vs truth across all replicates (right).")

# ================================================================== 16. Discussion
H1("Discussion, limitations and future work")
H2("What we learned")
BUL([f"<b>The model works on real data:</b> ROC-AUC ≈ {CVS[B]['ROC-AUC']['mean']:.2f} in cross-validation, far "
     "above chance (0.5), with well-calibrated probabilities (ECE ≈ 0.02).",
     "<b>Bayesian ≈ classical in accuracy.</b> With N(0,1) priors and 5,625 customers, the posterior mean equals "
     "L2-regularised LR to within 0.004 per coefficient. The Bayesian value is the uncertainty, not extra accuracy.",
     "<b>The uncertainty is informative:</b> confident predictions are much more accurate than unsure ones, and "
     "with small data the Bayesian model stays calibrated while maximum-likelihood LR does not.",
     "<b>PCA is optional.</b> It gives uncorrelated inputs and fewer parameters but slightly lowers accuracy and "
     "removes interpretability. The one-hot Bayesian model is both the most accurate and the most readable.",
     "<b>Calibration:</b> the random forest is the least calibrated model and also the least accurate here."])
H2("Limitations")
BUL(["One public dataset from one (fictional) company, a single snapshot in time. There is no time-to-churn "
     "information, so survival models are not possible.",
     "Threshold 0.5 is not cost-optimal. With churn at 27%, a lower threshold would raise recall; the right "
     "threshold depends on the cost of an incentive vs the value of a customer.",
     "Label encoding imposes an artificial order on nominal categories before PCA.",
     "Epistemic uncertainty in a linear model is small and assumes the model is correctly specified. It does "
     "not cover model misspecification or future data drift.",
     "The original project's exact code and environment were lost, so the README numbers can be reproduced only "
     "approximately (Chapter 15)."])
H2("Future work")
BUL(["Hierarchical priors (e.g. partial pooling by contract type) or sparsity priors (horseshoe) for feature selection.",
     "Bayesian decision analysis: choose who receives an offer by maximising expected profit under the posterior.",
     "Non-linear Bayesian models (Gaussian-process classification, BART, Bayesian neural nets) to capture interactions.",
     "Survival / time-to-event modelling with tenure as time, if dated data were available.",
     "Variational inference (ADVI) for very large customer bases, checked against NUTS."])

# ================================================================== 17. Code walkthrough
H1("Code walkthrough")
SIMPLE("The code is split like a kitchen: one place fetches the ingredients (download), one washes and cuts them "
       "(data, features), one cooks (models), one tastes (evaluate), one plates the food (plots, report), and one "
       "recipe card runs the whole meal (main.py, run_all_experiments.sh).")
TABLE([["File", "What it does"],
       ["scripts/download_data.py", "resumable, validated download of the CSV (mirrors, retries, .part file, schema check)"],
       ["src/data.py", "load_raw, clean (blank TotalCharges, Churn→0/1, keep_id flag), label_encode, onehot_encode, "
                       "load_dataset, make_synthetic"],
       ["src/features.py", "split (stratified 80/20); FeaturePipeline = StandardScaler + PCA(≥95%), fit on train only"],
       ["src/models.py", "baselines; hdi(); BayesianLogisticRegression (build/fit/prob_samples/predict_summary/"
                         "posterior_predictive_rate/diagnostics)"],
       ["src/evaluate.py", "evaluate() → 9 metrics; expected_calibration_error; reliability_curve; paired bootstrap_ci"],
       ["src/plots.py", "all result figures (ROC/PR, calibration, trace, rank, energy, forest, uncertainty, PPC)"],
       ["src/main.py", "end-to-end pipeline; writes results/main/* and the HTML report; --legacy-keep-id"],
       ["src/report.py", "self-contained HTML dashboard (figures embedded as base64)"],
       ["scripts/run_cv.py", "5-fold CV of all models, resumable per fold"],
       ["scripts/run_prior_sensitivity.py", "σ ∈ {0.01,0.1,1,10} + PSIS-LOO comparison"],
       ["scripts/run_data_size.py", "learning curves Bayes vs ≈MLE LR, interval width vs n"],
       ["scripts/run_synthetic.py", "SYNTHETIC recovery and HDI coverage check"],
       ["scripts/make_figures.py, make_guide_figures.py", "dataset figures and the explanatory diagrams in this guide"],
       ["scripts/run_all_experiments.sh", "reproduces every number in REPORT.md and this guide"],
       ["tests/test_pipeline.py", "16 pytest tests (cleaning, encodings, no leakage, PCA, metrics, HDI, NUTS on synthetic data)"]],
      [5.4, 11.2])
H2("Running it")
CODE("""pip install -r requirements.txt
python scripts/download_data.py              # data/WA_Fn-UseC_-Telco-Customer-Churn.csv
python src/main.py                           # results/main/ (+ report.html), ~2 min on 4 cores
bash scripts/run_all_experiments.sh          # everything (~45 min), resumable
python -m pytest -q                          # 16 tests""")
H2("The key function: posterior predictions")
CODE("""def prob_samples(self, X):                       # (S draws, N customers)
    b0, beta = self.posterior_samples()            # (S,), (S, D)
    return sigmoid(b0[:, None] + beta @ X.T)

def predict_summary(self, X, prob=0.94):
    ps = self.prob_samples(X)
    lo, hi = hdi(ps, prob)                         # shortest window holding 94% of draws
    return {"mean": ps.mean(0), "std": ps.std(0), "hdi_low": lo, "hdi_high": hi}""")
H2("The HDI in five lines")
CODE("""def hdi(samples, prob=0.94):
    s = np.sort(samples, axis=0); n = s.shape[0]; k = int(np.floor(prob * n))
    widths = s[k:] - s[: n - k]                   # every window that holds k+1 sorted draws
    i = np.argmin(widths, axis=0)                 # the narrowest one
    return np.stack([np.take_along_axis(s, i[None], 0)[0], np.take_along_axis(s, (i + k)[None], 0)[0]])""")
H2("Outputs")
BUL(["<font face='DVM'>results/main/metrics.json</font>: all metrics, CIs, diagnostics and uncertainty statistics",
     "<font face='DVM'>results/main/predictions_test.csv</font>: every test customer, every model's probability, the Bayesian SD and HDI",
     "<font face='DVM'>results/main/posterior_summary.csv</font>: mean, sd, HDI, MCSE, ESS and R-hat of every coefficient",
     "<font face='DVM'>results/main/models/</font>: fitted scaler/PCA, LR, RF (joblib) and posterior draws (NetCDF)",
     "<font face='DVM'>results/main/report.html</font>: the automated dashboard"])

# ================================================================== 18. Viva
H1("Likely viva questions with answers")
QA = [
    ("What is churn and why predict it?", "A customer leaving the service. Retaining customers is cheaper than "
     "acquiring new ones, so early identification lets the company intervene."),
    ("Which dataset did you use?", f"The real IBM Telco Customer Churn dataset: 7,043 customers, 21 columns; "
     f"7,032 after removing 11 rows with blank TotalCharges; {pct(INFO['churn_rate'])} churn."),
    ("Why Bayesian?", "To get a full posterior over the parameters and therefore an uncertainty interval for "
     "every prediction, which supports risk-aware decisions. It also regularises naturally through the prior."),
    ("What is the prior and why N(0,1)?", "Each coefficient ~ N(0,1) on standardised inputs: weakly informative. "
     "It allows effects up to about ±2 log-odds per SD and is overwhelmed by the data. The sensitivity study "
     "shows that σ from 0.1 to 10 gives the same results."),
    ("Write the likelihood.", "∏ p<sub>i</sub><sup>y<sub>i</sub></sup>(1−p<sub>i</sub>)<sup>1−y<sub>i</sub></sup> "
     "with p<sub>i</sub> = σ(β<sub>0</sub> + βᵀz<sub>i</sub>)."),
    ("Why not compute the posterior exactly?", "The normalising constant ∫p(D|β)p(β)dβ has no closed form for "
     "the logistic likelihood, so we sample with MCMC."),
    ("Explain HMC in one minute.", "Parameters are a particle moving on the surface U(β) = −log posterior. We "
     "give it random momentum, simulate Hamiltonian dynamics with leapfrog steps using the gradient, and accept "
     "or reject with a Metropolis step based on the change in total energy H = U + K. Gradients let it travel far "
     "with high acceptance."),
    ("What does NUTS add?", "It picks the trajectory length automatically by doubling until the path makes a "
     "U-turn, and PyMC tunes the step size and mass matrix during warm-up."),
    ("How do you know it converged?", f"R-hat ≤ {DIAG['max_rhat']:.3f} (&lt; 1.01), bulk ESS ≥ {DIAG['min_ess_bulk']:.0f}, "
     "0 divergences, E-BFMI ≈ 1, overlapping traces and uniform rank plots across 4 chains."),
    ("What is an HDI?", "The narrowest interval containing a given posterior probability (94%). Unlike a frequentist "
     "confidence interval, it is a direct probability statement about the quantity, given the model and data."),
    ("Why does the Bayesian model score the same as logistic regression?", "With a Gaussian prior, the MAP is "
     "L2-penalised LR. With 5,625 observations the posterior is narrow, so its mean ≈ the MAP. The difference is "
     "the uncertainty, not the point prediction."),
    ("Then what is the benefit?", "Per-customer intervals, selective prediction (accuracy rises when unsure cases "
     "are deferred), robustness with little data (Chapter 15.4) and principled model comparison (LOO)."),
    ("Why PCA? Was it a good idea?", "To decorrelate and compress 19 features into 16 components (97% variance). "
     "Honestly, it slightly hurt accuracy and interpretability. Models on the original features did better."),
    ("Why is accuracy only ~78–80%?", "Churn is partly random behaviour and the data lacks key drivers "
     "(competitor offers, complaints, usage). Also, 73% is the trivial baseline, so AUC is the fairer measure."),
    ("What is calibration and is your model calibrated?", "Predicted probabilities match observed frequencies. "
     "Yes: ECE ≈ 0.02 for the Bayesian/LR models, and the reliability curves follow the diagonal."),
    ("Why a stratified split?", "So that train and test have the same 26.6% churn rate. Scaler and PCA are fit on "
     "the training split only, to avoid leakage."),
    ("What was the customerID issue?", "The original run label-encoded customerID and used it as a feature (20 "
     "features, 17 components). An ID carries no information, so the rebuild drops it. The original numbers are "
     "reproduced when it is kept."),
    ("What is a posterior predictive check?", "Simulating new data from the fitted model and comparing it with the "
     "real data. Our simulated churn rate interval contains the observed 26.6%."),
    ("What is the difference between epistemic and aleatoric uncertainty?", "Epistemic: uncertainty about the "
     "parameters, which shrinks with more data (the HDI). Aleatoric: the inherent randomness of the outcome given p."),
    ("How would you choose the decision threshold?", "Minimise expected cost: offer an incentive if p·(value of "
     "customer saved × success rate) &gt; incentive cost, possibly using the HDI lower bound to be conservative."),
    ("What would you do next?", "Hierarchical or horseshoe priors, Bayesian decision analysis on profit, non-linear "
     "Bayesian models, survival analysis on tenure."),
]
for q, a in QA:
    story.append(KeepTogether([Paragraph(f"<b>Q: {q}</b>", S["body"]),
                               Paragraph(f"<font color='#2E7D4F'><b>A:</b></font> {a}", S["body"]), Spacer(1, 3)]))

# ================================================================== 19. Glossary
H1("Glossary")
GLOSS = [
    ("Bayes' theorem", "posterior ∝ likelihood × prior."),
    ("Bootstrap", "Resampling the test set with replacement to estimate the uncertainty of a metric."),
    ("Brier score", "Mean squared error between predicted probability and the 0/1 outcome."),
    ("Calibration", "Agreement between predicted probabilities and observed frequencies."),
    ("Chain", "One independent run of the MCMC sampler."),
    ("Churn", "A customer cancelling the service."),
    ("Credible interval / HDI", "Interval containing a given posterior probability; the HDI is the narrowest one."),
    ("Divergence", "A numerically unstable HMC trajectory; a warning that the posterior is hard to explore."),
    ("ECE", "Expected calibration error: weighted average |observed − predicted| over probability bins."),
    ("elpd / LOO", "Expected log predictive density estimated by leave-one-out cross-validation (PSIS)."),
    ("ESS", "Effective sample size: how many independent draws the correlated chain is worth."),
    ("HMC", "Hamiltonian Monte Carlo: MCMC that uses gradients and simulated physics to propose distant moves."),
    ("Label / one-hot encoding", "Categories → integer codes / → one 0-1 column per category."),
    ("Leapfrog", "Numerical integrator for Hamilton's equations used inside HMC."),
    ("Likelihood", "Probability of the observed data for given parameter values."),
    ("Log-odds (logit)", "log(p/(1−p)); the linear score of logistic regression."),
    ("MAP", "Maximum a posteriori: the single most probable parameter value."),
    ("MCMC", "Markov chain Monte Carlo: draws correlated samples whose distribution converges to the posterior."),
    ("NUTS", "No-U-Turn Sampler: HMC with automatic trajectory length."),
    ("PCA", "Principal component analysis: rotation to uncorrelated directions ordered by variance."),
    ("Posterior", "Updated belief about the parameters after seeing the data."),
    ("Posterior predictive check", "Comparing data simulated from the fitted model with the real data."),
    ("Prior", "Belief about the parameters before seeing the data."),
    ("R-hat", "Convergence statistic comparing between- and within-chain variance (≈1 is good)."),
    ("ROC-AUC", "Probability that a random churner is ranked above a random stayer."),
    ("Sigmoid", "σ(z) = 1/(1+e<sup>−z</sup>): maps a score to a probability."),
    ("Stratified split", "Train/test split that keeps the class proportions equal."),
    ("z-score", "(x − mean) / standard deviation."),
]
TABLE([["Term", "Meaning"]] + [[a, b] for a, b in GLOSS], [4.4, 12.2])

# ================================================================== References
H1("References")
for r in [
    "IBM. Telco Customer Churn sample dataset (IBM Watson Analytics / Cognos). Kaggle: blastchar/telco-customer-churn.",
    "C. M. Bishop. <i>Pattern Recognition and Machine Learning</i>. Springer, 2006.",
    "K. P. Murphy. <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press, 2012.",
    "R. M. Neal. MCMC using Hamiltonian dynamics. In <i>Handbook of Markov Chain Monte Carlo</i>, CRC Press, 2011.",
    "D. Koller &amp; N. Friedman. <i>Probabilistic Graphical Models</i>. MIT Press, 2009.",
    "M. D. Hoffman &amp; A. Gelman. The No-U-Turn Sampler. <i>JMLR</i> 15:1593–1623, 2014.",
    "A. Gelman et al. <i>Bayesian Data Analysis</i>, 3rd ed. CRC Press, 2013.",
    "A. Vehtari, A. Gelman, D. Simpson, B. Carpenter, P.-C. Bürkner. Rank-normalization, folding, and localization: "
    "an improved R-hat. <i>Bayesian Analysis</i> 16(2), 2021.",
    "A. Vehtari, A. Gelman, J. Gabry. Practical Bayesian model evaluation using leave-one-out cross-validation and "
    "WAIC. <i>Statistics and Computing</i> 27, 2017.",
    "M. Betancourt. A conceptual introduction to Hamiltonian Monte Carlo. arXiv:1701.02434, 2017.",
    "O. Abril-Pla et al. PyMC: a modern and comprehensive probabilistic programming framework in Python. "
    "<i>PeerJ Computer Science</i> 9:e1516, 2023.",
    "R. Kumar et al. ArviZ: a unified library for exploratory analysis of Bayesian models. <i>JOSS</i> 4(33), 2019.",
    "C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On calibration of modern neural networks. ICML, 2017.",
]:
    story.append(Paragraph(r, S["bullet"], bulletText="•"))


# ------------------------------------------------------------------ document template with TOC + page numbers
class Doc(BaseDocTemplate):
    def __init__(self, fn):
        super().__init__(fn, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm,
                         bottomMargin=2 * cm, title="Bayesian Churn Prediction — Complete Project Guide",
                         author="Team 12, Amrita Vishwa Vidyapeetham")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("p", [frame], onPage=self.footer)])

    @staticmethod
    def footer(canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont("DV", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(2.2 * cm, 1.2 * cm, "Bayesian Churn Prediction — Complete Project Guide")
        canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"{doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if hasattr(flowable, "_toc"):
            self.notify("TOCEntry", (0, flowable._toc, self.page))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        OUT = sys.argv[1]
    Doc(OUT).multiBuild(story)
    print("wrote", OUT)
