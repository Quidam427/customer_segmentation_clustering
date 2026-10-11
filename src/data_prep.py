"""Loading and cleaning the invoice lines, and building one row of features per customer.

These functions follow Steps 2 and 3 of the notebook, so that the same preparation can be
re-run from a script without opening the notebook.
"""
import numpy as np
import pandas as pd

from .config import CACHE_FILE, RAW_FILE

ORIGINAL_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
CANDIDATE_FEATURES = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "TotalUnits", "UniqueProducts",
                      "ProductsPerOrder", "AvgUnitPrice", "Tenure", "ReturnRate"]


def load_raw(use_cache=True):
    """Read both sheets of the Excel file and stack them, keeping the overlapping days only once."""
    if use_cache and CACHE_FILE.exists():
        return pd.read_pickle(CACHE_FILE)

    sheets = pd.read_excel(RAW_FILE, sheet_name=None)
    for name, sheet in sheets.items():
        sheet["SourceSheet"] = name
    first, second = sheets["Year 2009-2010"], sheets["Year 2010-2011"]

    # The first sheet ends on 9 Dec 2010 and the second starts on 1 Dec 2010
    overlap_start = second["InvoiceDate"].min()
    raw = pd.concat([first[first["InvoiceDate"] < overlap_start], second], ignore_index=True)
    raw["Invoice"] = raw["Invoice"].astype(str)
    raw["StockCode"] = raw["StockCode"].astype(str)

    if use_cache:
        raw.to_pickle(CACHE_FILE)
    return raw


def clean_transactions(raw):
    """Apply the cleaning rules and return two tables: genuine sales, and returns (cancellations)."""
    df = raw.dropna(subset=["Customer ID"]).copy()
    df["Customer ID"] = df["Customer ID"].astype(int)
    df = df.drop_duplicates(subset=ORIGINAL_COLUMNS)
    df = df[df["StockCode"].str.fullmatch(r"\d{5}[A-Za-z]*")]      # drop postage, fees and adjustments
    df["LineValue"] = df["Quantity"] * df["Price"]

    is_cancellation = df["Invoice"].str.startswith("C")
    returns = df[is_cancellation & (df["Quantity"] < 0) & (df["Price"] > 0)].copy()
    sales = df[~is_cancellation & (df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    return sales, returns


def build_customer_features(sales, returns, today=None):
    """Squeeze each customer's invoice lines into one row: RFM, the extra features, country and RFM scores."""
    if today is None:
        today = sales["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    today = pd.Timestamp(today)

    per_customer = sales.groupby("Customer ID").agg(
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("LineValue", "sum"),
        TotalUnits=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        Lines=("Invoice", "size"),
    )

    customers = pd.DataFrame(index=per_customer.index)
    customers["Recency"] = (today - per_customer["LastPurchase"]).dt.days
    customers["Frequency"] = per_customer["Frequency"]
    customers["Monetary"] = per_customer["Monetary"]
    customers["AvgOrderValue"] = per_customer["Monetary"] / per_customer["Frequency"]
    customers["TotalUnits"] = per_customer["TotalUnits"]
    customers["UniqueProducts"] = per_customer["UniqueProducts"]
    customers["ProductsPerOrder"] = per_customer["Lines"] / per_customer["Frequency"]
    customers["AvgUnitPrice"] = per_customer["Monetary"] / per_customer["TotalUnits"]
    customers["Tenure"] = (today - per_customer["FirstPurchase"]).dt.days

    returned_value = (-returns["LineValue"]).groupby(returns["Customer ID"]).sum()
    customers["ReturnRate"] = (returned_value.reindex(customers.index).fillna(0) / customers["Monetary"]).clip(0, 1)

    # Country is recorded for the fairness check. It is not used by the model.
    customers["Country"] = sales.groupby("Customer ID")["Country"].agg(lambda s: s.mode().iat[0])
    customers["IsUK"] = (customers["Country"] == "United Kingdom").astype(int)

    # Classic RFM scores from 1 (worst) to 5 (best)
    customers["R_Score"] = pd.qcut(customers["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    customers["F_Score"] = pd.qcut(customers["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    customers["M_Score"] = pd.qcut(customers["Monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    customers["RFM_Total"] = customers[["R_Score", "F_Score", "M_Score"]].sum(axis=1)
    return customers


def prepare_model_input(customers, preprocessing_config, scaler):
    """Cap, log-transform and scale the selected features, using the recipe saved by the notebook."""
    features = preprocessing_config["selected_features"]
    caps = pd.Series(preprocessing_config["caps"])[features]
    prepared = customers[features].astype(float).clip(upper=caps, axis=1)
    log_features = [f for f in preprocessing_config["log_features"] if f in features]
    prepared[log_features] = np.log1p(prepared[log_features])
    return pd.DataFrame(scaler.transform(prepared), index=customers.index, columns=features)
