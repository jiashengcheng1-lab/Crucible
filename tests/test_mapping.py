import pandas as pd

from crucible.mapping import build_history


def _df(rows, cols):
    return pd.DataFrame([{"label": l, "concept": c, **{p: v for p, v in zip(cols, vals)}} for l, c, vals in rows])


def test_mapping_from_edgartools_like_frames_ties_totals():
    cols = ["2023-12-31", "2024-12-31"]
    is_df = _df([
        ("Total Revenue", "us-gaap:Revenues", [1000, 1200]),
        ("Cost of Revenue", "us-gaap:CostOfRevenue", [600, 700]),
        ("Gross Profit", "us-gaap:GrossProfit", [400, 500]),
        ("Selling, General and Administrative", "us-gaap:SellingGeneralAndAdministrativeExpense", [150, 170]),
        ("Amortization of intangibles", "us-gaap:AmortizationOfIntangibleAssets", [20, 20]),  # unmapped -> residual
        ("Operating Income", "us-gaap:OperatingIncomeLoss", [230, 310]),
        ("Interest Expense", "us-gaap:InterestExpense", [30, 28]),
        ("Income Before Income Taxes", "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", [205, 290]),
        ("Income Tax Expense", "us-gaap:IncomeTaxExpenseBenefit", [45, 62]),
        ("Net Income", "us-gaap:NetIncomeLoss", [160, 228]),
        ("Diluted Weighted Average Shares", "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding", [100, 100]),
    ], cols)
    bs_df = _df([
        ("Cash and Cash Equivalents", "us-gaap:CashAndCashEquivalentsAtCarryingValue", [100, 150]),
        ("Accounts Receivable, Net", "us-gaap:AccountsReceivableNetCurrent", [200, 230]),
        ("Inventory", "us-gaap:InventoryNet", [150, 160]),
        ("Prepaid expenses", "us-gaap:PrepaidExpenseCurrent", [30, 40]),  # unmapped -> other current assets
        ("Total Current Assets", "us-gaap:AssetsCurrent", [480, 580]),
        ("Property, Plant and Equipment, Net", "us-gaap:PropertyPlantAndEquipmentNet", [300, 320]),
        ("Goodwill", "us-gaap:Goodwill", [400, 400]),
        ("Total Assets", "us-gaap:Assets", [1300, 1420]),
        ("Accounts Payable", "us-gaap:AccountsPayableCurrent", [120, 130]),
        ("Total Current Liabilities", "us-gaap:LiabilitiesCurrent", [250, 270]),
        ("Long-term Debt", "us-gaap:LongTermDebtNoncurrent", [400, 380]),
        ("Total Liabilities", "us-gaap:Liabilities", [700, 690]),
        ("Total Stockholders' Equity", "us-gaap:StockholdersEquity", [600, 730]),
        ("Total Liabilities and Stockholders' Equity", "us-gaap:LiabilitiesAndStockholdersEquity", [1300, 1420]),
    ], cols)
    cf_df = _df([
        ("Net Income", "us-gaap:NetIncomeLoss", [160, 228]),
        ("Depreciation and Amortization", "us-gaap:DepreciationDepletionAndAmortization", [50, 55]),
        ("Stock-based Compensation", "us-gaap:ShareBasedCompensation", [10, 12]),
        ("Net Cash Provided by Operating Activities", "us-gaap:NetCashProvidedByUsedInOperatingActivities", [190, 260]),
        ("Purchases of Property, Plant and Equipment", "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment", [-60, -75]),
        ("Net Cash Used in Investing Activities", "us-gaap:NetCashProvidedByUsedInInvestingActivities", [-80, -75]),
        ("Dividends Paid", "us-gaap:PaymentsOfDividends", [-40, -45]),
        ("Repayments of Debt", "us-gaap:RepaymentsOfLongTermDebt", [-20, -20]),  # unmapped -> net_debt_issuance residual
        ("Net Cash Used in Financing Activities", "us-gaap:NetCashProvidedByUsedInFinancingActivities", [-60, -135]),
        ("Net Increase in Cash", "us-gaap:CashAndCashEquivalentsPeriodIncreaseDecrease", [50, 50]),
    ], cols)
    hist, reports = build_history(is_df, bs_df, cf_df)
    assert list(hist.columns) == [2023, 2024]
    # residuals tie the statements to reported totals
    assert hist.loc["other_opex", 2024] == 20 and hist.loc["other_current_assets", 2024] == 40
    assert (hist.loc["total_assets"] - hist.loc["total_liabilities_and_equity"]).abs().max() == 0
    assert hist.loc["capex", 2024] == 75 and hist.loc["dividends", 2024] == 45  # magnitudes
    assert hist.loc["net_debt_issuance", 2024] == -90  # cff + dividends + buybacks
    assert hist.loc["d_nwc", 2024] == 260 - 228 - 55 - 12
    assert "Amortization of intangibles" in reports["IS"].unmapped and "Prepaid expenses" in reports["BS"].unmapped
    assert reports["IS"].residuals["other_opex"] > 0


def test_label_fallback_when_concepts_are_missing():
    cols = ["FY2024", "FY2025"]
    is_df = pd.DataFrame({"label": ["Net sales", "Cost of goods sold", "Operating income", "Net income"], "FY2024": [10, 6, 2, 1.5], "FY2025": [12, 7, 3, 2]})
    bs_df = pd.DataFrame({"label": ["Cash", "Total current assets", "Total assets", "Total current liabilities", "Total liabilities", "Total equity"],
                          "FY2024": [1, 4, 10, 2, 5, 5], "FY2025": [2, 5, 12, 3, 6, 6]})
    cf_df = pd.DataFrame({"label": ["Net income", "Cash flows from operating activities", "Capital expenditures", "Cash flows from investing activities", "Cash flows from financing activities"],
                          "FY2024": [1.5, 2, -1, -1, -0.5], "FY2025": [2, 3, -1, -1, -1]})
    hist, reports = build_history(is_df, bs_df, cf_df)
    assert hist.loc["revenue", 2025] == 12 and hist.loc["gross_profit", 2025] == 5 and hist.loc["capex", 2025] == 1
    assert hist.loc["total_liabilities_and_equity", 2025] == 12 and not reports["IS"].unmapped
