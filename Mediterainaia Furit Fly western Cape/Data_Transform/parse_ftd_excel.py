import argparse
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "FTDs_multiple_years_Dean.xlsx")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "wc_trap_ftd_long.csv")

# Sheet name -> canonical region name used everywhere downstream.
SHEET_TO_REGION = {
    "Warm bokkeveld": "Warm Bokkeveld",
    "Wolseley": "Wolseley",
    "Tulbagh": "Tulbagh",
    "Elgin&Grabouw": "Elgin & Grabouw",
    "Vyeboom": "Vyeboom",
}

HEADER_ROW = 2  # 0-indexed row containing "Week", 2010, 2011, ...


def parse_sheet(xls, sheet_name, region):
    raw = xls.parse(sheet_name, header=HEADER_ROW)
    raw = raw.rename(columns={raw.columns[0]: "Week"} if raw.columns[0] != "Week" else {})

    # The "Elgin&Grabouw" and "Vyeboom" sheets have a leading blank column
    # (data starts in column B), which pandas reads in as an unnamed column
    # of all-NaN ahead of "Week". Drop any fully-empty columns before the
    # real header so the row alignment below is consistent across sheets.
    raw = raw.dropna(axis=1, how="all")
    raw = raw.rename(columns={raw.columns[0]: "Week"})

    year_cols = [c for c in raw.columns if c != "Week"]

    records = []
    for _, row in raw.iterrows():
        week = row["Week"]
        # Stop at the trailing "Average" summary row (and any other
        # non-numeric junk) rather than trying to parse it as a week.
        try:
            week_num = int(week)
        except (TypeError, ValueError):
            continue

        for year_col in year_cols:
            value = row[year_col]
            if pd.isna(value):
                continue
            if isinstance(value, str):
                cleaned = value.replace("\xa0", "").strip()
                if cleaned in ("", "Not avail", "N/A", "#DIV/0!"):
                    continue
                cleaned = cleaned.replace(",", ".")
                try:
                    value = float(cleaned)
                except ValueError:
                    continue
            try:
                year = int(year_col)
            except (TypeError, ValueError):
                continue

            records.append({
                "region": region,
                "year": year,
                "week": week_num,
                "ftd": float(value),
            })

    return pd.DataFrame.from_records(records)


def main():
    parser = argparse.ArgumentParser(
        description="Parse Ghian du Toit's per-region FTD workbook into a long-format CSV."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to the FTD .xlsx workbook")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to write the long-format CSV")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"FTD workbook not found at '{args.input}'. Pass --input pointing at the "
            "Excel file Ghian du Toit shared (sheets: Warm bokkeveld, Wolseley, Tulbagh, "
            "Elgin&Grabouw, Vyeboom)."
        )

    xls = pd.ExcelFile(args.input)

    frames = []
    for sheet_name, region in SHEET_TO_REGION.items():
        if sheet_name not in xls.sheet_names:
            print(f"WARNING: expected sheet '{sheet_name}' not found, skipping.")
            continue
        frames.append(parse_sheet(xls, sheet_name, region))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["region", "year", "week"]).reset_index(drop=True)
    df.to_csv(args.output, index=False)

    print(f"Parsed {len(df)} region-week FTD records across {df['region'].nunique()} regions.")
    print(df.groupby("region")["year"].agg(["min", "max", "count"]))
    print(f"Saved to: '{args.output}'")


if __name__ == "__main__":
    main()
