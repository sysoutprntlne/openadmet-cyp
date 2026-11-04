"""Build a reading-assignment spreadsheet of open-code leaderboard submissions.

Pulls the live regression leaderboard, keeps rows with ``Open Code == Yes``,
and writes results/lit/lit_assignments.xlsx with a link per submission, an
alternating Assigned column (Denali, John, Denali, ...) and a Done checkbox.

Usage:
    python scripts/make_lit_sheet.py [--board live|interim] [--out PATH]
"""

import argparse
import csv
import io
import sys
from itertools import cycle
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_lit import BOARDS, OUT, fetch_leaderboard  # noqa: E402

ASSIGNEES = ["Denali", "John"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--board", choices=sorted(BOARDS), default="live")
    ap.add_argument("--out", type=Path, default=OUT / "lit_assignments.xlsx")
    args = ap.parse_args()

    rows = csv.DictReader(io.StringIO(fetch_leaderboard(args.board)))
    open_rows = [
        r for r in rows
        if r["Open Code"].strip().lower() == "yes" and r["Model Report Link"].startswith("http")
    ]

    wb = Workbook()
    ws = wb.active
    ws.title = "Open-code submissions"
    header = ["Rank", "Username", "Submitted", "MA-ST-RAE", "MA-MAE", "MA-R²", "Link", "Assigned", "Done"]
    ws.append(header)
    for c in ws[1]:
        c.font = Font(bold=True)

    for r, who in zip(open_rows, cycle(ASSIGNEES)):
        ws.append([
            int(r["Rank"]), r["Username"], r["Submitted"],
            round(float(r["MA-ST-RAE"]), 4), round(float(r["MA-MAE"]), 4), round(float(r["MA-R²"]), 4),
            r["Model Report Link"], who, False,
        ])
        link = ws.cell(row=ws.max_row, column=header.index("Link") + 1)
        link.hyperlink = r["Model Report Link"]
        link.font = Font(color="0563C1", underline="single")

    # Done column: TRUE/FALSE dropdown (renders as a checkbox-style toggle)
    last = ws.max_row
    done_col = ws.cell(row=1, column=header.index("Done") + 1).column_letter
    dv = DataValidation(type="list", formula1='"TRUE,FALSE"', allow_blank=False)
    ws.add_data_validation(dv)
    dv.add(f"{done_col}2:{done_col}{last}")

    for col, width in zip("ABCDEFGHI", [6, 26, 20, 11, 9, 9, 70, 10, 7]):
        ws.column_dimensions[col].width = width
    ws.freeze_panes = "A2"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.out)
    print(f"{len(open_rows)} submissions -> {args.out}")


if __name__ == "__main__":
    main()
