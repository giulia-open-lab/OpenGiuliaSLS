"""
Created on Fri Apr 11 17:24:43 2025

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org 
matteo.bernabe@iteam.upv.es
"""
# === Post-processing: print colored Summary to terminal (custom renderer) ===
import re
import html
import pandas as pd

def _print_summary_to_terminal(md_path, manual_header=("Attribute", "Expected", "Actual", "Difference", "Success")):
    """
    Parse Summary.md and print tables to the terminal with:
    - HTML stripped from cells,
    - YES (green) / NO (red) using ANSI,
    - 'Success' column RIGHT-aligned (ANSI-safe),
    - other columns left-aligned.
    """

    # ANSI colors
    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"

    TAG_RE  = re.compile(r"<[^>]+>")          # strip HTML tags
    ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")   # strip ANSI codes for length calc

    def _clean_text(x: str) -> str:
        s = html.unescape(str(x))
        s = TAG_RE.sub("", s)
        return s.strip().replace('"', "")

    def _visible_len(s: str) -> int:
        """Length of a string ignoring ANSI codes."""
        return len(ANSI_RE.sub("", s))

    def _parse_markdown_table(table_str, manual_header=None) -> pd.DataFrame:
        rows = [r.strip() for r in table_str.splitlines() if r.strip()]
        if not rows:
            return pd.DataFrame()

        # header detection
        if len(rows) >= 2 and re.match(r'^[\-\|\s:]+$', rows[1]):
            header = [c.strip() for c in rows[0].split("|") if c.strip()]
            data_rows = rows[2:]
        else:
            first_cells = [c.strip() for c in rows[0].split("|") if c.strip()]
            header = manual_header if manual_header is not None else first_cells
            data_rows = rows[1:] if len(rows) > 1 else rows

        # rows -> dataframe
        data = []
        for r in data_rows:
            cells = [c.strip() for c in r.split("|") if c.strip()]
            if len(cells) < len(header):
                cells += [""] * (len(header) - len(cells))
            elif len(cells) > len(header):
                cells = cells[:len(header)]
            data.append(cells)

        df = pd.DataFrame(data, columns=header)
        # strip HTML from all cells
        for col in df.columns:
            df[col] = df[col].map(_clean_text)
        return df

    def _color_yes_no(s: str) -> str:
        low = s.lower()
        if "yes" in low and "no" not in low:
            return f"{GREEN}{s}{RESET}"
        if "no" in low and "yes" not in low:
            return f"{RED}{s}{RESET}"
        return s

    def _print_table(df: pd.DataFrame):
        """ANSI-aware fixed-width table printer with right-aligned 'Success'."""
        if df.empty:
            print("(no rows)")
            return

        cols = list(df.columns)

        # Prepare display values (apply colors only to Success)
        disp = []
        for _, row in df.iterrows():
            out_row = []
            for c in cols:
                val = str(row[c])
                if c == "Success":
                    val = _color_yes_no(val)
                out_row.append(val)
            disp.append(out_row)

        # Compute column widths using visible (ANSI-stripped) lengths
        widths = []
        for j, c in enumerate(cols):
            header_w = _visible_len(c)
            col_vals = [ _visible_len(row[j]) if c != "Success"
                        else _visible_len(ANSI_RE.sub("", row[j]))  # strip ANSI for width
                        for row in disp ]
            widths.append(max([header_w] + col_vals))

        # Helpers to pad cells with ANSI-awareness
        def pad_left(text, width):   # right-align
            vis = _visible_len(text)
            return (" " * max(0, width - vis)) + text

        def pad_right(text, width):  # left-align
            vis = _visible_len(text)
            return text + (" " * max(0, width - vis))

        # Header
        header_cells = []
        for j, c in enumerate(cols):
            if c == "Success":
                header_cells.append(pad_left(c, widths[j]))
            else:
                header_cells.append(pad_right(c, widths[j]))
        header_line = "  ".join(header_cells)
        sep_line    = "  ".join("-" * w for w in widths)
        print("\t" + header_line)
        print("\t" + sep_line)

        # Rows
        for row in disp:
            line_cells = []
            for j, c in enumerate(cols):
                cell = row[j]
                if c == "Success":
                    line_cells.append(pad_left(cell, widths[j]))
                else:
                    line_cells.append(pad_right(cell, widths[j]))
            print("\t" + "  ".join(line_cells))

    # -------- Parse Summary.md into blocks --------
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    lines = md_text.splitlines()
    blocks, current, current_sub = [], None, None
    for line in lines:
        s = line.strip()
        if s.startswith("##"):
            header = s.lstrip("#").strip()
            if header.lower() in ("mean", "median"):
                current_sub = header.lower()
                if current is not None:
                    current[current_sub + "_table"] = []
            else:
                if current is not None:
                    blocks.append(current)
                current = {"title": header, "run_time": None}
                current_sub = None
        else:
            if "**Run time**" in s:
                m = re.search(r"\*\*Run time\*\*:\s*(.+)", s)
                if m and current is not None:
                    current["run_time"] = _clean_text(m.group(1))
            elif s.startswith("|") and current_sub in ("mean", "median"):
                assert current is not None and current_sub is not None,\
                      "current and current_sub must not be None"
                current[current_sub + "_table"].append(s)
    if current is not None:
        blocks.append(current)

    # -------- Print blocks --------
    for block in blocks:
        print("=" * 80)
        print(f"\033[1m### {block.get('title','')}\033[0m")

        if block.get("run_time"):
            print(f"\tRun time: {float(block['run_time']):0.2f} seconds")

        for sub in ("mean_table", "median_table"):
            if sub in block:
                print(f"\n\t{sub.replace('_table','').capitalize()} results:")
                df = _parse_markdown_table("\n".join(block[sub]), manual_header=manual_header)
                _print_table(df)
        print()