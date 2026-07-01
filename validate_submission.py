#!/usr/bin/env python3
import csv, re, sys
from pathlib import Path

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")

def validate_submission(csv_path):
    errors = []
    path = Path(csv_path)
    if path.suffix.lower() != ".csv":
        errors.append("Filename must use a .csv extension.")
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                errors.append("File is empty.")
                return errors
            if header != REQUIRED_HEADER:
                errors.append(f"Header must be exactly: {','.join(REQUIRED_HEADER)}\nFound: {','.join(header)}")
            data_rows = [row for row in reader if any(cell.strip() for cell in row)]
    except Exception as e:
        errors.append(f"Cannot read file: {e}")
        return errors

    if len(data_rows) != 100:
        errors.append(f"Expected 100 data rows, found {len(data_rows)}.")

    seen_ids = set(); seen_ranks = set(); by_rank = []
    for i, cells in enumerate(data_rows):
        row_num = 2 + i
        if len(cells) != 4:
            errors.append(f"Row {row_num}: expected 4 columns, got {len(cells)}.")
            continue
        row = dict(zip(REQUIRED_HEADER, cells))
        cid = row["candidate_id"].strip()
        rank_s = row["rank"].strip()
        score_s = row["score"].strip()
        if not CANDIDATE_ID_PATTERN.match(cid):
            errors.append(f"Row {row_num}: invalid candidate_id '{cid}'.")
        elif cid in seen_ids:
            errors.append(f"Row {row_num}: duplicate candidate_id.")
        else:
            seen_ids.add(cid)
        try:
            rank = int(rank_s)
            if not 1 <= rank <= 100: errors.append(f"Row {row_num}: rank out of range.")
            elif rank in seen_ranks: errors.append(f"Row {row_num}: duplicate rank {rank}.")
            else: seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {row_num}: rank must be integer."); rank = None
        try:
            score = float(score_s)
        except ValueError:
            errors.append(f"Row {row_num}: score must be float."); score = None
        if rank and score is not None and cid:
            by_rank.append((rank, score, cid))

    missing = set(range(1,101)) - seen_ranks
    if missing: errors.append(f"Missing ranks: {sorted(missing)[:10]}")

    by_rank.sort()
    for i in range(len(by_rank)-1):
        r1,s1,_ = by_rank[i]; r2,s2,_ = by_rank[i+1]
        if s1 < s2:
            errors.append(f"score not non-increasing: rank {r1}({s1}) < rank {r2}({s2})")
    return errors

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_submission.py <file>.csv"); sys.exit(1)
    errors = validate_submission(sys.argv[1])
    if errors:
        print(f"Validation FAILED ({len(errors)} issues):")
        for e in errors: print(f"  - {e}")
        sys.exit(1)
    print("✅ Submission is VALID.")

if __name__ == "__main__":
    main()