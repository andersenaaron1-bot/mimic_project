# scripts/build_meas_vocab.py
import argparse, torch
import meds_reader as mr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    db = mr.SubjectDatabase(args.db)
    meas_codes = set()
    for sid in db:
        for ev in db[int(sid)].events:
            if getattr(ev, "numeric_value", None) is not None:
                c = getattr(ev, "code", None)
                if c:
                    meas_codes.add(str(c))

    meas_codes = sorted(meas_codes)             # deterministic
    code2id = {c: i+1 for i, c in enumerate(meas_codes)}  # 1..N
    torch.save(code2id, args.out)
    print(f"measurement codes: {len(code2id)} → {args.out}")

if __name__ == "__main__":
    main()
