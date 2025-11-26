import meds_reader as mr
from collections import Counter

db = mr.SubjectDatabase(r"C:\MedsDemo_reader")

prefix_counts = Counter()
for i, sid in enumerate(db):
    subj = db[int(sid)]
    for ev in subj.events:
        code = str(ev.code)
        # crude: split on '//' and take the first piece as an event family
        prefix = code.split("//", 1)[0]
        prefix_counts[prefix] += 1
    if i >= 50:  # sample first ~50 subjects
        break

print("Top code prefixes:")
for k, v in prefix_counts.most_common(30):
    print(f"{k:40s} {v}")



