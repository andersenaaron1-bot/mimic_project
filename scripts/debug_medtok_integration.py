# scripts/debug_medtok_integration.py
import random
import meds_reader as mr
from src.ehr_hier.data.token_types import TokenCategory
from src.ehr_hier.tokenizers.base_encoder import build_base_encoders
from src.ehr_hier.tokenizers.measurement_encoder import MeasurementEncoderConfig
from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline
from src.ehr_hier.tokenizers.simple_categorical_encoders import CategoryVocab

DB = r"C:\MedsDemo_reader"   # your meds_reader db

def main():
    db = mr.SubjectDatabase(DB)
    # --- stub vocabs for non-diagnosis categories (map everything to UNK)
    empty_vocab = CategoryVocab(offset=2_000_000, mapping={}).with_unk()

    # --- measurement encoder config you already trained
    meas_cfg = MeasurementEncoderConfig(
        cvae_ckpt=r"C:\MedsArtifacts\cvae_ckpt.pt",
        tokenizer_ckpt=r"C:\MedsArtifacts\value_tokenizer.pt",
        z_dim=64, num_codebooks=2, codebook_size=256,
        n_vars=1117,                     # your meas vocab size
        mean_by_var=torch.load(r"C:\MedsArtifacts\meas_mean.pt"),
        std_by_var=torch.load(r"C:\MedsArtifacts\meas_std.pt"),
        code2id=torch.load(r"C:\MedsArtifacts\code2id_meas.pt"),
    )

    enc = build_base_encoders(
        meas_cfg,
        proc_vocab=empty_vocab,
        med_vocab=empty_vocab,
        struct_vocab=empty_vocab,
    )

    # pick a subject with at least one diagnosis event (you can loop until you find one)
    sid = random.choice(list(db))
    subj = db[int(sid)]

    print(f"Subject {sid} raw diagnosis snippets (first 10):")
    shown = 0
    for ev in subj.events:
        code = getattr(ev, "code", None)
        if code and "ICD" in str(code).upper():
            print("  ", code)
            shown += 1
            if shown >= 10: break

    # Build timeline (will include measurements + MedTok diagnoses + specials)
    toks = build_subject_timeline(db, subject_id=int(sid), encoders=enc)
    # Show first 30 tokens
    print("\nFirst 30 tokens:")
    for t in toks[:30]:
        print(t)

if __name__ == "__main__":
    import torch
    main()
