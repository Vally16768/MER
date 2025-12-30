from mer_builder.prepare.split_speakers import speaker_overlap, split_speakers


def test_split_speakers_disjoint():
    speakers = [f"spk{i}" for i in range(20)]
    mapping = split_speakers(speakers, seed=123)
    speakers_by_split = {"train": set(), "val": set(), "test": set()}
    for spk, split in mapping.items():
        speakers_by_split[split].add(spk)
    assert speaker_overlap(speakers_by_split) == {}


def test_split_speakers_reproducible():
    speakers = [f"spk{i}" for i in range(10)]
    m1 = split_speakers(speakers, seed=999)
    m2 = split_speakers(speakers, seed=999)
    assert m1 == m2


def test_split_speakers_ratio_basic():
    speakers = [f"spk{i}" for i in range(10)]
    mapping = split_speakers(speakers, seed=1)
    counts = {"train": 0, "val": 0, "test": 0}
    for s in mapping.values():
        counts[s] += 1
    assert counts["train"] == 8
    assert counts["val"] == 1
    assert counts["test"] == 1

