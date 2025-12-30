from mer_builder.prepare.map_labels import map_emotion


def test_meld_identity():
    assert map_emotion("meld", "anger").emotion == "anger"
    assert map_emotion("meld", "sadness").emotion == "sadness"


def test_cremad_mapping():
    assert map_emotion("cremad", "HAP").emotion == "joy"
    assert map_emotion("cremad", "happy").emotion == "joy"
    assert map_emotion("cremad", "ANG").emotion == "anger"


def test_ravdess_mapping():
    assert map_emotion("ravdess", "calm").emotion == "neutral"
    assert map_emotion("ravdess", "happy").emotion == "joy"
    assert map_emotion("ravdess", "02").emotion == "neutral"
    assert map_emotion("ravdess", "03").emotion == "joy"


def test_esd_mapping():
    assert map_emotion("esd", "angry").emotion == "anger"
    assert map_emotion("esd", "happy").emotion == "joy"
    assert map_emotion("esd", "sad").emotion == "sadness"


def test_mead_contempt_modes():
    assert map_emotion("mead", "contempt", mead_contempt="drop").emotion is None
    r = map_emotion("mead", "contempt", mead_contempt="map_to_disgust")
    assert r.emotion == "disgust"
    assert "contempt" in (r.notes or "")


def test_emovdb_unknown_drops():
    assert map_emotion("emovdb", "sleepiness").emotion is None


def test_emovdb_amused_maps_to_joy():
    r = map_emotion("emovdb", "amused")
    assert r.emotion == "joy"


def test_emovdb_sleepy_can_map_to_neutral():
    r = map_emotion("emovdb", "sleepy", emovdb_sleepy="map_to_neutral")
    assert r.emotion == "neutral"
    assert "sleepy" in (r.notes or "").lower()


def test_iemocap_mapping():
    assert map_emotion("iemocap", "ang").emotion == "anger"
    assert map_emotion("iemocap", "sad").emotion == "sadness"
    assert map_emotion("iemocap", "neu").emotion == "neutral"
    assert map_emotion("iemocap", "sur").emotion == "surprise"
    assert map_emotion("iemocap", "fea").emotion == "fear"
    assert map_emotion("iemocap", "dis").emotion == "disgust"
    assert map_emotion("iemocap", "hap").emotion == "joy"
    assert map_emotion("iemocap", "exc").emotion == "joy"
    assert map_emotion("iemocap", "fru").emotion == "anger"
    assert map_emotion("iemocap", "xxx").emotion is None
