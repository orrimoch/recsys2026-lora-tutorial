# tests/test_bge_ce_training_data.py
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "music-crs-baselines"))
from build_bge_ce_training_data import select_ce_negatives, build_ce_row

def test_select_ce_negatives_drops_gold_and_caps():
    assert select_ce_negatives("g", ["g","n1","n2","n3","n4"], 3) == ["n1","n2","n3"]

def test_select_ce_negatives_gold_absent_keeps_all_capped():
    assert select_ce_negatives("g", ["n1","n2","n3"], 10) == ["n1","n2","n3"]

def test_build_ce_row_uses_text_map_and_keeps_index_alignment():
    text_map = {"g":"gold txt","n1":"neg1 txt","n2":"neg2 txt"}
    row = build_ce_row(query="goal: chill\nput on something mellow", gold_tid="g",
                       neg_tids=["n1","nX","n2"], text_map=text_map, user_id="u1", session_id="s1")
    assert row["pos"] == "gold txt"
    assert row["neg"] == ["neg1 txt","neg2 txt"]   # nX (absent) dropped, order kept
    assert row["neg_tids"] == ["n1","n2"] and row["pos_tid"] == "g" and row["user_id"] == "u1"
