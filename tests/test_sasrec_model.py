from mcrs.retrieval_modules.sasrec_model import build_session_examples


def test_build_session_examples_includes_empty_prefix_first_turn():
    ex = build_session_examples([[10, 11, 12]], max_len=50)
    assert ex == [([], 10), ([10], 11), ([10, 11], 12)]


def test_build_session_examples_left_truncates_to_max_len():
    ex = build_session_examples([[1, 2, 3, 4]], max_len=2)
    assert ([2, 3], 4) in ex
    assert ([1, 2], 3) in ex
    assert build_session_examples([[7]], max_len=50) == [([], 7)]
