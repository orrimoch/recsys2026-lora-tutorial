from mcrs.query_rewriters.hyde import (
    build_hyde_messages,
    parse_hyde_output,
    HydeGenerator,
)


def test_build_hyde_messages_has_system_and_user():
    msgs = build_hyde_messages("user: something upbeat", "SYS")
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1]["role"] == "user"
    assert "something upbeat" in msgs[1]["content"]


def test_parse_hyde_output_extracts_intent_and_docs():
    text = (
        "INTENT: dreamy synth-pop for a late night drive\n"
        "1. A hazy mid-tempo synth-pop track, breathy female vocals, 80s analog pads\n"
        "2. Downtempo electronic, reverb-heavy guitar, nostalgic and melancholic\n"
        "3. Slow dream-pop, shoegaze textures, soft male vocals\n"
    )
    out = parse_hyde_output(text)
    assert out["intent_query"] == "dreamy synth-pop for a late night drive"
    assert len(out["hyde_docs"]) == 3
    assert out["hyde_docs"][0].startswith("A hazy mid-tempo")


def test_parse_hyde_output_tolerates_missing_intent_and_paren_numbering():
    out = parse_hyde_output("1) only one description here\n")
    assert out["hyde_docs"] == ["only one description here"]
    assert out["intent_query"] == ""


class _FakeLM:
    """Minimal stub — HydeGenerator only stores the lm; generation is batched
    via _generate_raw, which the tests monkeypatch (no real model needed)."""


def test_generator_caches_and_parses(tmp_path, monkeypatch):
    prompt = tmp_path / "p.txt"
    prompt.write_text("SYS", encoding="utf-8")
    gen = HydeGenerator(_FakeLM(), str(prompt), cache_dir=str(tmp_path))

    monkeypatch.setattr(
        gen, "_generate_raw",
        lambda convs: ["INTENT: x\n1. doc one\n2. doc two\n" for _ in convs])
    out1 = gen.generate_batch(["conv A"])
    assert out1[0]["hyde_docs"] == ["doc one", "doc two"]
    assert (tmp_path / "hyde").exists()

    # Second call for the same conversation must hit cache (no generation).
    called = {"n": 0}

    def _spy(convs):
        called["n"] += len(convs)
        return ["" for _ in convs]

    monkeypatch.setattr(gen, "_generate_raw", _spy)
    out2 = gen.generate_batch(["conv A"])
    assert out2[0]["hyde_docs"] == ["doc one", "doc two"]
    assert called["n"] == 0


def test_generator_fallback_when_no_docs(tmp_path, monkeypatch):
    prompt = tmp_path / "p.txt"
    prompt.write_text("SYS", encoding="utf-8")
    gen = HydeGenerator(_FakeLM(), str(prompt), cache_dir=str(tmp_path))
    monkeypatch.setattr(gen, "_generate_raw",
                        lambda convs: ["garbage with no numbers" for _ in convs])
    out = gen.generate_batch(["conv B"])
    assert out[0]["hyde_docs"] == ["conv B"]  # falls back to the conversation


def test_generator_batches_only_misses_and_preserves_order(tmp_path, monkeypatch):
    prompt = tmp_path / "p.txt"
    prompt.write_text("SYS", encoding="utf-8")
    gen = HydeGenerator(_FakeLM(), str(prompt), cache_dir=str(tmp_path))

    def _echo(convs):
        return [f"INTENT: i\n1. d::{c}\n" for c in convs]

    monkeypatch.setattr(gen, "_generate_raw", _echo)
    gen.generate_batch(["B"])  # pre-cache conversation "B"

    seen = {"convs": None}

    def _spy(convs):
        seen["convs"] = list(convs)
        return _echo(convs)

    monkeypatch.setattr(gen, "_generate_raw", _spy)
    out = gen.generate_batch(["A", "B", "C"])  # B is cached
    assert seen["convs"] == ["A", "C"]            # only misses are generated
    assert out[0]["hyde_docs"] == ["d::A"]
    assert out[1]["hyde_docs"] == ["d::B"]        # served from cache
    assert out[2]["hyde_docs"] == ["d::C"]
