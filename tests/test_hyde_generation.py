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
    """Stub matching the LLAMA_MODEL surface HydeGenerator uses."""

    def __init__(self, canned: str):
        self.canned = canned

        class _Tok:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "PROMPT"

        self.tokenizer = _Tok()
        self.lm = self            # _generate_one is monkeypatched in the test
        self.device = "cpu"


def test_generator_caches_and_parses(tmp_path, monkeypatch):
    prompt = tmp_path / "p.txt"
    prompt.write_text("SYS", encoding="utf-8")
    lm = _FakeLM("INTENT: x\n1. doc one\n2. doc two\n")
    gen = HydeGenerator(lm, str(prompt), cache_dir=str(tmp_path))

    # Stub the raw generation so the test needs no real model.
    monkeypatch.setattr(gen, "_generate_one", lambda conv: lm.canned)

    out1 = gen.generate_batch(["conv A"])
    assert out1[0]["hyde_docs"] == ["doc one", "doc two"]
    assert (tmp_path / "hyde").exists()

    # Second call for the same conversation must hit cache (no new generation).
    called = {"n": 0}
    monkeypatch.setattr(
        gen, "_generate_one",
        lambda conv: (called.__setitem__("n", called["n"] + 1), "")[1],
    )
    out2 = gen.generate_batch(["conv A"])
    assert out2[0]["hyde_docs"] == ["doc one", "doc two"]
    assert called["n"] == 0  # served from disk cache


def test_generator_fallback_when_no_docs(tmp_path, monkeypatch):
    prompt = tmp_path / "p.txt"
    prompt.write_text("SYS", encoding="utf-8")
    gen = HydeGenerator(_FakeLM(""), str(prompt), cache_dir=str(tmp_path))
    monkeypatch.setattr(gen, "_generate_one", lambda conv: "garbage with no numbers")
    out = gen.generate_batch(["conv B"])
    assert out[0]["hyde_docs"] == ["conv B"]  # falls back to the conversation
