from mcrs.query_rewriters.hyde import build_hyde_messages, parse_hyde_output


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
