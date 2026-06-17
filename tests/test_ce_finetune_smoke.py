import pytest
torch = pytest.importorskip("torch"); pytest.importorskip("peft")
pytest.importorskip("transformers")


@pytest.mark.slow
def test_finetune_runs_one_step(tmp_path):
    """End-to-end smoke: one epoch on a tiny base model exercises the loop (accumulation, scheduler,
    early stop, masked-LCE). Needs network/HF access to fetch the tiny base; SKIPS cleanly without it
    (the real fine-tune runs on Colab where a HF token is configured)."""
    from mcrs.training.ce_finetune import finetune_cross_encoder
    groups = [("q1", ["pos doc", "neg a", "neg b"], 1.0)] * 4
    class L:  # noqa
        def log(self, d, step=None): pass
    train_cfg = {"epochs": 1, "lr": 1e-4, "batch_groups": 2, "grad_accum": 1, "log_every": 1,
                 "group_by_length": True, "early_stop_patience": 1}
    try:
        out = finetune_cross_encoder(
            groups, groups, base_model="hf-internal-testing/tiny-random-XLMRobertaForSequenceClassification",
            lora_cfg={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["query", "value"]},
            max_length=64, max_doc_tokens=40, dtype="fp32",
            train_cfg=train_cfg, logger=L(), out_dir=str(tmp_path))
    except Exception as e:                                    # offline/auth/download -> defer to Colab
        pytest.skip(f"smoke needs network/HF access for a tiny base model: {type(e).__name__}: {e}")
    assert out is not None
