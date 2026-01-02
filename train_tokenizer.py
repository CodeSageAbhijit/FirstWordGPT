import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="gutenberg_clean.txt",
    model_prefix="gutenberg_bpe",
    model_type="bpe",

    vocab_size=32000,
    character_coverage=0.9995,

    input_sentence_size=5000000,
    shuffle_input_sentence=True,

    bos_id=1,
    eos_id=2,
    pad_id=0,
    unk_id=3,

    normalization_rule_name="nmt_nfkc"
)

print("✅ Gutenberg BPE tokenizer trained")
