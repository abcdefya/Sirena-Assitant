def test_config_has_groq_keys():
    from src.cores import config
    assert hasattr(config, "GROQ_API_KEY")
    assert hasattr(config, "GROQ_MODEL")
    assert config.GROQ_MODEL == "llama-3.3-70b-versatile"

def test_config_has_hf_model():
    from src.cores import config
    assert hasattr(config, "HF_MODEL")
    assert config.HF_MODEL == "paraphrase-multilingual-MiniLM-L12-v2"

def test_config_has_chroma_settings():
    from src.cores import config
    assert hasattr(config, "CHROMA_PERSIST_DIR")
    assert hasattr(config, "CHROMA_COLLECTION_NAME")

def test_config_has_data_dir():
    from src.cores import config
    from pathlib import Path
    assert isinstance(config.DATA_DIR, Path)

def test_config_has_rate_limits():
    from src.cores import config
    assert hasattr(config, "RATE_LIMIT_PER_MINUTE")
    assert hasattr(config, "RATE_LIMIT_PER_HOUR")
    assert isinstance(config.RATE_LIMIT_PER_MINUTE, int)
    assert isinstance(config.RATE_LIMIT_PER_HOUR, int)
