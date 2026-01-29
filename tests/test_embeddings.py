"""Tests for embedding configuration and management."""



class TestEmbeddingConfig:
    """Test EmbeddingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from rlm_dspy.core.embeddings import EmbeddingConfig

        config = EmbeddingConfig()

        assert config.model == "openai/text-embedding-3-small"
        assert config.local_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.batch_size == 100
        assert config.caching is True

    def test_custom_values(self):
        """Test custom configuration values."""
        from rlm_dspy.core.embeddings import EmbeddingConfig

        config = EmbeddingConfig(
            model="cohere/embed-english-v3.0",
            batch_size=50,
            api_key="test-key",
        )

        assert config.model == "cohere/embed-english-v3.0"
        assert config.batch_size == 50
        assert config.api_key == "test-key"

    def test_repr_hides_api_key(self):
        """Test that repr doesn't expose API key."""
        from rlm_dspy.core.embeddings import EmbeddingConfig

        config = EmbeddingConfig(api_key="secret-key-12345")
        repr_str = repr(config)

        assert "secret-key-12345" not in repr_str
        assert "***" in repr_str

    def test_from_user_config(self):
        """Test loading from user config."""
        from rlm_dspy.core.embeddings import EmbeddingConfig

        config = EmbeddingConfig.from_user_config()

        # Should have valid defaults
        assert config.model is not None
        assert config.batch_size > 0

    def test_env_var_override(self, monkeypatch):
        """Test environment variable override."""
        from rlm_dspy.core.embeddings import EmbeddingConfig

        monkeypatch.setenv("RLM_EMBEDDING_MODEL", "voyage/voyage-3")
        monkeypatch.setenv("RLM_EMBEDDING_BATCH_SIZE", "200")

        # Clear cache to force reload
        from rlm_dspy.core.embeddings import clear_embedder_cache
        clear_embedder_cache()

        config = EmbeddingConfig.from_user_config()

        assert config.model == "voyage/voyage-3"
        assert config.batch_size == 200


class TestGetEmbedder:
    """Test embedder creation."""

    def test_get_embedder_returns_embedder(self):
        """Test that get_embedder returns a valid embedder."""
        from rlm_dspy.core.embeddings import get_embedder, EmbeddingConfig

        config = EmbeddingConfig(model="openai/text-embedding-3-small")
        embedder = get_embedder(config)

        # Should be a dspy.Embedder
        assert hasattr(embedder, "__call__")
        assert hasattr(embedder, "model")

    def test_embedder_caching(self):
        """Test that embedders are cached."""
        from rlm_dspy.core.embeddings import get_embedder, clear_embedder_cache, EmbeddingConfig

        clear_embedder_cache()

        config = EmbeddingConfig(model="openai/text-embedding-3-small")
        embedder1 = get_embedder(config)
        embedder2 = get_embedder(config)

        # Should be the same instance
        assert embedder1 is embedder2

    def test_clear_embedder_cache(self):
        """Test cache clearing."""
        from rlm_dspy.core.embeddings import get_embedder, clear_embedder_cache, EmbeddingConfig

        config = EmbeddingConfig(model="openai/text-embedding-3-small")
        embedder1 = get_embedder(config)

        clear_embedder_cache()

        embedder2 = get_embedder(config)

        # Should be different instances after cache clear
        assert embedder1 is not embedder2

    def test_local_embedder_requires_sentence_transformers(self):
        """Test that local embedder requires sentence-transformers."""
        from rlm_dspy.core.embeddings import get_embedder, clear_embedder_cache, EmbeddingConfig

        clear_embedder_cache()
        config = EmbeddingConfig(model="local")

        # This will either work (if sentence-transformers is installed)
        # or raise ImportError
        try:
            embedder = get_embedder(config)
            # If it works, should be callable
            assert callable(embedder)
        except ImportError as e:
            assert "sentence-transformers" in str(e)


class TestEmbeddingDimensions:
    """Test embedding dimension detection."""

    def test_known_model_dimensions(self):
        """Test known model dimension lookup."""
        from rlm_dspy.core.embeddings import get_embedding_dim, EmbeddingConfig

        config = EmbeddingConfig(model="openai/text-embedding-3-small")
        dim = get_embedding_dim(config)

        assert dim == 1536

    def test_openai_large_dimensions(self):
        """Test OpenAI large model dimensions."""
        from rlm_dspy.core.embeddings import get_embedding_dim, EmbeddingConfig

        config = EmbeddingConfig(model="openai/text-embedding-3-large")
        dim = get_embedding_dim(config)

        assert dim == 3072

    def test_cohere_dimensions(self):
        """Test Cohere model dimensions."""
        from rlm_dspy.core.embeddings import get_embedding_dim, EmbeddingConfig

        config = EmbeddingConfig(model="cohere/embed-english-v3.0")
        dim = get_embedding_dim(config)

        assert dim == 1024


class TestUserConfigEmbeddings:
    """Test embedding settings in user config."""

    def test_default_config_has_embedding_settings(self):
        """Test that DEFAULT_CONFIG includes embedding settings."""
        from rlm_dspy.core.user_config import DEFAULT_CONFIG

        assert "embedding_model" in DEFAULT_CONFIG
        assert "local_embedding_model" in DEFAULT_CONFIG
        assert "embedding_batch_size" in DEFAULT_CONFIG
        assert "index_dir" in DEFAULT_CONFIG
        assert "use_faiss" in DEFAULT_CONFIG

    def test_config_template_has_embedding_section(self):
        """Test that CONFIG_TEMPLATE includes embedding section."""
        from rlm_dspy.core.user_config import CONFIG_TEMPLATE

        assert "Embedding Settings" in CONFIG_TEMPLATE
        assert "embedding_model" in CONFIG_TEMPLATE
        assert "local_embedding_model" in CONFIG_TEMPLATE
        assert "Vector Index Settings" in CONFIG_TEMPLATE
        assert "use_faiss" in CONFIG_TEMPLATE

    def test_save_config_includes_embedding_settings(self, tmp_path, monkeypatch):
        """Test that save_config writes embedding settings."""
        from rlm_dspy.core.user_config import save_config

        # Temporarily change config location
        test_config_dir = tmp_path / ".rlm"
        test_config_file = test_config_dir / "config.yaml"

        monkeypatch.setattr("rlm_dspy.core.user_config.CONFIG_DIR", test_config_dir)
        monkeypatch.setattr("rlm_dspy.core.user_config.CONFIG_FILE", test_config_file)

        config = {
            "model": "openai/gpt-4o",
            "embedding_model": "cohere/embed-english-v3.0",
            "use_faiss": False,
        }

        save_config(config)

        content = test_config_file.read_text()
        assert "embedding_model: cohere/embed-english-v3.0" in content
        assert "use_faiss: false" in content
