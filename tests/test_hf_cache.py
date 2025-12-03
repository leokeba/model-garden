"""Tests for HuggingFace cache configuration utilities."""

import os


class TestHFCacheConfiguration:
    """Tests for model_garden.utils.hf_cache module."""

    def test_configure_hf_cache_sets_environment_variables(self):
        """Test that configure_hf_cache sets all required environment variables."""
        from model_garden.utils.hf_cache import configure_hf_cache

        # Clear environment variables first
        env_vars = ["HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE"]
        original_values = {var: os.environ.get(var) for var in env_vars}

        try:
            # Remove env vars to test default behavior
            for var in env_vars:
                if var in os.environ:
                    del os.environ[var]

            result = configure_hf_cache()

            # Check all required environment variables are set
            assert "HF_HOME" in os.environ
            assert "TRANSFORMERS_CACHE" in os.environ
            assert "HF_DATASETS_CACHE" in os.environ
            assert "HUGGINGFACE_HUB_CACHE" in os.environ

            # Check return value contains expected keys
            assert "hf_home" in result
            assert "transformers_cache" in result
            assert "datasets_cache" in result
            assert "hub_cache" in result

            # Check paths are consistent
            assert os.environ["TRANSFORMERS_CACHE"].endswith("/hub")
            assert os.environ["HF_DATASETS_CACHE"].endswith("/datasets")

        finally:
            # Restore original values
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value
                elif var in os.environ:
                    del os.environ[var]

    def test_configure_hf_cache_respects_existing_hf_home(self):
        """Test that configure_hf_cache respects existing HF_HOME setting."""
        from model_garden.utils.hf_cache import configure_hf_cache

        custom_path = "/custom/hf/cache"
        original_hf_home = os.environ.get("HF_HOME")

        try:
            os.environ["HF_HOME"] = custom_path
            result = configure_hf_cache()

            assert result["hf_home"] == custom_path
            assert os.environ["HF_HOME"] == custom_path

        finally:
            if original_hf_home is not None:
                os.environ["HF_HOME"] = original_hf_home
            elif "HF_HOME" in os.environ:
                del os.environ["HF_HOME"]

    def test_get_hf_token_returns_token(self):
        """Test that get_hf_token returns the HF_TOKEN value."""
        from model_garden.utils.hf_cache import get_hf_token

        original_token = os.environ.get("HF_TOKEN")

        try:
            test_token = "hf_test_token_12345"
            os.environ["HF_TOKEN"] = test_token

            result = get_hf_token()
            assert result == test_token

        finally:
            if original_token is not None:
                os.environ["HF_TOKEN"] = original_token
            elif "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]

    def test_get_hf_token_returns_none_when_not_set(self):
        """Test that get_hf_token returns None when HF_TOKEN is not set."""
        from model_garden.utils.hf_cache import get_hf_token

        original_token = os.environ.get("HF_TOKEN")

        try:
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]

            result = get_hf_token()
            assert result is None

        finally:
            if original_token is not None:
                os.environ["HF_TOKEN"] = original_token

    def test_configure_pytorch_memory_sets_environment_variables(self):
        """Test that configure_pytorch_memory sets CUDA allocator config."""
        from model_garden.utils.hf_cache import configure_pytorch_memory

        original_values = {
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
        }

        try:
            configure_pytorch_memory()

            assert "PYTORCH_CUDA_ALLOC_CONF" in os.environ
            assert "PYTORCH_ALLOC_CONF" in os.environ
            assert "expandable_segments" in os.environ["PYTORCH_CUDA_ALLOC_CONF"]

        finally:
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value
                elif var in os.environ:
                    del os.environ[var]

    def test_configure_unsloth_settings_disables_statistics(self):
        """Test that configure_unsloth_settings disables Unsloth stats."""
        from model_garden.utils.hf_cache import configure_unsloth_settings

        original_value = os.environ.get("UNSLOTH_DISABLE_STATISTICS")

        try:
            configure_unsloth_settings()
            assert os.environ.get("UNSLOTH_DISABLE_STATISTICS") == "1"

        finally:
            if original_value is not None:
                os.environ["UNSLOTH_DISABLE_STATISTICS"] = original_value
            elif "UNSLOTH_DISABLE_STATISTICS" in os.environ:
                del os.environ["UNSLOTH_DISABLE_STATISTICS"]

    def test_configure_all_calls_all_configuration_functions(self):
        """Test that configure_all calls all individual configuration functions."""
        from model_garden.utils.hf_cache import configure_all

        original_values = {
            "HF_HOME": os.environ.get("HF_HOME"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "UNSLOTH_DISABLE_STATISTICS": os.environ.get("UNSLOTH_DISABLE_STATISTICS"),
        }

        try:
            result = configure_all()

            # Check that result contains cache paths
            assert "hf_home" in result

            # Check that all configuration functions were called
            assert "HF_HOME" in os.environ
            assert "PYTORCH_CUDA_ALLOC_CONF" in os.environ
            assert os.environ.get("UNSLOTH_DISABLE_STATISTICS") == "1"

        finally:
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value
                elif var in os.environ:
                    del os.environ[var]
