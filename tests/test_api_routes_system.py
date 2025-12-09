"""Tests for system API routes.

These tests verify the system management endpoints work correctly.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Mark all tests to bypass mock_heavy_imports
pytestmark = pytest.mark.requires_gpu


@pytest.fixture
def app():
    """Create a test FastAPI application."""
    from model_garden.api.app import create_app

    return create_app()


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_storage():
    """Mock storage manager."""
    with patch("model_garden.api.storage.get_storage_manager") as mock:
        storage = MagicMock()
        storage.load_models.return_value = {}
        storage.load_training_jobs.return_value = {}
        mock.return_value = storage
        yield storage


class TestSystemStatus:
    """Tests for GET /api/v1/system/status."""

    def test_get_system_status(self, client: TestClient):
        """Test getting system status."""
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200

        data = response.json()
        assert "system" in data
        assert "gpu" in data
        assert "storage" in data

    def test_get_system_status_structure(self, client: TestClient):
        """Test system status response structure."""
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200

        data = response.json()
        # System info
        assert "cpu_count" in data["system"]
        assert "cpu_percent" in data["system"]
        assert "memory_total" in data["system"]
        # GPU info
        assert "available" in data["gpu"]


class TestListBackends:
    """Tests for GET /api/v1/system/backends."""

    def test_list_backends(self, client: TestClient):
        """Test listing training backends."""
        mock_backends = [
            {
                "id": "unsloth",
                "name": "Unsloth",
                "description": "Fast training backend",
                "available": True,
            },
            {
                "id": "transformers",
                "name": "Transformers",
                "description": "Standard backend",
                "available": True,
            },
        ]

        with patch("model_garden.training.backends.list_backends", return_value=mock_backends):
            response = client.get("/api/v1/system/backends")
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True
            assert data["total"] == 2


class TestCleanupGPU:
    """Tests for POST /api/v1/system/cleanup."""

    def test_cleanup_gpu_no_cuda(self, client: TestClient):
        """Test GPU cleanup when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("gc.collect", return_value=10):
                response = client.post("/api/v1/system/cleanup")
                assert response.status_code == 200

                data = response.json()
                assert data["success"] is True
                assert "CUDA not available" in str(data["actions"])

    def test_cleanup_gpu_with_cuda(self, client: TestClient):
        """Test GPU cleanup with CUDA available."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.synchronize"):
                with patch("torch.cuda.memory_allocated", side_effect=[1_000_000_000, 500_000_000]):
                    with patch("torch.cuda.empty_cache"):
                        with patch("gc.collect", return_value=10):
                            response = client.post("/api/v1/system/cleanup")
                            assert response.status_code == 200

                            data = response.json()
                            assert data["success"] is True


class TestGetSettings:
    """Tests for GET /api/v1/system/settings."""

    def test_get_settings_no_unsloth(self, client: TestClient):
        """Test getting settings when Unsloth is not installed."""
        with patch("model_garden.utils.optional_deps.is_unsloth_installed", return_value=False):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1)

                response = client.get("/api/v1/system/settings")
                assert response.status_code == 200

                data = response.json()
                assert data["success"] is True
                assert "optional_dependencies" in data["data"]
                assert data["data"]["optional_dependencies"]["unsloth"]["installed"] is False

    def test_get_settings_with_unsloth(self, client: TestClient):
        """Test getting settings when Unsloth is installed."""
        mock_unsloth = MagicMock()
        mock_unsloth.__version__ = "2024.12.0"

        with patch("model_garden.utils.optional_deps.is_unsloth_installed", return_value=True):
            with patch.dict("sys.modules", {"unsloth": mock_unsloth}):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=1)

                    response = client.get("/api/v1/system/settings")
                    assert response.status_code == 200

                    data = response.json()
                    assert data["data"]["optional_dependencies"]["unsloth"]["installed"] is True


class TestInstallUnsloth:
    """Tests for POST /api/v1/system/unsloth/install."""

    def test_install_already_installed(self, client: TestClient):
        """Test installing when Unsloth is already installed."""
        with patch("model_garden.utils.optional_deps.is_unsloth_installed", return_value=True):
            response = client.post("/api/v1/system/unsloth/install")
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is False
            assert "already installed" in data["message"]

    def test_install_starts_background_task(self, client: TestClient):
        """Test that install starts a background task."""
        with patch("model_garden.utils.optional_deps.is_unsloth_installed", return_value=False):
            # Reset operation status
            import model_garden.api.routes.system as system_module

            system_module._package_operation_status["in_progress"] = False

            response = client.post("/api/v1/system/unsloth/install")
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True
            assert "installation started" in data["message"].lower()

    def test_install_operation_in_progress(self, client: TestClient):
        """Test installing when another operation is in progress."""
        with patch("model_garden.utils.optional_deps.is_unsloth_installed", return_value=False):
            # Set operation in progress
            import model_garden.api.routes.system as system_module

            system_module._package_operation_status["in_progress"] = True
            system_module._package_operation_status["operation"] = "other_operation"

            try:
                response = client.post("/api/v1/system/unsloth/install")
                assert response.status_code == 200

                data = response.json()
                assert data["success"] is False
                assert "Another package operation" in data["message"]
            finally:
                system_module._package_operation_status["in_progress"] = False


class TestUninstallUnsloth:
    """Tests for POST /api/v1/system/unsloth/uninstall."""

    def test_uninstall_not_installed(self, client: TestClient):
        """Test uninstalling when Unsloth is not installed."""
        with patch("model_garden.utils.optional_deps.is_unsloth_installed", return_value=False):
            response = client.post("/api/v1/system/unsloth/uninstall")
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is False
            assert "not installed" in data["message"]

    def test_uninstall_starts_background_task(self, client: TestClient):
        """Test that uninstall starts a background task."""
        with patch("model_garden.utils.optional_deps.is_unsloth_installed", return_value=True):
            # Reset operation status
            import model_garden.api.routes.system as system_module

            system_module._package_operation_status["in_progress"] = False

            response = client.post("/api/v1/system/unsloth/uninstall")
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True
            assert "uninstallation started" in data["message"].lower()


class TestUnslothOperationStatus:
    """Tests for GET /api/v1/system/unsloth/status."""

    def test_get_operation_status(self, client: TestClient):
        """Test getting package operation status."""
        import model_garden.api.routes.system as system_module

        system_module._package_operation_status = {
            "in_progress": False,
            "operation": None,
            "output": [],
            "success": True,
            "error": None,
        }

        response = client.get("/api/v1/system/unsloth/status")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "data" in data


class TestRestartService:
    """Tests for POST /api/v1/system/restart."""

    def test_restart_not_systemd(self, client: TestClient):
        """Test restart when not running as systemd service."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)

            response = client.post("/api/v1/system/restart")
            assert response.status_code == 400
            assert "not running as a systemd service" in response.json()["detail"]

    def test_restart_no_sudo(self, client: TestClient):
        """Test restart when passwordless sudo is not available."""
        with patch("subprocess.run") as mock_run:
            # First call: service is active
            # Second call: sudo test fails
            mock_run.side_effect = [
                MagicMock(returncode=0),  # systemctl is-active
                MagicMock(returncode=1, stdout=""),  # sudo -n -l
            ]

            response = client.post("/api/v1/system/restart")
            assert response.status_code == 403
            assert "Passwordless sudo not configured" in response.json()["detail"]

    def test_restart_success(self, client: TestClient):
        """Test successful service restart."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0),  # systemctl is-active
                MagicMock(
                    returncode=0,
                    stdout="/usr/bin/systemctl restart model-garden.service",
                ),  # sudo -n -l
            ]
            with patch("subprocess.Popen"):
                response = client.post("/api/v1/system/restart")
                assert response.status_code == 200

                data = response.json()
                assert data["success"] is True
                assert "restart initiated" in data["message"]


class TestRunPackageCommand:
    """Tests for background package command execution."""

    def test_run_package_command_success(self):
        """Test successful package command execution."""
        from model_garden.api.routes import system as system_module
        from model_garden.api.routes.system import _run_package_command

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = iter(["Installing...\n", "Done\n"])
            mock_process.wait.return_value = None
            mock_process.returncode = 0
            mock_popen.return_value = mock_process

            _run_package_command(["uv", "pip", "install", "test"], "test_install")

            assert system_module._package_operation_status["success"] is True
            assert system_module._package_operation_status["in_progress"] is False

    def test_run_package_command_failure(self):
        """Test failed package command execution."""
        from model_garden.api.routes import system as system_module
        from model_garden.api.routes.system import _run_package_command

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = iter(["Error: package not found\n"])
            mock_process.wait.return_value = None
            mock_process.returncode = 1
            mock_popen.return_value = mock_process

            _run_package_command(["uv", "pip", "install", "invalid"], "test_install")

            assert system_module._package_operation_status["success"] is False
            assert system_module._package_operation_status["error"] is not None

    def test_run_package_commands_sequence(self):
        """Test running multiple package commands in sequence."""
        from model_garden.api.routes import system as system_module
        from model_garden.api.routes.system import _run_package_commands_sequence

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = iter(["Step 1...\n", "Step 2...\n"])
            mock_process.wait.return_value = None
            mock_process.returncode = 0
            mock_popen.return_value = mock_process

            commands = [
                ["uv", "pip", "uninstall", "pkg"],
                ["uv", "pip", "install", "new-pkg"],
            ]
            _run_package_commands_sequence(commands, "test_sequence")

            assert system_module._package_operation_status["success"] is True
            assert system_module._package_operation_status["in_progress"] is False
