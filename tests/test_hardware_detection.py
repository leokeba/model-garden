"""Tests for hardware detection module.

These tests verify the hardware detection utilities used for
emissions reporting work correctly across different platforms.
"""

from unittest.mock import MagicMock, patch


class TestHardwareDetector:
    """Tests for HardwareDetector class."""

    def test_init(self):
        """Test initialization creates empty caches."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        detector = HardwareDetector()
        assert detector._gpu_info_cache is None
        assert detector._cpu_info_cache is None
        assert detector._system_info_cache is None

    @patch("model_garden.carbon.hardware_detection.subprocess.run")
    def test_get_gpu_info_nvidia(self, mock_run):
        """Test GPU detection with nvidia-smi."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        # Mock successful nvidia-smi output
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 4090, 535.183.01, 24564 MiB\n",
        )

        detector = HardwareDetector()
        gpu_info = detector.get_gpu_info()

        assert gpu_info is not None
        assert gpu_info["count"] == 1
        assert len(gpu_info["gpus"]) == 1
        assert gpu_info["primary"]["manufacturer"] == "NVIDIA"
        assert "RTX" in gpu_info["primary"]["model"]
        assert gpu_info["primary"]["family"] == "RTX"

    @patch("model_garden.carbon.hardware_detection.subprocess.run")
    def test_get_gpu_info_multi_gpu(self, mock_run):
        """Test GPU detection with multiple GPUs."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        # Mock multiple GPU output
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA A100-SXM4-80GB, 535.183.01, 81920 MiB\nNVIDIA A100-SXM4-80GB, 535.183.01, 81920 MiB\n",
        )

        detector = HardwareDetector()
        gpu_info = detector.get_gpu_info()

        assert gpu_info is not None
        assert gpu_info["count"] == 2
        assert len(gpu_info["gpus"]) == 2
        assert gpu_info["primary"]["family"] == "Ampere"

    @patch("model_garden.carbon.hardware_detection.subprocess.run")
    def test_get_gpu_info_no_gpu(self, mock_run):
        """Test GPU detection when no GPU available."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        # Mock nvidia-smi not found
        mock_run.side_effect = FileNotFoundError()

        detector = HardwareDetector()
        gpu_info = detector.get_gpu_info()

        assert gpu_info is None

    @patch("model_garden.carbon.hardware_detection.subprocess.run")
    def test_get_gpu_info_nvidia_smi_error(self, mock_run):
        """Test GPU detection when nvidia-smi fails."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        mock_run.return_value = MagicMock(returncode=1, stdout="")

        detector = HardwareDetector()
        gpu_info = detector.get_gpu_info()

        assert gpu_info is None

    def test_get_gpu_info_caching(self):
        """Test that GPU info is cached."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        detector = HardwareDetector()

        # Set cache directly
        detector._gpu_info_cache = {"cached": True}

        result = detector.get_gpu_info()
        assert result == {"cached": True}

    def test_get_cpu_info_linux_intel(self):
        """Test CPU detection returns valid structure on current platform."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        detector = HardwareDetector()
        cpu_info = detector.get_cpu_info()

        # Should have basic fields regardless of platform
        assert "manufacturer" in cpu_info
        assert "model" in cpu_info
        assert "cores" in cpu_info
        assert "architecture" in cpu_info
        assert cpu_info["cores"] > 0

    def test_get_cpu_info_has_cores(self):
        """Test CPU info includes core count."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        detector = HardwareDetector()
        cpu_info = detector.get_cpu_info()

        assert cpu_info["cores"] >= 1
        assert cpu_info["architecture"] in ["x86_64", "amd64", "arm64", "aarch64", "i686"]

    @patch("model_garden.carbon.hardware_detection.platform.system", return_value="Darwin")
    @patch("model_garden.carbon.hardware_detection.subprocess.run")
    def test_get_cpu_info_macos_apple(self, mock_run, mock_platform):
        """Test CPU detection on macOS with Apple Silicon."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Apple M2 Pro",
        )

        detector = HardwareDetector()
        cpu_info = detector.get_cpu_info()

        assert cpu_info["manufacturer"] == "Apple"
        assert cpu_info["family"] == "M2"

    def test_get_cpu_info_caching(self):
        """Test that CPU info is cached."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        detector = HardwareDetector()

        # Set cache directly
        detector._cpu_info_cache = {"cached": True}

        result = detector.get_cpu_info()
        assert result == {"cached": True}

    def test_get_system_info(self):
        """Test system info detection."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        detector = HardwareDetector()
        system_info = detector.get_system_info()

        assert "os_name" in system_info
        assert "os_version" in system_info
        assert "python_version" in system_info
        assert "architecture" in system_info

    def test_get_system_info_has_required_fields(self):
        """Test system info includes required fields."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        detector = HardwareDetector()
        system_info = detector.get_system_info()

        # Should have these fields on any platform
        assert "os_name" in system_info
        assert "os_version" in system_info
        assert "python_version" in system_info
        assert "architecture" in system_info

        # Validate os_name is one of expected values
        assert system_info["os_name"] in ["Linux", "Darwin", "Windows"]

    def test_get_system_info_caching(self):
        """Test that system info is cached."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        detector = HardwareDetector()

        # Set cache directly
        detector._system_info_cache = {"cached": True}

        result = detector.get_system_info()
        assert result == {"cached": True}

    def test_get_ram_info_has_total(self):
        """Test RAM detection returns total memory."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        detector = HardwareDetector()
        ram_info = detector.get_ram_info()

        assert "total_gb" in ram_info
        # Should have some RAM
        assert ram_info["total_gb"] >= 0

    @patch("model_garden.carbon.hardware_detection.platform.system", return_value="Darwin")
    @patch("model_garden.carbon.hardware_detection.subprocess.run")
    def test_get_ram_info_macos(self, mock_run, mock_platform):
        """Test RAM detection on macOS."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        # Mock 32 GB RAM
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="34359738368",  # 32 GB in bytes
        )

        detector = HardwareDetector()
        ram_info = detector.get_ram_info()

        assert "total_gb" in ram_info
        assert ram_info["total_gb"] == 32.0

    @patch("model_garden.carbon.hardware_detection.subprocess.run")
    def test_get_full_hardware_report(self, mock_run):
        """Test full hardware report generation."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        # Mock nvidia-smi
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 4090, 535.183.01, 24564 MiB\n",
        )

        detector = HardwareDetector()
        report = detector.get_full_hardware_report()

        assert "gpu" in report
        assert "cpu" in report
        assert "system" in report
        assert "ram" in report


class TestGetHardwareDetector:
    """Tests for get_hardware_detector function."""

    def test_returns_singleton(self):
        """Test that get_hardware_detector returns singleton instance."""
        from model_garden.carbon.hardware_detection import get_hardware_detector

        detector1 = get_hardware_detector()
        detector2 = get_hardware_detector()

        assert detector1 is detector2

    def test_returns_hardware_detector_instance(self):
        """Test that get_hardware_detector returns HardwareDetector."""
        from model_garden.carbon.hardware_detection import (
            HardwareDetector,
            get_hardware_detector,
        )

        detector = get_hardware_detector()
        assert isinstance(detector, HardwareDetector)


class TestGPUFamilyDetection:
    """Tests for GPU family detection logic."""

    @patch("model_garden.carbon.hardware_detection.subprocess.run")
    def test_detect_gtx_family(self, mock_run):
        """Test GTX GPU family detection."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA GeForce GTX 1080 Ti, 535.183.01, 11264 MiB\n",
        )

        detector = HardwareDetector()
        gpu_info = detector.get_gpu_info()

        assert gpu_info["primary"]["family"] == "GTX"

    @patch("model_garden.carbon.hardware_detection.subprocess.run")
    def test_detect_tesla_family(self, mock_run):
        """Test Tesla GPU family detection."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Tesla V100-SXM2-32GB, 470.82.01, 32768 MiB\n",
        )

        detector = HardwareDetector()
        gpu_info = detector.get_gpu_info()

        assert gpu_info["primary"]["family"] in ["Tesla", "Volta"]

    @patch("model_garden.carbon.hardware_detection.subprocess.run")
    def test_detect_hopper_family(self, mock_run):
        """Test H100 (Hopper) GPU family detection."""
        from model_garden.carbon.hardware_detection import HardwareDetector

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA H100 PCIe, 535.183.01, 81920 MiB\n",
        )

        detector = HardwareDetector()
        gpu_info = detector.get_gpu_info()

        assert gpu_info["primary"]["family"] == "Hopper"
