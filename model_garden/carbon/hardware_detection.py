"""Hardware detection utilities for detailed emissions reporting."""

import os
import platform
import subprocess
from typing import Dict, Optional, List


class HardwareDetector:
    """Detect hardware and system information."""
    
    def __init__(self):
        """Initialize hardware detector."""
        self._gpu_info_cache: Optional[Dict] = None
        self._cpu_info_cache: Optional[Dict] = None
        self._system_info_cache: Optional[Dict] = None
    
    def get_gpu_info(self) -> Optional[Dict]:
        """
        Detect GPU information using nvidia-smi.
        
        Returns:
            Dictionary with GPU details or None if no GPU detected
        """
        if self._gpu_info_cache is not None:
            return self._gpu_info_cache
        
        try:
            # Try nvidia-smi command
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = []
                
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 3:
                            gpu_name = parts[0]
                            driver_version = parts[1]
                            memory = parts[2]
                            
                            # Extract manufacturer and model
                            manufacturer = "NVIDIA"
                            model = gpu_name.replace("NVIDIA ", "").replace("GeForce ", "")
                            
                            # Determine family (simplified)
                            family = "Unknown"
                            if "RTX" in gpu_name:
                                family = "RTX"
                            elif "GTX" in gpu_name:
                                family = "GTX"
                            elif "Tesla" in gpu_name:
                                family = "Tesla"
                            elif "A100" in gpu_name or "A40" in gpu_name or "A30" in gpu_name:
                                family = "Ampere"
                            elif "V100" in gpu_name:
                                family = "Volta"
                            elif "H100" in gpu_name:
                                family = "Hopper"
                            
                            gpus.append({
                                "manufacturer": manufacturer,
                                "model": model,
                                "family": family,
                                "driver_version": driver_version,
                                "memory": memory,
                                "full_name": gpu_name
                            })
                
                if gpus:
                    self._gpu_info_cache = {
                        "count": len(gpus),
                        "gpus": gpus,
                        "primary": gpus[0]  # First GPU as primary
                    }
                    return self._gpu_info_cache
        
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            print(f"Could not detect GPU: {e}")
        
        return None
    
    def get_cpu_info(self) -> Dict:
        """
        Detect CPU information.
        
        Returns:
            Dictionary with CPU details
        """
        if self._cpu_info_cache is not None:
            return self._cpu_info_cache
        
        cpu_info = {
            "manufacturer": "Unknown",
            "model": "Unknown",
            "family": "Unknown",
            "cores": os.cpu_count() or 1,
            "architecture": platform.machine()
        }
        
        try:
            # Try to get detailed CPU info on Linux
            if platform.system() == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    
                    # Extract model name
                    for line in cpuinfo.split('\n'):
                        if 'model name' in line.lower():
                            model_name = line.split(':')[1].strip()
                            cpu_info["full_name"] = model_name
                            
                            # Determine manufacturer
                            if "Intel" in model_name:
                                cpu_info["manufacturer"] = "Intel"
                                # Extract family (e.g., "Core i7", "Xeon")
                                if "Core" in model_name:
                                    for part in model_name.split():
                                        if part.startswith(('i3', 'i5', 'i7', 'i9')):
                                            cpu_info["family"] = f"Core {part}"
                                            break
                                elif "Xeon" in model_name:
                                    cpu_info["family"] = "Xeon"
                                cpu_info["model"] = model_name.replace("Intel(R) ", "").replace("(R)", "").replace("(TM)", "").strip()
                            
                            elif "AMD" in model_name:
                                cpu_info["manufacturer"] = "AMD"
                                if "Ryzen" in model_name:
                                    cpu_info["family"] = "Ryzen"
                                elif "EPYC" in model_name:
                                    cpu_info["family"] = "EPYC"
                                elif "Threadripper" in model_name:
                                    cpu_info["family"] = "Threadripper"
                                cpu_info["model"] = model_name.replace("AMD ", "").strip()
                            
                            break
            
            elif platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    model_name = result.stdout.strip()
                    cpu_info["full_name"] = model_name
                    
                    if "Intel" in model_name:
                        cpu_info["manufacturer"] = "Intel"
                        cpu_info["model"] = model_name.replace("Intel ", "").strip()
                    elif "Apple" in model_name:
                        cpu_info["manufacturer"] = "Apple"
                        if "M1" in model_name:
                            cpu_info["family"] = "M1"
                        elif "M2" in model_name:
                            cpu_info["family"] = "M2"
                        elif "M3" in model_name:
                            cpu_info["family"] = "M3"
                        cpu_info["model"] = model_name
            
            elif platform.system() == "Windows":
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'name'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        model_name = lines[1].strip()
                        cpu_info["full_name"] = model_name
                        
                        if "Intel" in model_name:
                            cpu_info["manufacturer"] = "Intel"
                            cpu_info["model"] = model_name.replace("Intel(R) ", "").replace("(R)", "").replace("(TM)", "").strip()
                        elif "AMD" in model_name:
                            cpu_info["manufacturer"] = "AMD"
                            cpu_info["model"] = model_name.replace("AMD ", "").strip()
        
        except Exception as e:
            print(f"Could not detect detailed CPU info: {e}")
        
        self._cpu_info_cache = cpu_info
        return cpu_info
    
    def get_system_info(self) -> Dict:
        """
        Detect system information.
        
        Returns:
            Dictionary with OS and system details
        """
        if self._system_info_cache is not None:
            return self._system_info_cache
        
        system_info = {
            "os_name": platform.system(),
            "os_version": platform.release(),
            "os_full": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.machine()
        }
        
        # Try to get more specific Linux distribution info
        if platform.system() == "Linux":
            try:
                # Try to read /etc/os-release
                with open('/etc/os-release', 'r') as f:
                    os_release = {}
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            os_release[key] = value.strip('"')
                    
                    if 'PRETTY_NAME' in os_release:
                        system_info["os_distribution"] = os_release['PRETTY_NAME']
                    elif 'NAME' in os_release:
                        system_info["os_distribution"] = os_release['NAME']
                        if 'VERSION' in os_release:
                            system_info["os_distribution"] += f" {os_release['VERSION']}"
            except Exception:
                pass
        
        self._system_info_cache = system_info
        return system_info
    
    def get_ram_info(self) -> Dict:
        """
        Detect RAM information.
        
        Returns:
            Dictionary with RAM details
        """
        ram_info: Dict = {
            "total_gb": 0.0,
            "available_gb": 0.0
        }
        
        try:
            if platform.system() == "Linux":
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if 'MemTotal:' in line:
                            # Convert KB to GB
                            kb = int(line.split()[1])
                            ram_info["total_gb"] = round(kb / (1024 * 1024), 1)
                        elif 'MemAvailable:' in line:
                            kb = int(line.split()[1])
                            ram_info["available_gb"] = round(kb / (1024 * 1024), 1)
            
            elif platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    bytes_total = int(result.stdout.strip())
                    ram_info["total_gb"] = round(bytes_total / (1024**3), 1)
            
            elif platform.system() == "Windows":
                result = subprocess.run(
                    ['wmic', 'computersystem', 'get', 'totalphysicalmemory'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        bytes_total = int(lines[1].strip())
                        ram_info["total_gb"] = round(bytes_total / (1024**3), 1)
        
        except Exception as e:
            print(f"Could not detect RAM info: {e}")
        
        return ram_info
    
    def get_full_hardware_report(self) -> Dict:
        """
        Get complete hardware report.
        
        Returns:
            Dictionary with all hardware information
        """
        return {
            "gpu": self.get_gpu_info(),
            "cpu": self.get_cpu_info(),
            "system": self.get_system_info(),
            "ram": self.get_ram_info()
        }


# Global instance
_hardware_detector: Optional[HardwareDetector] = None


def get_hardware_detector() -> HardwareDetector:
    """Get the global hardware detector instance."""
    global _hardware_detector
    if _hardware_detector is None:
        _hardware_detector = HardwareDetector()
    return _hardware_detector
