import os
from typing import Optional

from codecarbon import EmissionsTracker
from codecarbon.external.hardware import CPU, GB_TO_B, GPU
from codecarbon.external.ram import RAM
from omegaconf import DictConfig, OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def resolve_codecarbon_gpu_ids(cfg: DictConfig) -> str:
    """Resolve physical GPU IDs that training will use for CodeCarbon monitoring.

    Priority:
    1. Explicit ``codecarbon.gpu_ids`` (unless ``null`` or ``auto``)
    2. ``CUDA_VISIBLE_DEVICES`` / ``ROCR_VISIBLE_DEVICES``
    3. Lightning ``trainer.devices``
    """
    gpu_ids = OmegaConf.select(cfg, "codecarbon.gpu_ids")
    if gpu_ids not in (None, "", "auto"):
        return str(gpu_ids)

    for env_var in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        visible_devices = os.environ.get(env_var, "").strip()
        if visible_devices:
            log.info(
                f"CodeCarbon GPU auto-detection: using {env_var}={visible_devices}"
            )
            return visible_devices

    trainer_cfg = cfg.get("trainer") or {}
    accelerator = OmegaConf.select(trainer_cfg, "accelerator", default="auto")
    if accelerator == "cpu":
        return "0"

    devices = OmegaConf.select(trainer_cfg, "devices", default="auto")
    if devices in ("auto", 1):
        log.info("CodeCarbon GPU auto-detection: defaulting to GPU 0")
        return "0"

    if isinstance(devices, int):
        resolved = ",".join(str(i) for i in range(devices))
        log.info(f"CodeCarbon GPU auto-detection: using trainer devices 0..{devices - 1}")
        return resolved

    try:
        device_list = [int(d) for d in devices]
    except (TypeError, ValueError):
        log.info("CodeCarbon GPU auto-detection: defaulting to GPU 0")
        return "0"

    if not device_list:
        log.info("CodeCarbon GPU auto-detection: defaulting to GPU 0")
        return "0"

    resolved = ",".join(str(d) for d in device_list)
    log.info(f"CodeCarbon GPU auto-detection: using trainer devices [{resolved}]")
    return resolved


def patch_codecarbon_gpu_monitoring() -> None:
    """Fix CodeCarbon IndexError when monitoring non-zero GPU indices.

    CodeCarbon 3.2.x indexes ``_gpu_details_history`` with the physical GPU
    index from ``enumerate(gpu_details)`` instead of the index within the
    monitored GPU list.
    """
    if getattr(EmissionsTracker, "_gpu_monitoring_patched", False):
        return

    def _monitor_power(self) -> None:
        with self._scheduler_monitor_lock:
            for hardware in self._hardware:
                if isinstance(hardware, CPU):
                    hardware.monitor_power()
                    self._cpu_utilization_history.append(hardware.extra_data()["cpu_load"])
                elif isinstance(hardware, RAM):
                    extra_data = hardware.extra_data()
                    self._ram_utilization_history.append(extra_data["percent"])
                    self._ram_used_history.append(extra_data["used"])
                elif isinstance(hardware, GPU):
                    gpu_ids_to_monitor = hardware.gpu_ids
                    gpu_details = hardware.devices.get_gpu_details()
                    for gpu_index, gpu_detail in enumerate(gpu_details):
                        resolved_gpu_index = gpu_detail.get("gpu_index", gpu_index)
                        if resolved_gpu_index in gpu_ids_to_monitor:
                            monitoring_index = gpu_ids_to_monitor.index(resolved_gpu_index)
                            for key in (
                                "gpu_utilization",
                                "temperature",
                                "fan_percent",
                                "power_limit",
                            ):
                                self._gpu_details_history[key][monitoring_index].append(
                                    gpu_detail[key]
                                )
                            self._gpu_details_history["used_memory"][monitoring_index].append(
                                gpu_detail["used_memory"] / GB_TO_B
                            )

    EmissionsTracker._monitor_power = _monitor_power
    EmissionsTracker._gpu_monitoring_patched = True


def configure_codecarbon_gpu_ids(cfg: DictConfig) -> Optional[str]:
    """Resolve and apply CodeCarbon GPU IDs, including upstream bug workaround."""
    resolved_gpu_ids = resolve_codecarbon_gpu_ids(cfg)
    cfg.codecarbon.gpu_ids = resolved_gpu_ids
    patch_codecarbon_gpu_monitoring()
    log.info(f"CodeCarbon will monitor GPU(s): {resolved_gpu_ids}")
    return resolved_gpu_ids
