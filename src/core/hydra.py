from pathlib import Path
from typing import List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, DictConfig


class HydraContainer:
    """
    Compose Hydra configs programmatically (i.e. outside @hydra.main).
    Useful for notebooks, integration tests, and custom runners.
    """

    def __init__(self, config_dir=None):
        if config_dir is None:
            project = Path(__file__).resolve().parents[1]
            self.config_dir = str((project / "configs").resolve())
        else:
            self.config_dir = str(Path(config_dir).resolve())

        self._cfg = None

    def compose(
        self,
        config_name: str = "defaults",
        overrides: Optional[List[str]] = None,
        return_dictconfig: bool = True,
    ):
        """
        Compose config like CLI: python -m src.main model=ecg train=single_gpu_l
        """
        overrides = overrides or []

        initialize_config_dir(config_dir=self.config_dir, job_name="container_compose")
        cfg = compose(config_name=config_name, overrides=overrides)

        self._cfg = cfg
        return cfg if return_dictconfig else OmegaConf.to_container(cfg)

    @property
    def cfg(self) -> DictConfig:
        if self._cfg is None:
            raise RuntimeError("Config not composed yet. Call compose() first.")
        return self._cfg

    def print(self):
        if self._cfg is None:
            raise RuntimeError("Config not composed yet.")
        print(OmegaConf.to_yaml(self._cfg))
