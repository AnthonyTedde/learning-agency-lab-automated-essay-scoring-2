import os
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
import torch.utils
import torch.utils.data

DATABRICKS_STR: str = "DATABRICKS"
KAGGLE_STR: str = "KAGGLE"
LOCAL_STR: str = "LOCAL"
MATPLOTBLUE: str = "#1f77b4"
SEED: int = 1010
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DEBERTA_V3_CKPT: str = "microsoft/deberta-v3-base"
DISTILBERT: str = "distilbert/distilbert-base-uncased"
NUM_LABELS: str = 5
DATALOADER_BATCH: str = 64


@dataclass(frozen=True)
class ConfigurationSetting:
    """
    Configuration settings for a machine learning environment.

    Attributes:
        name (str): The name of the environment (e.g., "databricks", "kaggle", "local").
        model_ckpt (str): The model checkpoint to use.
        num_labels (int): The number of labels for the task.
        n_worker (int): The number of workers for data loading.
        plot_color (str): The color to use for plotting.
        seed (int): The seed value for random number generation.
        data_path (Union[Path, None]): The path to the data directory or None.
        device (str): The device to use for computations (e.g., "cpu", "cuda").
        torch_device (torch.device): The torch device object created based on the `device` attribute.
        dataloader_batch (int): The batch size for the data loader.
    """
    name: str
    model_ckpt: str
    num_labels: int
    n_worker: int
    plot_color: str
    seed: int
    data_path: Optional[Path]
    device: str
    torch_device: torch.device = field(init=False)
    dataloader_batch: int = DATALOADER_BATCH

    def __post_init__(self):
        """
        Initialize the torch_device attribute based on the device string.

        This method is called automatically after the dataclass is initialized.
        Since the dataclass is frozen, `object.__setattr__` is used to set the
        value of the torch_device attribute.
        """
        object.__setattr__(self, "torch_device", torch.device(self.device))


def configuration_builder(
        model_ckpt="google-bert/bert-base-uncased",
        plot_color="#FFFFFF",
        seed=1010,
        device=None
) -> ConfigurationSetting:
    """
    Builds and returns a configuration setting for the environment.

    Parameters:
        model_ckpt (str): The model checkpoint to use. Default is "google-bert/bert-base-uncased".
        plot_color (str): The color to use for plotting. Default is "#FFFFFF".
        seed (int): The seed value for random number generation. Default is 1010.
        device (Optional[str]): The device to use for computations (e.g., "cpu", "cuda").
                                If None, defaults to "cpu". Default is None.

    Returns:
        ConfigurationSetting: An instance of ConfigurationSetting with the specified parameters.
    """
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        environment_name = DATABRICKS_STR
        n_worker = 8
        data_path = None
    elif os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        environment_name = KAGGLE_STR
        n_worker = 2
        data_path = Path(
            "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
        )
    else:
        environment_name = LOCAL_STR
        n_worker = math.floor(os.cpu_count() * 3 / 8)  # type: ignore
        data_path = Path("../data")
    return ConfigurationSetting(
        name=environment_name,
        model_ckpt=model_ckpt,
        num_labels=5,
        n_worker=n_worker,
        plot_color=plot_color,
        seed=seed,
        data_path=data_path,
        device="cpu" if device is None else device
    )