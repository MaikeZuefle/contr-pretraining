import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

SOURCE_LANGUAGE_PLACEHOLDER = "<source-language>"
TARGET_LANGUAGE_PLACEHOLDER = "<target-language>"

LANGUAGES_CODE_NAME = {
    "en": {
        "en": "english",
        "es": "spanish",
        "fr": "french",
        "de": "german",
        "it": "italian",
    },
    "es": {
        "en": "inglés",
        "es": "español",
        "fr": "francés",
        "de": "alemán",
        "it": "italiano",
    },
    "fr": {
        "en": "anglais",
        "es": "espagnol",
        "fr": "français",
        "de": "allemand",
        "it": "italien",
    },
    "de": {
        "en": "englisch",
        "es": "spanisch",
        "fr": "französisch",
        "de": "deutsch",
        "it": "italienisch",
    },
    "it": {
        "en": "inglese",
        "es": "spagnolo",
        "fr": "francese",
        "de": "tedesco",
        "it": "italiano",
    },
}


@dataclass
class BaseConfig:
    DATA: Dict
    OUTPUTS_TEXT_LIST: Optional[Dict] = None
    INPUTS_TEXT_LIST: Optional[Dict] = None


@dataclass
class CustomDatasetConfig:
    task: str
    datapath: str
    partitions: Dict
    languages: Optional[List[str]] = None
    OUTPUTS_TEXT_LIST: Optional[Dict] = None
    INPUTS_TEXT_LIST: Optional[Dict] = None
    audio_nwp: Optional[bool]=False
    no_punctuation: Optional[bool]=False


class EnvVarSafeLoader(yaml.SafeLoader):
    """
    A YAML SafeLoader to process environment variables in YAML.
    """

    def __init__(self, stream):
        super().__init__(stream)
        self.add_implicit_resolver(
            "!env_variable", re.compile(r"\$\{[^}]+\}"), None
        )
        self.add_constructor("!env_variable", type(self).env_constructor)

    @staticmethod
    def env_constructor(loader, node):
        """
        Constructor that replaces ${VAR_NAME} with the value of the VAR_NAME environment variable.
        """
        value = loader.construct_scalar(node)
        pattern = re.compile(r"\$\{([^}]+)\}")
        match = pattern.findall(value)
        if match:
            for var_name in match:
                env_value = os.getenv(var_name)
                if env_value is not None:
                    value = value.replace(f"${{{var_name}}}", env_value)
        return value


def safe_load_with_env_vars(stream):
    """
    Load a YAML file with environment variables.
    """
    return yaml.load(stream, EnvVarSafeLoader)


class DatasetsWrapperConfig:
    def __init__(
        self,
        config_path: str = None,
    ):
        self.config_path = config_path

    def from_yml(self):
        with open(self.config_path, "r") as f:
            config = safe_load_with_env_vars(f)
        for key in ["OUTPUTS_TEXT_LIST", "INPUTS_TEXT_LIST"]:
            if key in config and config[key] is not None:
                with open(f"{config[key]}", "r") as f:
                    config[key] = json.load(f)
            else:
                config[key] = None
        return BaseConfig(**config)
