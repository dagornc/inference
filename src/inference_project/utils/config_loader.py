import os
from typing import Any

import yaml


def load_config(step_name: str, config_path: str = "config") -> dict[str, Any]:
    """
    Loads and merges global and step-specific YAML configurations.

    Args:
        step_name (str): The name of the step (e.g., '01_embedding').
        config_path (str): The path to the config directory.

    Returns:
        Dict[str, Any]: The merged configuration.
    """
    global_config_path = os.path.join(config_path, "global.yaml")
    step_config_path = os.path.join(config_path, f"{step_name}.yaml")

    # Load global config
    with open(global_config_path) as f:
        global_config_str = f.read()
    global_config = yaml.safe_load(os.path.expandvars(global_config_str))

    # Load step-specific config
    with open(step_config_path) as f:
        step_config_str = f.read()
    step_config = yaml.safe_load(os.path.expandvars(step_config_str))

    # Merge configs (step-specific overrides global)
    merged_config = {**global_config, **step_config}

    return merged_config


if __name__ == "__main__":
    # Example usage:
    # Make sure to set the OPENAI_API_KEY environment variable
    # export OPENAI_API_KEY="your_key_here"
    config = load_config("01_embedding")
    print(config)
