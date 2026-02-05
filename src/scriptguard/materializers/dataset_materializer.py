"""Custom materializer for HuggingFace Dataset objects.

This materializer properly handles HuggingFace Dataset serialization/deserialization
and fixes path issues on Windows.
"""
from pathlib import Path
from typing import Type, Any

from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType
from datasets import Dataset, load_from_disk


class HuggingFaceDatasetMaterializer(BaseMaterializer):
    """Materializer for HuggingFace Dataset objects.

    Uses HuggingFace's native save_to_disk/load_from_disk methods
    which properly handle cross-platform paths.
    """

    ASSOCIATED_TYPES = (Dataset,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[Any]) -> Dataset:
        """Load a HuggingFace Dataset from disk.

        Args:
            data_type: The type of the data to load.

        Returns:
            The loaded Dataset object.
        """
        # Fix Windows path issues - normalize the path
        artifact_path = str(self.uri).replace('\\\\', '\\').replace('/', '\\')

        # The artifact path is a directory
        dataset_path = Path(artifact_path)

        # Load using HuggingFace's native loader
        try:
            dataset = load_from_disk(str(dataset_path))
            return dataset
        except Exception as e:
            # If direct loading fails, try to find the dataset directory
            # Sometimes ZenML adds extra subdirectories
            if dataset_path.is_dir():
                # Look for dataset_dict.json or dataset_info.json
                for subdir in dataset_path.rglob("*"):
                    if subdir.is_dir() and (
                        (subdir / "dataset_dict.json").exists() or
                        (subdir / "dataset_info.json").exists()
                    ):
                        dataset = load_from_disk(str(subdir))
                        return dataset

            # If still fails, raise the original error
            raise RuntimeError(
                f"Failed to load HuggingFace Dataset from {dataset_path}: {e}"
            ) from e

    def save(self, data: Dataset) -> None:
        """Save a HuggingFace Dataset to disk.

        Args:
            data: The Dataset to save.
        """
        # Fix Windows path issues - normalize the path
        artifact_path = str(self.uri).replace('\\\\', '\\').replace('/', '\\')

        # Create the directory if it doesn't exist
        dataset_path = Path(artifact_path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Save using HuggingFace's native saver
        try:
            data.save_to_disk(str(dataset_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to save HuggingFace Dataset to {dataset_path}: {e}"
            ) from e
