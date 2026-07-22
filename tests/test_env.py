"""Validate that required Praroop-AI dependencies import correctly."""

import importlib

import pytest

REQUIRED_MODULES = [
    ("cv2", "opencv-python"),
    ("PIL", "Pillow"),
    ("numpy", "numpy"),
    ("pytest", "pytest"),
    ("ultralytics", "ultralytics"),
    ("easyocr", "easyocr"),
]


@pytest.mark.parametrize("module_name,package_name", REQUIRED_MODULES)
def test_library_import(module_name: str, package_name: str) -> None:
    """Ensure each required library can be imported."""
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(f"Failed to import '{module_name}' (package: {package_name}): {exc}")


def test_all_libraries_present() -> None:
    """Collect-style check that every required module is available."""
    missing = []
    for module_name, package_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(f"{package_name} (import as {module_name})")

    assert not missing, "Missing libraries:\n  - " + "\n  - ".join(missing)
