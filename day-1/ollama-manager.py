
#!/usr/bin/env python3
"""
Ollama Model Manager - A GUI application for managing local and remote Ollama models.

This application provides a comprehensive interface for:
- Viewing and managing local Ollama models
- Connecting to and managing remote Ollama models
- Running prompts on models with streaming support
- Saving models to archive files
- Transferring models between instances
"""

# Standard library imports
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# GUI imports
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Third-party imports (with graceful fallbacks)
try:
    import humanize
    HUMANIZE_AVAILABLE = True
except ImportError:
    HUMANIZE_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class ModelName:
    """Represents parsed model name components."""
    host: str = "registry.ollama.ai"
    namespace: str = "library"
    model: str = ""
    tag: str = "latest"


class OllamaModelManagerError(Exception):
    """Custom exception for Ollama Model Manager errors."""


class DependencyError(OllamaModelManagerError):
    """Raised when required dependencies are missing."""


class OllamaConnectionError(OllamaModelManagerError):
    """Raised when connection to Ollama server fails."""


class ModelParser:
    """Utility class for parsing model names and manifests."""

    @staticmethod
    def parse_model_name(name: str) -> ModelName:
        """
        Parse an Ollama model name into its components.

        Args:
            name: The model name string (e.g., "llama2:latest" or "hf.co/Qwen/Qwen3-Embedding-4B-GGUF")

        Returns:
            ModelName: A ModelName object with the parsed components

        Raises:
            ValueError: If the name format is invalid
        """
        patterns = [
            # Pattern for full qualified names like "hf.co/Qwen/Qwen3-Embedding-4B-GGUF:latest"
            r"^(?P<host>[^/]+\.[^/]+)/(?P<namespace>[^/]+)/(?P<model>[^:]+):(?P<tag>.+)$",
            # Pattern for full qualified names without tag like "hf.co/Qwen/Qwen3-Embedding-4B-GGUF"
            r"^(?P<host>[^/]+\.[^/]+)/(?P<namespace>[^/]+)/(?P<model>[^:]+)$",
            # Pattern for namespace/model:tag (e.g., "library/llama2:latest")
            r"^(?P<namespace>[^/]+)/(?P<model>[^:]+):(?P<tag>.+)$",
            # Pattern for namespace/model (e.g., "library/llama2")
            r"^(?P<namespace>[^/]+)/(?P<model>[^:]+)$",
            # Pattern for model:tag (e.g., "llama2:latest")
            r"^(?P<model>[^:]+):(?P<tag>.+)$",
            # Pattern for model only (e.g., "llama2")
            r"^(?P<model>[^:]+)$",
        ]

        for pattern in patterns:
            match = re.match(pattern, name)
            if match:
                groups = match.groupdict()

                # Determine the host
                host = groups.get("host", "registry.ollama.ai")

                # For models with explicit host like hf.co, namespace is the org
                # For regular models, namespace defaults to library
                if host != "registry.ollama.ai":
                    namespace = groups.get("namespace", "library")
                else:
                    namespace = groups.get("namespace", "library")

                return ModelName(
                    host=host,
                    namespace=namespace,
                    model=groups["model"],
                    tag=groups.get("tag", "latest"),
                )

        raise ValueError(f"Invalid name format: {name}")

    @staticmethod
    def parse_manifest(path: Path) -> List[str]:
        """
        Parse an Ollama manifest file to extract blob SHAs.

        Args:
            path: Path to the manifest file

        Returns:
            List of blob SHA identifiers

        Raises:
            FileNotFoundError: If the manifest file doesn't exist
            json.JSONDecodeError: If the manifest file is not valid JSON
        """
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")

        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            raise json.JSONDecodeError(
                f"Invalid JSON in manifest: {exc}", exc.doc, exc.pos
            ) from exc

        shas = [data["config"]["digest"]]
        shas.extend(layer["digest"] for layer in data["layers"])
        return ["sha256-" + s[7:] for s in shas]

    @staticmethod
    def get_file_paths(model_name: ModelName, model_path: Path) -> List[Path]:
        """
        Get the file paths for a model's blobs and manifest.

        Args:
            model_name: The parsed model name
            model_path: The base path to Ollama models

        Returns:
            List of relative paths to the model's files
        """
        manifest_abs_path = (
                model_path
                / "manifests"
                / model_name.host
                / model_name.namespace
                / model_name.model
                / model_name.tag
        )
        blob_shas = ModelParser.parse_manifest(manifest_abs_path)
        # Generate relative paths
        blob_paths = []
        blob_paths.append(
            Path("manifests")
            / model_name.host
            / model_name.namespace
            / model_name.model
            / model_name.tag
        )
        for sha in blob_shas:
            blob_paths.append(Path("blobs") / sha)
        return blob_paths

    @staticmethod
    def infer_model_name_from_full_name(full_name: str, model_path: Path) -> ModelName:
        """
        Infer the correct model name components by checking what actually exists on disk.
        This is a fallback when standard parsing fails.

        Args:
            full_name: The full model name as reported by Ollama
            model_path: The base path to Ollama models

        Returns:
            ModelName: A ModelName object with the correct components

        Raises:
            FileNotFoundError: If no matching manifest is found
        """
        # First try standard parsing
        try:
            parsed = ModelParser.parse_model_name(full_name)
            manifest_path = (
                    model_path / "manifests" / parsed.host /
                    parsed.namespace / parsed.model / parsed.tag
            )
            if manifest_path.exists():
                return parsed
        except ValueError:
            pass

        # If that fails, search for the manifest file
        manifests_dir = model_path / "manifests"
        if not manifests_dir.exists():
            raise FileNotFoundError(f"Manifests directory not found: {manifests_dir}")

        # Extract name and tag
        if ":" in full_name:
            name_part, tag = full_name.rsplit(":", 1)
        else:
            name_part = full_name
            tag = "latest"

        # Search through all possible host/namespace combinations
        for host_dir in manifests_dir.iterdir():
            if not host_dir.is_dir():
                continue

            for namespace_dir in host_dir.iterdir():
                if not namespace_dir.is_dir():
                    continue

                # Check if this could be our model
                potential_model_dir = namespace_dir / name_part.split("/")[-1]
                if potential_model_dir.is_dir():
                    manifest_file = potential_model_dir / tag
                    if manifest_file.exists():
                        return ModelName(
                            host=host_dir.name,
                            namespace=namespace_dir.name,
                            model=name_part.split("/")[-1],
                            tag=tag
                        )

        raise FileNotFoundError(f"No manifest found for model: {full_name}")


class SizeFormatter:
    """Utility class for formatting file sizes."""

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Format bytes to human-readable size."""
        if size_bytes == 0:
            return "0 B"

        if HUMANIZE_AVAILABLE:
            return humanize.naturalsize(size_bytes, binary=True)

        # Fallback formatting
        size_names = ("B", "KB", "MB", "GB", "TB")
        size_value = float(size_bytes)
        unit_index = 0

        while size_value >= 1024 and unit_index < len(size_names) - 1:
            size_value /= 1024
            unit_index += 1

        return f"{size_value:.1f} {size_names[unit_index]}"

    @staticmethod
    def parse_size(size_str: str) -> int:
        """Convert readable size back to bytes for sorting."""
        try:
            parts = size_str.strip().split()
            if len(parts) != 2:
                return 0

            number = float(parts[0])
            unit = parts[1]

            multipliers = {
                "B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3,
                "TB": 1024**4, "KiB": 1024, "MiB": 1024**2,
                "GiB": 1024**3, "TiB": 1024**4
            }

            return int(number * multipliers.get(unit, 0))
        except (ValueError, IndexError):
            return 0


class OllamaAPIClient:
    """Wrapper for Ollama API operations with error handling."""

    def __init__(self, host: str = "localhost", port: int = 11434):
        """Initialize the API client."""
        if not OLLAMA_AVAILABLE:
            raise DependencyError("Ollama package is not available")

        self.host = host
        self.port = port
        ollama_url = f"http://{host}:{port}"
        self.client = ollama.Client(host=ollama_url)

    def test_connection(self) -> bool:
        """Test if the connection to Ollama server is working."""
        try:
            self.client.list()
            return True
        except Exception:  # pylint: disable=broad-exception-caught
            return False

    def get_version(self) -> str:
        """Get Ollama version from API."""
        if not REQUESTS_AVAILABLE:
            return "Unknown"

        try:
            response = requests.get(
                f"http://{self.host}:{self.port}/api/version",
                timeout=10
            )
            if response.status_code == 200:
                version_data = response.json()
                return version_data.get("version", "Unknown")
            return "Unknown"
        except (requests.RequestException, json.JSONDecodeError):
            return "Unknown"

    def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        models = self.client.list()

        if hasattr(models, 'models') and isinstance(models.models, list):
            return models.models
        if isinstance(models, dict) and 'models' in models:
            return models.get('models', [])
        return models if isinstance(models, list) else []

    def get_running_models(self) -> List[Dict[str, Any]]:
        """Get list of currently running models."""
        if not REQUESTS_AVAILABLE:
            return []

        try:
            api_url = f"http://{self.host}:{self.port}/api/ps"
            response = requests.get(api_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
            return []
        except (requests.RequestException, json.JSONDecodeError):
            return []

    def delete_model(self, model_name: str) -> bool:
        """Delete a model."""
        if not REQUESTS_AVAILABLE:
            return False

        try:
            api_url = f"http://{self.host}:{self.port}/api/delete"
            response = requests.delete(
                api_url,
                json={"name": model_name},
                timeout=30
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate_response(self, model: str, prompt: str, system: str = "",
                          options: Optional[Dict[str, Any]] = None,
                          stream: bool = False, think: bool = False):
        """Generate a response from the model."""
        generate_kwargs = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": stream,
        }

        if options:
            generate_kwargs["options"] = options

        if think:
            generate_kwargs["think"] = think

        return self.client.generate(**generate_kwargs)

    def show_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model."""
        try:
            return self.client.show(model=model_name)
        except Exception:  # pylint: disable=broad-exception-caught
            return None


class ApplicationConfig:  # pylint: disable=too-few-public-methods
    """Configuration constants for the application."""

    # UI Constants
    DEFAULT_WINDOW_SIZE = "1200x700"
    DEFAULT_PROMPT = "Hello"
    DEFAULT_SYSTEM_PROMPT = "You are an intelligent assistant"
    DEFAULT_TEMPERATURE = 0.7
    OLLAMA_PORT = 11434
    USE_MARKDOWN_DEFAULT = True
    RUNNING_MODELS_REFRESH_INTERVAL = 10000  # 10 seconds
    TEMP_DIR = os.path.expanduser("~/ollama_temp")
    REMOTE_TRANSFER_PATH = "/Volumes/pierredavies/.ollama/models"

    # Models that support the thinking API
    THINKING_MODELS = ["deepseek-r1", "qwen", "qwen3"]

    # Text styling
    TEXT_TAGS = {
        "bold": {"font": "TkDefaultFont 10 bold"},
        "italic": {"font": "TkDefaultFont 10 italic"},
        "heading1": {"font": "TkDefaultFont 14 bold"},
        "heading2": {"font": "TkDefaultFont 12 bold"},
        "heading3": {"font": "TkDefaultFont 11 bold"},
        "code": {"background": "#f0f0f0", "font": "Courier 10"},
        "code_block": {"background": "#f8f8f8", "font": "Courier 10"}
    }

    # Tree view configuration
    TREE_COLUMNS = ("name", "tag", "size", "created", "modified",
                    "quantization", "type")
    COLUMN_WIDTHS = {
        "name": 120, "tag": 80, "size": 80, "created": 120,
        "modified": 120, "quantization": 80, "type": 80
    }


class ModelInfoExtractor:  # pylint: disable=too-few-public-methods
    """Utility class for extracting model information."""

    @staticmethod
    def extract_model_info(model: Dict[str, Any],
                           host: str = "localhost") -> Dict[str, Any]:
        """Extract structured information from an Ollama model object."""
        full_name = model.get('model', '') or model.get('name', '')
        name_parts = full_name.split(':')
        name = name_parts[0] if name_parts else full_name
        tag = name_parts[1] if len(name_parts) > 1 else 'latest'

        details = model.get('details', {})

        return {
            "name": name,
            "tag": tag,
            "size": model.get('size', 0),
            "created": ModelInfoExtractor._get_creation_date(name, tag, host),
            "modified": ModelInfoExtractor._get_modified_date(model),
            "quantization": ModelInfoExtractor._extract_quantization(
                details, full_name
            ),
            "model_type": ModelInfoExtractor._extract_model_type(
                details, full_name
            ),
            "full_name": full_name
        }

    @staticmethod
    def _extract_quantization(details: Dict[str, Any], full_name: str) -> str:
        """Extract quantization information."""
        if details and hasattr(details, 'quantization_level'):
            return details.quantization_level

        full_name_lower = full_name.lower()
        quantization_map = {
            'q4': 'Q4', 'q5': 'Q5', 'q8': 'Q8', 'fp16': 'FP16'
        }

        for key, value in quantization_map.items():
            if key in full_name_lower:
                return value

        return 'unknown'

    @staticmethod
    def _extract_model_type(details: Dict[str, Any], full_name: str) -> str:
        """Extract model type information."""
        if details and hasattr(details, 'family'):
            return details.family.capitalize()

        full_name_lower = full_name.lower()
        type_map = {
            'llama': 'LLaMA', 'mistral': 'Mistral', 'vicuna': 'Vicuna',
            'gemma': 'Gemma', 'phi': 'Phi', 'qwen': 'Qwen'
        }

        for key, value in type_map.items():
            if key in full_name_lower:
                return value

        return 'unknown'

    @staticmethod
    def _get_modified_date(model: Dict[str, Any]) -> str:
        """Extract modification date."""
        if hasattr(model, 'modified_at') and model.modified_at:
            try:
                return model.modified_at.strftime('%Y-%m-%d %H:%M')
            except (AttributeError, ValueError):
                pass

        if model.get('modified', 0) > 0:
            try:
                timestamp = model.get('modified', 0)
                return datetime.fromtimestamp(timestamp).strftime(
                    '%Y-%m-%d %H:%M'
                )
            except (OSError, ValueError):
                pass

        return "Unknown"

    @staticmethod
    def _get_creation_date(model_name: str, tag: str, host: str) -> str:
        """Get creation date (only for localhost)."""
        if host != "localhost":
            return "Unknown"

        try:
            model_path = Path(
                os.getenv("OLLAMA_MODELS", "~/.ollama/models")
            ).expanduser()

            # Simple creation time detection
            manifest_dir = (
                model_path / "manifests" / "registry.ollama.ai" /
                "library" / model_name
            )
            if '/' in model_name:
                org, base_name = model_name.split('/', 1)
                manifest_dir = (
                    model_path / "manifests" / "registry.ollama.ai" /
                    org / base_name
                )

            manifest_file = manifest_dir / tag

            if manifest_file.exists():
                stat_info = os.stat(manifest_file)
                if hasattr(stat_info, 'st_birthtime'):
                    creation_time = datetime.fromtimestamp(
                        stat_info.st_birthtime
                    )
                    return creation_time.strftime('%Y-%m-%d %H:%M')

            return "Unknown"
        except (OSError, ValueError):
            return "Unknown"


class OllamaModelManager:  # pylint: disable=too-many-instance-attributes
    """Main application class for the Ollama Model Manager."""

    def __init__(self, root: tk.Tk):
        """Initialize the application."""
        self._check_dependencies()

        self.root = root
        self.root.title("Ollama Model Manager")
        self.root.geometry(ApplicationConfig.DEFAULT_WINDOW_SIZE)

        # Initialize state
        self.host = "localhost"
        self.api_client: Optional[OllamaAPIClient] = None
        self.ollama_version = "Unknown"
        self.models_data: List[Dict[str, Any]] = []
        self.running_models: List[Dict[str, Any]] = []

        # UI components (will be initialized in _init_ui)
        self.host_entry: Optional[ttk.Entry] = None
        self.connect_button: Optional[ttk.Button] = None
        self.running_models_text: Optional[tk.Text] = None
        self.models_tree: Optional[ttk.Treeview] = None
        self.search_var: Optional[tk.StringVar] = None
        self.status_var: Optional[tk.StringVar] = None
        self.models_frame: Optional[ttk.LabelFrame] = None
        self.models_frame_text = "Ollama Models"

        # Menu components
        self.context_menu: Optional[tk.Menu] = None
        self.middle_button_menu: Optional[tk.Menu] = None

        # Initialize UI and connect
        self._init_ui()
        self._connect_ollama()

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        missing_deps = []

        if not OLLAMA_AVAILABLE:
            missing_deps.append("ollama")
        if not REQUESTS_AVAILABLE:
            missing_deps.append("requests")

        if missing_deps:
            deps_str = ", ".join(missing_deps)
            error_msg = (
                f"Missing required dependencies: {deps_str}\n"
                f"Please install with: pip install {' '.join(missing_deps)}"
            )
            messagebox.showerror("Missing Dependencies", error_msg)
            self.root.destroy()
            raise DependencyError(error_msg)

    def _init_ui(self):
        """Initialize the user interface components."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self._create_connection_section(main_frame)
        self._create_models_section(main_frame)
        self._create_status_bar(main_frame)

    def _create_connection_section(self, parent: ttk.Frame):
        """Create connection settings and running models sections."""
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, padx=5, pady=5)

        # Connection settings
        conn_frame = ttk.LabelFrame(
            top_frame, text="Connection Settings", padding="10"
        )
        conn_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        ttk.Label(conn_frame, text="Ollama Host:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.host_entry = ttk.Entry(conn_frame, width=30)
        self.host_entry.insert(0, "localhost")
        self.host_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        self.connect_button = ttk.Button(
            conn_frame, text="Connect", command=self._connect_ollama
        )
        self.connect_button.grid(row=0, column=2, padx=5, pady=2)

        # Running models display
        self._create_running_models_section(top_frame)

    def _create_running_models_section(self, parent: ttk.Frame):
        """Create the running models display section."""
        running_frame = ttk.LabelFrame(
            parent, text="Running Models", padding="10"
        )
        running_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        self.running_models_text = tk.Text(
            running_frame, height=9, width=75,
            font=("Courier", 10), wrap=tk.NONE
        )
        self.running_models_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(
            running_frame, orient="vertical",
            command=self.running_models_text.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.running_models_text.configure(yscrollcommand=scrollbar.set)

        self.running_models_text.insert(tk.END, "No running models")
        self.running_models_text.config(state=tk.DISABLED)

    def _create_models_section(self, parent: ttk.Frame):
        """Create the models display section."""
        models_frame = ttk.Frame(parent)
        models_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.models_frame = ttk.LabelFrame(
            models_frame, text=self.models_frame_text, padding="10"
        )
        self.models_frame.pack(fill=tk.BOTH, expand=True, padx=5)

        self._create_search_section()
        self._create_models_tree()
        self._create_context_menus()

    def _create_search_section(self):
        """Create the search functionality."""
        search_frame = ttk.Frame(self.models_frame)
        search_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._on_search_changed)
        search_entry = ttk.Entry(
            search_frame, textvariable=self.search_var, width=30
        )
        search_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    def _create_models_tree(self):
        """Create the models treeview."""
        self.models_tree = ttk.Treeview(
            self.models_frame, columns=ApplicationConfig.TREE_COLUMNS
        )
        self.models_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configure columns
        self.models_tree.heading("#0", text="", anchor=tk.W)
        self.models_tree.column("#0", width=0, stretch=tk.NO)

        for col in ApplicationConfig.TREE_COLUMNS:
            self.models_tree.heading(
                col, text=col.capitalize(), anchor=tk.W,
                command=lambda _col=col: self._sort_column(_col, False)
            )
            self.models_tree.column(
                col,
                width=ApplicationConfig.COLUMN_WIDTHS[col],
                anchor=tk.W
            )

        # Bind events
        self.models_tree.bind("<Double-1>", self._on_model_double_click)
        self.models_tree.bind("<Button-3>", self._show_context_menu)

        # Bind middle mouse button for different platforms
        self.models_tree.bind("<Button-2>", self._show_middle_button_menu)  # Unix/Linux
        if sys.platform == 'darwin':  # macOS
            self.models_tree.bind("<ButtonRelease-2>", self._show_middle_button_menu)
        elif sys.platform == 'win32':  # Windows
            self.models_tree.bind("<Control-Button-3>", self._show_middle_button_menu)

    def _create_context_menus(self):
        """Create the context and middle-button menus."""
        # Right-click context menu
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(
            label="Run Prompt", command=self._show_prompt_dialog
        )
        self.context_menu.add_command(
            label="Model Details", command=self._show_model_details
        )
        self.context_menu.add_command(
            label="Delete Model", command=self._delete_model
        )
        self.context_menu.add_command(
            label="Save Model", command=self._save_model
        )
        self.context_menu.add_command(
            label="Transfer Model", command=self._transfer_model
        )

        # Middle-click menu
        self.middle_button_menu = tk.Menu(self.root, tearoff=0)
        self.middle_button_menu.add_command(
            label="Run Prompt", command=self._show_prompt_dialog
        )
        self.middle_button_menu.add_command(
            label="Model Details", command=self._show_model_details
        )
        self.middle_button_menu.add_command(
            label="Delete Model", command=self._delete_model
        )
        self.middle_button_menu.add_command(
            label="Save Model", command=self._save_model
        )
        self.middle_button_menu.add_command(
            label="Transfer Model", command=self._transfer_model
        )

    def _create_status_bar(self, parent: ttk.Frame):
        """Create the status bar."""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            parent, textvariable=self.status_var,
            relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)

    def _on_search_changed(self, *_args):
        """Handle search text changes."""
        self._filter_models()

    def _on_model_double_click(self, _event):
        """Handle double-click on model."""
        self._show_prompt_dialog()

    def _show_context_menu(self, event):
        """Show context menu for model operations."""
        item = self.models_tree.identify_row(event.y)
        if not item:
            return

        self.models_tree.selection_set(item)
        self._update_menu_states(self.context_menu)
        self.context_menu.post(event.x_root, event.y_root)

    def _show_middle_button_menu(self, event):
        """Show middle-button menu for model operations."""
        item = self.models_tree.identify_row(event.y)
        if not item:
            return

        self.models_tree.selection_set(item)
        self._update_menu_states(self.middle_button_menu)
        self.middle_button_menu.post(event.x_root, event.y_root)

    def _update_menu_states(self, menu: tk.Menu):
        """Update menu item states based on connection and host."""
        if not self.api_client:
            # Disable all items if not connected
            for i in range(menu.index("end") + 1):
                menu.entryconfigure(i, state=tk.DISABLED)
        else:
            # Enable all items if connected
            for i in range(menu.index("end") + 1):
                menu.entryconfigure(i, state=tk.NORMAL)

            # Disable localhost-only features for remote connections
            if self.host != "localhost":
                try:
                    menu.entryconfigure("Save Model", state=tk.DISABLED)
                    menu.entryconfigure("Transfer Model", state=tk.DISABLED)
                    menu.entryconfigure("Delete Model", state=tk.DISABLED)
                except tk.TclError:
                    pass  # Menu item might not exist

    def _sort_column(self, col: str, reverse: bool):
        """Sort treeview content when a column header is clicked."""
        items = self.models_tree.get_children('')

        # Determine column type and sort accordingly
        if col == "size":
            # Convert size to bytes for correct sorting
            data = [(SizeFormatter.parse_size(self.models_tree.set(k, col)), k)
                   for k in items]
        elif col in ["created", "modified"]:
            # For date columns, handle "Unknown" values
            data = []
            for k in items:
                date_str = self.models_tree.set(k, col)
                # Put "Unknown" dates at the end when sorting ascending
                if date_str == "Unknown":
                    sort_key = "9999-99-99" if not reverse else "0000-00-00"
                else:
                    sort_key = date_str
                data.append((sort_key, k))
        else:
            # Default string sorting
            data = [(self.models_tree.set(k, col), k) for k in items]

        # Sort the data
        data.sort(reverse=reverse)

        # Rearrange items in sorted positions
        for index, (_, k) in enumerate(data):
            self.models_tree.move(k, '', index)

        # Configure the heading for next click
        self.models_tree.heading(
            col, command=lambda: self._sort_column(col, not reverse)
        )

    def _connect_ollama(self):
        """Connect to the Ollama server."""
        host_input = self.host_entry.get() or "localhost"
        self.host = host_input

        # Parse host and port
        if ":" in host_input:
            host_part, port_str = host_input.split(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                port = ApplicationConfig.OLLAMA_PORT
        else:
            host_part = host_input
            port = ApplicationConfig.OLLAMA_PORT

        self.status_var.set(f"Connecting to Ollama server at {self.host}...")

        try:
            self.api_client = OllamaAPIClient(host_part, port)
            threading.Thread(target=self._test_connection, daemon=True).start()
        except DependencyError as exc:
            self._handle_connection_error(str(exc))

    def _test_connection(self):
        """Test the connection in a background thread."""
        try:
            if self.api_client.test_connection():
                self.ollama_version = self.api_client.get_version()
                self.root.after(0, self._update_ui_after_connect)
            else:
                self.root.after(
                    0, lambda: self._handle_connection_error(
                        "Connection test failed"
                    )
                )
        except Exception as error:  # pylint: disable=broad-exception-caught
            self.root.after(
                0, lambda: self._handle_connection_error(str(error))
            )

    def _update_ui_after_connect(self):
        """Update UI after successful connection."""
        host_display = self.host if self.host != "localhost" else "Local"
        self.models_frame_text = (
            f"Ollama Models - {host_display} (v{self.ollama_version})"
        )
        self.models_frame.configure(text=self.models_frame_text)

        self.status_var.set(f"Connected to {host_display} Ollama server")
        self.connect_button.config(text="Reconnect")

        self._refresh_models()
        self._start_running_models_refresh()

    def _handle_connection_error(self, error_message: str):
        """Handle connection errors."""
        messagebox.showerror(
            "Connection Error",
            f"Failed to connect to Ollama server: {error_message}"
        )
        self.status_var.set("Connection failed")

    def _start_running_models_refresh(self):
        """Start periodic refresh of running models."""
        self._fetch_running_models()
        self.root.after(
            ApplicationConfig.RUNNING_MODELS_REFRESH_INTERVAL,
            self._start_running_models_refresh
        )

    def _fetch_running_models(self):
        """Fetch currently running models."""
        if not self.api_client:
            return

        def fetch_thread():
            try:
                models = self.api_client.get_running_models()
                self.root.after(0, lambda: self._update_running_models_ui(models))
            except Exception:  # pylint: disable=broad-exception-caught
                pass  # Silently handle errors for background updates

        threading.Thread(target=fetch_thread, daemon=True).start()

    def _update_running_models_ui(self, models: List[Dict[str, Any]]):
        """Update the running models display."""
        self.running_models = models

        self.running_models_text.config(state=tk.NORMAL)
        self.running_models_text.delete("1.0", tk.END)

        if not models:
            self.running_models_text.insert(tk.END, "No running models")
        else:
            header = f"{'NAME':<18} {'SIZE':<10} {'UNTIL':<20}\n"
            separator = "-" * 50 + "\n"
            self.running_models_text.insert(tk.END, header + separator)

            for model in models:
                line = self._format_running_model(model)
                self.running_models_text.insert(tk.END, line)

        self.running_models_text.config(state=tk.DISABLED)

    def _format_running_model(self, model: Dict[str, Any]) -> str:
        """Format a running model for display."""
        name = model.get('model', 'unknown')
        if len(name) > 17:
            name = name[:14] + "..."

        size_bytes = model.get('size', 0)
        size_display = SizeFormatter.format_size(size_bytes)

        expires_at = model.get('expires_at', '')
        until_display = "N/A"
        if expires_at:
            try:
                expires_time = datetime.fromisoformat(
                    expires_at.replace('Z', '+00:00')
                )
                now = datetime.now().astimezone()
                diff = expires_time - now

                if diff.total_seconds() > 0:
                    minutes = int(diff.total_seconds() // 60)
                    if minutes < 60:
                        until_display = f"{minutes}m"
                    else:
                        hours = minutes // 60
                        until_display = f"{hours}h"
                else:
                    until_display = "Expired"
            except (ValueError, AttributeError):
                until_display = "N/A"

        return f"{name:<18} {size_display:<10} {until_display:<20}\n"

    def _refresh_models(self):
        """Refresh the models list."""
        if not self.api_client:
            return

        def refresh_thread():
            try:
                models = self.api_client.list_models()
                processed_models = []

                for model in models:
                    model_info = ModelInfoExtractor.extract_model_info(
                        model, self.host
                    )
                    processed_models.append(model_info)

                self.root.after(
                    0, lambda: self._update_models_ui(processed_models)
                )
            except Exception as error:
                self.root.after(
                    0, lambda: self._handle_connection_error(str(error))
                )

        threading.Thread(target=refresh_thread, daemon=True).start()

    def _update_models_ui(self, models: List[Dict[str, Any]]):
        """Update the models display."""
        # Clear existing items
        for item in self.models_tree.get_children():
            self.models_tree.delete(item)

        self.models_data = models

        # Insert models
        for model in models:
            self.models_tree.insert("", tk.END, values=(
                model["name"],
                model["tag"],
                SizeFormatter.format_size(model["size"]),
                model["created"],
                model["modified"],
                model["quantization"],
                model["model_type"]
            ))

        host_display = self.host if self.host != "localhost" else "local"
        self.status_var.set(f"Loaded {len(models)} models from {host_display}")

    def _filter_models(self):
        """Filter models based on search term."""
        search_term = self.search_var.get().lower()

        # Clear existing items
        for item in self.models_tree.get_children():
            self.models_tree.delete(item)

        # Filter and insert matching models
        for model in self.models_data:
            if self._model_matches_search(model, search_term):
                self.models_tree.insert("", tk.END, values=(
                    model["name"],
                    model["tag"],
                    SizeFormatter.format_size(model["size"]),
                    model["created"],
                    model["modified"],
                    model["quantization"],
                    model["model_type"]
                ))

    def _model_matches_search(self, model: Dict[str, Any],
                              search_term: str) -> bool:
        """Check if a model matches the search term."""
        searchable_fields = ["name", "tag", "quantization", "model_type"]
        return any(
            search_term in str(model[field]).lower()
            for field in searchable_fields
        )

    def _show_prompt_dialog(self):
        """Show dialog for running prompts."""
        selected_item = self.models_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "No model selected")
            return

        values = self.models_tree.item(selected_item, 'values')
        model_name = values[0]
        model_tag = values[1]
        model_full = f"{model_name}:{model_tag}"

        # Create a simplified prompt dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Run Prompt - {model_full}")
        dialog.geometry("600x400")
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Prompt entry
        ttk.Label(frame, text="Enter prompt:").pack(anchor=tk.W, pady=(0, 5))
        prompt_var = tk.StringVar(value=ApplicationConfig.DEFAULT_PROMPT)
        prompt_entry = ttk.Entry(frame, textvariable=prompt_var)
        prompt_entry.pack(fill=tk.X, pady=(0, 10))
        prompt_entry.focus_set()

        # Response area
        ttk.Label(frame, text="Response:").pack(anchor=tk.W, pady=(0, 5))
        response_text = tk.Text(frame, wrap=tk.WORD, height=15)
        response_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Status
        status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(frame, textvariable=status_var)
        status_label.pack(fill=tk.X, pady=(0, 10))

        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X)

        def run_prompt():
            prompt = prompt_var.get().strip()
            if not prompt:
                status_var.set("Please enter a prompt")
                return

            status_var.set("Generating response...")
            response_text.delete("1.0", tk.END)

            def generate_thread():
                try:
                    response = self.api_client.generate_response(
                        model=model_full,
                        prompt=prompt,
                        system=ApplicationConfig.DEFAULT_SYSTEM_PROMPT
                    )

                    response_content = response.get('response', 'No response')
                    dialog.after(
                        0, lambda: self._update_prompt_response(
                            response_text, status_var, response_content
                        )
                    )
                except Exception as error:  # pylint: disable=broad-exception-caught
                    dialog.after(
                        0, lambda: self._handle_prompt_error(
                            status_var, str(error)
                        )
                    )

            threading.Thread(target=generate_thread, daemon=True).start()

        ttk.Button(button_frame, text="Run", command=run_prompt).pack(
            side=tk.RIGHT, padx=(5, 0)
        )
        ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(
            side=tk.RIGHT
        )

        # Bind Enter key
        prompt_entry.bind("<Return>", lambda _: run_prompt())

    def _update_prompt_response(self, text_widget: tk.Text,
                                status_var: tk.StringVar, response: str):
        """Update the prompt response display."""
        text_widget.insert(tk.END, response)
        status_var.set("Response completed")

    def _handle_prompt_error(self, status_var: tk.StringVar, error: str):
        """Handle prompt generation errors."""
        status_var.set(f"Error: {error}")

    def _show_model_details(self):
        """Show detailed model information."""
        selected_item = self.models_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "No model selected")
            return

        values = self.models_tree.item(selected_item, 'values')
        model_name, model_tag = values[0], values[1]
        model_full = f"{model_name}:{model_tag}"

        # Create details dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Model Details - {model_full}")
        dialog.geometry("500x400")
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        # Model information
        info_text = tk.Text(frame, wrap=tk.WORD, height=20)
        info_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        details = (
            f"Model: {model_full}\n"
            f"Name: {values[0]}\n"
            f"Tag: {values[1]}\n"
            f"Size: {values[2]}\n"
            f"Created: {values[3]}\n"
            f"Modified: {values[4]}\n"
            f"Quantization: {values[5]}\n"
            f"Type: {values[6]}\n"
        )

        info_text.insert(tk.END, details)

        # Try to get additional details
        if self.api_client:
            try:
                model_info = self.api_client.show_model(model_full)
                if model_info:
                    info_text.insert(tk.END, "\n--- Additional Details ---\n")

                    if hasattr(model_info, 'template') and model_info.template:
                        info_text.insert(tk.END, f"Template: {model_info.template}\n")

                    if hasattr(model_info, 'parameters') and model_info.parameters:
                        info_text.insert(tk.END, "\nParameters:\n")
                        if hasattr(model_info.parameters, 'items'):
                            for key, value in model_info.parameters.items():
                                info_text.insert(tk.END, f"  {key}: {value}\n")
            except Exception:  # pylint: disable=broad-exception-caught
                pass  # Silently handle errors

        info_text.config(state=tk.DISABLED)

        # Close button
        ttk.Button(frame, text="Close", command=dialog.destroy).pack()

    def _delete_model(self):
        """Delete the selected model."""
        selected_item = self.models_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "No model selected")
            return

        values = self.models_tree.item(selected_item, 'values')
        model_name, model_tag = values[0], values[1]
        model_full = f"{model_name}:{model_tag}"

        # Confirm deletion
        result = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete model '{model_full}'?\n"
            "This action cannot be undone.",
            icon='warning'
        )

        if result:
            self.status_var.set(f"Deleting model {model_full}...")

            def delete_thread():
                try:
                    success = self.api_client.delete_model(model_full)
                    if success:
                        self.root.after(0, self._on_model_deleted)
                    else:
                        self.root.after(
                            0, lambda: self._handle_delete_error(
                                "Failed to delete model"
                            )
                        )
                except Exception as error:  # pylint: disable=broad-exception-caught
                    self.root.after(
                        0, lambda: self._handle_delete_error(str(error))
                    )

            threading.Thread(target=delete_thread, daemon=True).start()

    def _on_model_deleted(self):
        """Handle successful model deletion."""
        self.status_var.set("Model deleted successfully")
        self._refresh_models()

    def _handle_delete_error(self, error: str):
        """Handle model deletion errors."""
        messagebox.showerror("Deletion Error", f"Failed to delete model: {error}")
        self.status_var.set("Model deletion failed")

    def _save_model(self):
        """Save the selected model to a tarball file."""
        selected_item = self.models_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "No model selected")
            return

        values = self.models_tree.item(selected_item, 'values')
        model_name, model_tag = values[0], values[1]
        model_full = f"{model_name}:{model_tag}"

        if self.host != "localhost":
            messagebox.showerror(
                "Error",
                "Model saving is only supported for local Ollama instances.\n"
                f"You are currently connected to {self.host}."
            )
            return

        # Get save location
        default_filename = f"{model_name.replace('/', '_')}-{model_tag}.tar"
        save_path = filedialog.asksaveasfilename(
            title=f"Save model {model_full}",
            defaultextension=".tar",
            filetypes=[("Tar files", "*.tar"), ("All files", "*.*")],
            initialfile=default_filename
        )

        if not save_path:
            return

        self.status_var.set(f"Saving model {model_full}...")

        def save_thread():
            try:
                # Get model path
                model_path = Path(
                    os.getenv("OLLAMA_MODELS", "~/.ollama/models")
                ).expanduser()

                # Try to parse model name, with fallback to inference
                try:
                    model_name_obj = ModelParser.parse_model_name(model_full)
                except ValueError:
                    # Use inference as fallback
                    model_name_obj = ModelParser.infer_model_name_from_full_name(
                        model_full, model_path
                    )

                # Get file paths
                blob_paths = ModelParser.get_file_paths(model_name_obj, model_path)

                # Create tarball
                tar_command = ["tar", "-cf", save_path, "-C", str(model_path)]
                tar_command.extend([str(path) for path in blob_paths])

                subprocess.run(
                    tar_command, capture_output=True, text=True, check=True
                )

                self.root.after(0, lambda: self.status_var.set(
                    f"Model {model_full} saved successfully"
                ))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Success", f"Model saved to {save_path}"
                ))
            except Exception as error:
                error_msg = str(error)
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to save model: {error_msg}"
                ))
                self.root.after(0, lambda: self.status_var.set("Model save failed"))

        threading.Thread(target=save_thread, daemon=True).start()

    def _transfer_model(self):
        """Transfer the selected model to a remote location."""
        selected_item = self.models_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "No model selected")
            return

        values = self.models_tree.item(selected_item, 'values')
        model_name, model_tag = values[0], values[1]
        model_full = f"{model_name}:{model_tag}"

        if self.host != "localhost":
            messagebox.showerror(
                "Error",
                "Model transfer is only supported for local Ollama instances.\n"
                f"You are currently connected to {self.host}."
            )
            return

        # Confirm transfer
        result = messagebox.askyesno(
            "Confirm Transfer",
            f"Transfer model '{model_full}' to:\n"
            f"{ApplicationConfig.REMOTE_TRANSFER_PATH}?"
        )

        if not result:
            return

        self.status_var.set(f"Transferring model {model_full}...")

        def transfer_thread():
            try:
                # Parse model name
                model_name_obj = ModelParser.parse_model_name(model_full)

                # Get model path
                model_path = Path(
                    os.getenv("OLLAMA_MODELS", "~/.ollama/models")
                ).expanduser()

                # Get file paths
                blob_paths = ModelParser.get_file_paths(model_name_obj, model_path)

                # Transfer each file
                dest_path = Path(ApplicationConfig.REMOTE_TRANSFER_PATH)

                for rel_path in blob_paths:
                    src_file = model_path / rel_path
                    dst_file = dest_path / rel_path

                    # Create parent directories
                    dst_file.parent.mkdir(parents=True, exist_ok=True)

                    # Copy file using rsync or cp
                    try:
                        subprocess.run(
                            ["rsync", "-a", str(src_file), str(dst_file)],
                            check=True, capture_output=True
                        )
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        subprocess.run(
                            ["cp", str(src_file), str(dst_file)],
                            check=True, capture_output=True
                        )

                self.root.after(0, lambda: self.status_var.set(
                    f"Model {model_full} transferred successfully"
                ))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Success", f"Model transferred to {ApplicationConfig.REMOTE_TRANSFER_PATH}"
                ))
            except Exception as error:  # pylint: disable=broad-exception-caught
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to transfer model: {str(error)}"
                ))
                self.root.after(0, lambda: self.status_var.set("Model transfer failed"))

        threading.Thread(target=transfer_thread, daemon=True).start()


def main():
    """Main entry point of the application."""
    try:
        root = tk.Tk()
        OllamaModelManager(root)  # Create the app instance
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"An unexpected error occurred: {exc}")
        if 'root' in locals():
            messagebox.showerror("Error", f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
