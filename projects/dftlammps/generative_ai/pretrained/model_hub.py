"""
Pretrained Model Hub
====================

Download and load pretrained generative models.

Supported models:
- CrystalDiT checkpoints
- Flow matching models
- Consistency models
- MatterGen-compatible models
"""

import torch
import os
from typing import Optional, Dict, Any
from pathlib import Path
import hashlib
import json
import urllib.request
from tqdm import tqdm


# Model registry with download URLs
MODEL_REGISTRY = {
    "crystal_dit_base": {
        "description": "Base CrystalDiT model trained on MP-20",
        "url": "https://example.com/models/crystal_dit_base.pt",
        "size": "150MB",
        "md5": "placeholder",
        "config": {
            "latent_dim": 512,
            "num_layers": 12,
            "num_heads": 8
        }
    },
    "crystal_dit_large": {
        "description": "Large CrystalDiT model (450M parameters)",
        "url": "https://example.com/models/crystal_dit_large.pt",
        "size": "600MB",
        "md5": "placeholder",
        "config": {
            "latent_dim": 768,
            "num_layers": 24,
            "num_heads": 12
        }
    },
    "flow_mm_mp20": {
        "description": "FlowMM model trained on MP-20",
        "url": "https://example.com/models/flow_mm_mp20.pt",
        "size": "200MB",
        "md5": "placeholder",
        "config": {
            "hidden_dim": 256,
            "num_layers": 6
        }
    },
    "consistency_fast": {
        "description": "Consistency model for fast generation (1-step)",
        "url": "https://example.com/models/consistency_fast.pt",
        "size": "150MB",
        "md5": "placeholder",
        "config": {
            "hidden_dim": 256,
            "num_layers": 6
        }
    },
    "conditional_bandgap": {
        "description": "Conditional model for band gap targeting",
        "url": "https://example.com/models/conditional_bandgap.pt",
        "size": "150MB",
        "md5": "placeholder",
        "config": {
            "latent_dim": 512,
            "num_layers": 12,
            "num_properties": 1
        }
    },
    "joint_mol_crystal": {
        "description": "Joint molecular-crystal generator",
        "url": "https://example.com/models/joint_mol_crystal.pt",
        "size": "180MB",
        "md5": "placeholder",
        "config": {
            "latent_dim": 512,
            "num_layers": 12
        }
    }
}


class PretrainedModelHub:
    """
    Hub for downloading and loading pretrained models.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: Directory to cache downloaded models
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/dftlammps/generative_ai")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry = MODEL_REGISTRY
    
    def list_models(self) -> Dict[str, Dict]:
        """List available models."""
        return {
            name: {
                "description": info["description"],
                "size": info["size"],
                "downloaded": self._is_downloaded(name)
            }
            for name, info in self.registry.items()
        }
    
    def download(self, model_name: str, force: bool = False) -> Path:
        """
        Download a model.
        
        Args:
            model_name: Name of the model
            force: Redownload even if already cached
            
        Returns:
            Path to downloaded model
        """
        if model_name not in self.registry:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.registry.keys())}")
        
        model_info = self.registry[model_name]
        cache_path = self.cache_dir / f"{model_name}.pt"
        
        if cache_path.exists() and not force:
            print(f"Model {model_name} already cached at {cache_path}")
            return cache_path
        
        # Download
        url = model_info["url"]
        print(f"Downloading {model_name} from {url}...")
        
        # Create progress bar
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)
        
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=model_name) as t:
            urllib.request.urlretrieve(url, cache_path, reporthook=t.update_to)
        
        # Verify checksum (if not placeholder)
        if model_info["md5"] != "placeholder":
            md5 = self._compute_md5(cache_path)
            if md5 != model_info["md5"]:
                raise ValueError(f"Checksum mismatch for {model_name}")
        
        # Save config
        config_path = self.cache_dir / f"{model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_info["config"], f, indent=2)
        
        print(f"Downloaded {model_name} to {cache_path}")
        return cache_path
    
    def load(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        strict: bool = True
    ) -> torch.nn.Module:
        """
        Load a pretrained model.
        
        Args:
            model_name: Name of the model
            device: Device to load on
            strict: Strict loading
            
        Returns:
            Loaded model
        """
        # Download if needed
        model_path = self.download(model_name)
        
        # Load config
        config_path = self.cache_dir / f"{model_name}_config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        # Create model
        model = self._create_model(model_name, config)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=strict)
        model.to(device)
        model.eval()
        
        print(f"Loaded {model_name} on {device}")
        return model
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model_class,
        config: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> torch.nn.Module:
        """
        Load a model from a custom checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model_class: Model class to instantiate
            config: Model configuration
            device: Device to load on
            
        Returns:
            Loaded model
        """
        model = model_class(config)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model
    
    def _is_downloaded(self, model_name: str) -> bool:
        """Check if model is downloaded."""
        cache_path = self.cache_dir / f"{model_name}.pt"
        return cache_path.exists()
    
    def _compute_md5(self, file_path: Path) -> str:
        """Compute MD5 checksum of file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _create_model(self, model_name: str, config: Dict) -> torch.nn.Module:
        """Create model instance from config."""
        from ..models import CrystalDiT, CrystalFlow, ConsistencyCrystalModel
        from ..models import ConditionalDiffusion, JointMolecularCrystalGenerator
        
        if "crystal_dit" in model_name:
            from ..models.crystal_dit import CrystalDiTConfig
            cfg = CrystalDiTConfig(**config)
            return CrystalDiT(cfg)
        
        elif "flow" in model_name:
            from ..models.flow_matching import FlowMatchingConfig
            cfg = FlowMatchingConfig(**config)
            return CrystalFlow(cfg)
        
        elif "consistency" in model_name:
            from ..models.consistency import ConsistencyConfig
            cfg = ConsistencyConfig(**config)
            return ConsistencyCrystalModel(cfg)
        
        elif "conditional" in model_name:
            from ..models.crystal_dit import CrystalDiTConfig
            from ..models.conditional import ConditionalConfig, ConditionalDiffusion
            from ..models.crystal_dit import CrystalDiT
            
            base_cfg = CrystalDiTConfig(**config)
            base_model = CrystalDiT(base_cfg)
            
            cond_cfg = ConditionalConfig(
                num_properties=config.get("num_properties", 5)
            )
            return ConditionalDiffusion(base_model, cond_cfg)
        
        elif "joint" in model_name:
            from ..models.joint_generator import JointGeneratorConfig
            cfg = JointGeneratorConfig(**config)
            return JointMolecularCrystalGenerator(cfg)
        
        else:
            raise ValueError(f"Unknown model type for {model_name}")


def download_model(model_name: str, cache_dir: Optional[str] = None) -> Path:
    """
    Convenience function to download a model.
    
    Args:
        model_name: Name of the model
        cache_dir: Optional cache directory
        
    Returns:
        Path to downloaded model
    """
    hub = PretrainedModelHub(cache_dir)
    return hub.download(model_name)


def load_model(
    model_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    cache_dir: Optional[str] = None
) -> torch.nn.Module:
    """
    Convenience function to load a model.
    
    Args:
        model_name: Name of the model
        device: Device to load on
        cache_dir: Optional cache directory
        
    Returns:
        Loaded model
    """
    hub = PretrainedModelHub(cache_dir)
    return hub.load(model_name, device)
