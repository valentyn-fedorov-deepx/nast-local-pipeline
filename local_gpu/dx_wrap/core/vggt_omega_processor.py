import cv2
import torch
from pathlib import Path
import numpy as np

import sys

from core.vggt_processor import VGGTProcessor


ROOT_PATH = Path(__file__).parent.parent
VGGT_OMEGA_PATH = ROOT_PATH / "thirdparty" / "vggt-omega"


def _ensure_vggt_omega_imports():
    if str(VGGT_OMEGA_PATH) not in sys.path:
        sys.path.append(str(VGGT_OMEGA_PATH))

    try:
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera
    except ImportError as exc:
        raise ImportError(
            "Could not import VGGT-Omega. Expected the repository at "
            f"{VGGT_OMEGA_PATH}. Add it as a submodule or install it with "
            "`pip install -e thirdparty/vggt-omega`."
        ) from exc

    return VGGTOmega, load_and_preprocess_images, encoding_to_camera


class VGGTOmegaProcessor(VGGTProcessor):
    """
    VGGTProcessor-compatible wrapper for VGGT-Omega.

    The rest of the pipeline expects the old VGGT output contract:
      - depth
      - pose_enc
      - world_points
      - world_points_conf

    VGGT-Omega predicts depth/depth_conf and camera pose. This wrapper decodes
    cameras and unprojects depth into world_points so the existing methods in
    VGGTProcessor can be reused.
    """

    HF_FILENAME = "model.pt"
    root = Path(__file__).parent.parent
    CHECKPOINT_DIR = (root / "data" / "checkpoints" / "VGGTOMEGA").resolve()
    LOCAL_PT = CHECKPOINT_DIR / HF_FILENAME

    def __init__(
        self,
        model_path: Path = None,
        device: str = "cuda",
        skip_model: bool = False,
        image_resolution: int = 512,
        preprocess_mode: str = "balanced",
        patch_size: int = 16,
        enable_alignment: bool = False,
    ):
        self.device = device
        self.model = None
        self.image_resolution = int(image_resolution)
        self.preprocess_mode = preprocess_mode
        self.patch_size = int(patch_size)
        self.enable_alignment = bool(enable_alignment)
        self._load_and_preprocess_images = None
        self._encoding_to_camera = None

        if not skip_model:
            VGGTOmega, load_and_preprocess_images, encoding_to_camera = _ensure_vggt_omega_imports()
            self._load_and_preprocess_images = load_and_preprocess_images
            self._encoding_to_camera = encoding_to_camera

            model_path = Path(model_path) if model_path is not None else self._ensure_local_checkpoint()
            print(f"[vggt_omega] creating model on {device} checkpoint={model_path}", flush=True)
            self.model = VGGTOmega(enable_alignment=self.enable_alignment).to(device).eval()
            print("[vggt_omega] loading checkpoint to cpu", flush=True)
            state = torch.load(model_path, map_location="cpu")
            print("[vggt_omega] loading state dict", flush=True)
            self.model.load_state_dict(state)
            del state
            print("[vggt_omega] model ready", flush=True)

        self._last_preds: dict | None = None
        self._last_imgs: np.ndarray | None = None

    def _ensure_omega_helpers(self):
        if self._load_and_preprocess_images is None or self._encoding_to_camera is None:
            _, load_and_preprocess_images, encoding_to_camera = _ensure_vggt_omega_imports()
            self._load_and_preprocess_images = load_and_preprocess_images
            self._encoding_to_camera = encoding_to_camera
        return self._load_and_preprocess_images, self._encoding_to_camera

    def _ensure_local_checkpoint(self) -> Path:
        """
        Resolve the local VGGT-Omega checkpoint.

        Unlike the original VGGT wrapper, this does not auto-download weights.
        VGGT-Omega checkpoints are gated, so place the checkpoint at:
            dx_vyzai_orthoapi/data/checkpoints/VGGT/model.pt
        or pass model_path explicitly.
        """
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        if not self.LOCAL_PT.exists():
            raise FileNotFoundError(
                "VGGT-Omega checkpoint not found. Expected "
                f"{self.LOCAL_PT}. Put the Omega checkpoint there as model.pt "
                "or pass model_path=Path(...)."
            )
        return self.LOCAL_PT

    def preprocess(self, imgs: list[Path]) -> torch.Tensor:
        """
        Turn a list of image paths into a VGGT-Omega input tensor on self.device.
        """
        load_and_preprocess_images, _ = self._ensure_omega_helpers()
        return load_and_preprocess_images(
            imgs,
            mode=self.preprocess_mode,
            image_resolution=self.image_resolution,
            patch_size=self.patch_size,
        ).to(self.device)

    def run_model(self, img_t: torch.Tensor) -> dict:
        """
        Run the VGGT-Omega forward pass and adapt outputs to VGGTProcessor keys.
        """
        if self.model is None:
            raise RuntimeError("VGGTOmegaProcessor was initialized with skip_model=True")

        with torch.inference_mode():
            out = self.model(img_t.to(self.device))

        return self._adapt_predictions(out)

    def infer(self, imgs: list[Path]) -> dict:
        """
        Shortcut: preprocess -> run_model.
        """
        if self.model is None:
            raise RuntimeError("VGGTOmegaProcessor was initialized with skip_model=True")
        img_t = self.preprocess(imgs)
        raw = self.run_model(img_t)
        self._last_preds = raw
        self._last_imgs = np.stack(
            [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in imgs],
            axis=0,
        )
        return raw

    def _adapt_predictions(self, preds: dict) -> dict:
        """
        Add old-VGGT-compatible keys to a VGGT-Omega prediction dict.
        """
        extrinsic, intrinsic = self._decode_camera(preds)
        preds["extrinsic"] = extrinsic
        preds["intrinsic"] = intrinsic

        if "world_points" not in preds:
            preds["world_points"] = self._unproject_depth_map_to_point_map(
                preds["depth"],
                extrinsic,
                intrinsic,
            )

        if "world_points_conf" not in preds:
            preds["world_points_conf"] = preds["depth_conf"]

        return preds

    def _decode_camera(self, preds: dict) -> tuple[torch.Tensor, torch.Tensor]:
        _, encoding_to_camera = self._ensure_omega_helpers()
        depth_t = preds["depth"]
        image_size_hw = tuple(depth_t.shape[2:4])
        return encoding_to_camera(preds["pose_enc"], image_size_hw)

    def get_camera_params(
        self,
        preds: dict | None = None,
        squeezed=False,
        as_numpy=False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        """
        Decode VGGT-Omega pose encoding into extrinsic and intrinsic matrices.
        """
        preds = self._ensure_preds(preds)

        if "extrinsic" in preds and "intrinsic" in preds:
            ex_list = preds["extrinsic"]
            in_list = preds["intrinsic"]
        else:
            ex_list, in_list = self._decode_camera(preds)

        ex_list = self._output_handler(ex_list, squeezed, as_numpy)
        in_list = self._output_handler(in_list, squeezed, as_numpy)
        return ex_list, in_list

    def depth_to_point_cloud(
        self,
        depth: np.ndarray | torch.Tensor | None = None,
        extrinsic: np.ndarray | torch.Tensor | None = None,
        intrinsic: np.ndarray | torch.Tensor | None = None,
        squeezed: bool = False,
        flatten: bool = False,
    ) -> np.ndarray:
        """
        Convert depth to world points using VGGT-Omega camera conventions.
        """
        if depth is None and extrinsic is None and intrinsic is None:
            point_map = self.get_points(squeezed=False, as_numpy=True)
            if squeezed:
                point_map = point_map.squeeze()
            if flatten:
                point_map = point_map.reshape(-1, 3)
            return point_map

        preds = self._ensure_preds()
        if depth is None:
            depth = self.get_depth(preds, squeezed=False, as_numpy=False)
        if extrinsic is None or intrinsic is None:
            ex_list, in_list = self.get_camera_params(preds, squeezed=False, as_numpy=False)
            if extrinsic is None:
                extrinsic = ex_list
            if intrinsic is None:
                intrinsic = in_list

        point_map = self._unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
        if isinstance(point_map, torch.Tensor):
            point_map = self._as_numpy(point_map)
        if squeezed:
            point_map = point_map.squeeze()
        if flatten:
            point_map = point_map.reshape(-1, 3)
        return point_map

    def _unproject_depth_map_to_point_map(
        self,
        depth_map: np.ndarray | torch.Tensor,
        extrinsic: np.ndarray | torch.Tensor,
        intrinsic: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """
        Unproject VGGT-Omega depth into world coordinates.

        Supports the usual raw shapes:
          - depth: (B, N, H, W, 1)
          - extrinsic: (B, N, 3, 4)
          - intrinsic: (B, N, 3, 3)
        """
        return_numpy = isinstance(depth_map, np.ndarray)
        device = self.device

        depth_t = torch.as_tensor(depth_map, device=device)
        extrinsic_t = torch.as_tensor(extrinsic, device=device, dtype=depth_t.dtype)
        intrinsic_t = torch.as_tensor(intrinsic, device=device, dtype=depth_t.dtype)

        if depth_t.ndim == 2:
            depth_t = depth_t[None, None, ..., None]
        elif depth_t.ndim == 3:
            depth_t = depth_t[None, ..., None]
        elif depth_t.ndim == 4 and depth_t.shape[-1] != 1:
            depth_t = depth_t[..., None]

        if extrinsic_t.ndim == 2:
            extrinsic_t = extrinsic_t[None, None]
        elif extrinsic_t.ndim == 3:
            extrinsic_t = extrinsic_t[None]

        if intrinsic_t.ndim == 2:
            intrinsic_t = intrinsic_t[None, None]
        elif intrinsic_t.ndim == 3:
            intrinsic_t = intrinsic_t[None]

        depth = depth_t[..., 0]
        batch_size, num_frames, height, width = depth.shape

        y, x = torch.meshgrid(
            torch.arange(height, device=depth.device, dtype=depth.dtype),
            torch.arange(width, device=depth.device, dtype=depth.dtype),
            indexing="ij",
        )
        x = x[None, None].expand(batch_size, num_frames, height, width)
        y = y[None, None].expand(batch_size, num_frames, height, width)

        fx = intrinsic_t[..., 0, 0][..., None, None]
        fy = intrinsic_t[..., 1, 1][..., None, None]
        cx = intrinsic_t[..., 0, 2][..., None, None]
        cy = intrinsic_t[..., 1, 2][..., None, None]

        camera_points = torch.stack(
            [
                (x - cx) / fx * depth,
                (y - cy) / fy * depth,
                depth,
            ],
            dim=-1,
        )

        rotation = extrinsic_t[..., :3, :3]
        translation = extrinsic_t[..., :3, 3]
        world_points = torch.einsum(
            "bnij,bnhwj->bnhwi",
            rotation.transpose(-1, -2),
            camera_points - translation[..., None, None, :],
        )

        if return_numpy:
            return world_points.detach().cpu().to(torch.float32).numpy()
        return world_points
