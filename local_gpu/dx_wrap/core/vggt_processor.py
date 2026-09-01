import cv2
import torch
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
import shutil
import trimesh

import sys
ROOT_PATH = Path(__file__).parent.parent
VGGT_PATH = ROOT_PATH / "thirdparty" / "vggt"
sys.path.append(str(VGGT_PATH))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

class VGGTProcessor:
    """Loads VGGT model and runs inference."""
    REPO_ID = "facebook/VGGT-1B"
    HF_FILENAME = "model.pt"
    root = Path(__file__).parent.parent
    CHECKPOINT_DIR = (root / "data" / "checkpoints" / "VGGT").resolve()
    LOCAL_PT = CHECKPOINT_DIR / HF_FILENAME

    def __init__(self, model_path: Path = None, device: str='cuda', skip_model: bool = False):
        self.device = device
        self.model = None
        if not skip_model:
            model_path = model_path or self._ensure_local_checkpoint()
            self.model = VGGT().to(device).eval()
            state = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state)
        self._last_preds: dict | None = None
        self._last_imgs: np.ndarray | None = None
        
    def _ensure_local_checkpoint(self) -> Path:
        """
        Download from HF (if not already cached), resolve any internal symlinks,
        then copy to checkpoints/VGGT/model.pt so torch.load() always works.
        """
        # make sure checkpoint dir exists
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        # If model.pt already exists - then no download
        if self.LOCAL_PT.exists():
            return self.LOCAL_PT
        
        downloaded = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=self.HF_FILENAME,
            local_dir=str(self.CHECKPOINT_DIR),
            local_dir_use_symlinks=False,
        )

        # Usually this is exactly CHECKPOINT_DIR / "model.pt",
        # but let's be explicit:
        downloaded_path = Path(downloaded)
        if downloaded_path != self.LOCAL_PT:
            downloaded_path.replace(self.LOCAL_PT)

        return self.LOCAL_PT
    
    def _as_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a torch tensor to a numpy array, ensuring it's on CPU and float32.
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.detach().cpu().to(torch.float32).numpy()
    
    def _output_handler(self, array: torch.Tensor, squeezed: bool, as_numpy: bool):
        """
        Handle output conversion: squeeze and convert to numpy if needed.
        """
        if squeezed:
            array = array.squeeze()
        if as_numpy:
            array = self._as_numpy(array)
        return array
    
    def _ensure_preds(self, preds: dict | None = None) -> dict:
        if preds is None:
            if self._last_preds is None:
                raise ValueError("No preds provided and no cached preds available.")
            preds = self._last_preds
        return preds
    
    def _ensure_imgs(self, imgs: np.ndarray | None = None) -> np.ndarray:
        if imgs is None:
            if self._last_imgs is None:
                raise ValueError("No imgs provided and no cached imgs available.")
            imgs = self._last_imgs
        return imgs

    def clear_buffer(self) -> None:
        """Clear cached preds and imgs."""
        self._last_preds = None
        self._last_imgs = None
    
    def preprocess(self, imgs: list[Path]) -> torch.Tensor:
        """
        Turn a list of image-paths into a batched torch tensor on self.device.
        """
        img_t = load_and_preprocess_images(imgs).to(self.device)
        return img_t

    def run_model(self, img_t: torch.Tensor) -> dict:
        """
        Run the VGGT forward pass. Returns raw torch output dict.
        """
        if self.model is None:
            raise RuntimeError("VGGTProcessor was initialized with skip_model=True")
        
        # choose dtype based on device compute capability
        cap = torch.cuda.get_device_capability(self.device)[0] if "cuda" in self.device else 0
        dtype = torch.bfloat16 if cap >= 8 else torch.float16

        # cast input once
        img_t = img_t.to(dtype)

        with torch.no_grad():
            out = self.model(img_t)
        return out

    def infer(self, imgs: list[Path]) -> dict:
        """
        Shortcut: preprocess → run_model
        """
        if self.model is None:
            raise RuntimeError("VGGTProcessor was initialized with skip_model=True")
        img_t = self.preprocess(imgs)
        raw = self.run_model(img_t)
        self._last_preds = raw
        self._last_imgs = np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                                    for p in imgs], axis=0)
        return raw

    def print_shapes(self, preds: dict | None = None) -> None:
        """
        Print the shapes of all tensors in the preds dict.
        Useful for debugging and understanding model output.
        """
        preds = self._ensure_preds(preds)
        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape} (dtype: {v.dtype})")
            elif isinstance(v, np.ndarray):
                print(f"{k}: {v.shape} (dtype: {v.dtype})")
            else:
                print(f"{k}: {type(v)}")

    def get_depth(
        self,
        preds: dict | None = None,
        squeezed=False,
        as_numpy=False
        ) -> np.ndarray | torch.Tensor:
        """
        Extract the raw depth map from preds. [1, N, H, W, 1]
        Apply squeezing and numpy conversion if requested.
        """
        preds = self._ensure_preds(preds)
        depth = preds["depth"]
        depth = self._output_handler(depth, squeezed, as_numpy)
        return depth

    def get_pose_enc(
        self,
        preds: dict | None = None,
        squeezed=False,
        as_numpy=False,
        ) -> np.ndarray | torch.Tensor:
        """
        Extract the raw pose encoding. [1, N, 9]
        Apply squeezing and numpy conversion if requested.
        """
        preds = self._ensure_preds(preds)
        pose_enc = preds["pose_enc"]
        pose_enc = self._output_handler(pose_enc, squeezed, as_numpy)
        return pose_enc

    def get_camera_params(
        self, 
        preds: dict | None = None, 
        squeezed=False, 
        as_numpy=False
        ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        """
        From a preds dict (either raw torch tensors or numpy arrays),
        compute (extrinsic 4×4, intrinsic 3×3) as numpy arrays.
        """
        preds = self._ensure_preds(preds)
        pose_enc_t = self.get_pose_enc(preds)
        depth_t = self.get_depth(preds)
        depth_shape = tuple(depth_t.shape[2:4]) 
        
        # [1, N, 4, 4] extrinsics, [1, N, 3, 3] intrinsics
        ex_list, in_list = pose_encoding_to_extri_intri(pose_enc_t, depth_shape)
        ex_list = self._output_handler(ex_list, squeezed, as_numpy)
        in_list = self._output_handler(in_list, squeezed, as_numpy)
        
        return ex_list, in_list
    
    def get_camera_poses(
        self,
        preds: dict | None = None,
        as_numpy: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract camera poses for all frames in the sequence.
        
        Parameters:
            preds: Prediction dictionary from VGGT model
            as_numpy: Whether to return numpy arrays (True) or torch tensors (False)
            
        Returns:
            Tuple of (positions, x_axes, y_axes, z_axes) where:
            - positions: (N, 3) array of camera positions in world coordinates
            - x_axes: (N, 3) array of camera x-axis direction vectors
            - y_axes: (N, 3) array of camera y-axis direction vectors
            - z_axes: (N, 3) array of camera z-axis direction vectors (viewing direction)
        """
        # Get extrinsics for all frames
        preds = self._ensure_preds(preds)
        ex_all, _ = self.get_camera_params(preds=preds, squeezed=True, as_numpy=as_numpy)
        
        # Number of frames
        N = ex_all.shape[0] if len(ex_all.shape) > 2 else 1
        ex_all = ex_all.reshape(N, 3, 4)
        
        # Extract rotation matrices and translation vectors
        rot_matrices = ex_all[:, :3, :3]  # (N, 3, 3)
        t_vecs = ex_all[:, :3, 3]         # (N, 3)
        
        # Calculate camera positions in world coordinates
        # cam_pos = -R^T * t
        if isinstance(rot_matrices, np.ndarray):
            # Using numpy
            cam_positions = -np.einsum('nij,nj->ni', np.transpose(rot_matrices, (0, 2, 1)), t_vecs)
            
            # Camera axes (columns of R^T)
            x_axes = np.transpose(rot_matrices, (0, 2, 1))[:, :, 0]  # First column of R^T
            y_axes = np.transpose(rot_matrices, (0, 2, 1))[:, :, 1]  # Second column of R^T
            z_axes = np.transpose(rot_matrices, (0, 2, 1))[:, :, 2]  # Third column of R^T
        else:
            # Using torch
            cam_positions = -torch.bmm(torch.transpose(rot_matrices, 1, 2), 
                                    t_vecs.unsqueeze(-1)).squeeze(-1)
            
            # Camera axes (columns of R^T)
            R_t = torch.transpose(rot_matrices, 1, 2)
            x_axes = R_t[:, :, 0]
            y_axes = R_t[:, :, 1]
            z_axes = R_t[:, :, 2]
        
        return cam_positions, x_axes, y_axes, z_axes

    def get_camera_pose(
        self,
        preds: dict | None = None,
        frame_idx: int = 0,
        as_numpy: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract camera pose for a specific frame.
        
        Parameters:
            preds: Prediction dictionary from VGGT model
            frame_idx: Index of the frame to extract pose for
            as_numpy: Whether to return numpy arrays (True) or torch tensors (False)
            
        Returns:
            Tuple of (position, x_axis, y_axis, z_axis) where:
            - position: (3,) array of camera position in world coordinates
            - x_axis: (3,) array of camera x-axis direction vector
            - y_axis: (3,) array of camera y-axis direction vector
            - z_axis: (3,) array of camera z-axis direction vector (viewing direction)
        """
        # Get extrinsics for all frames
        preds = self._ensure_preds(preds)
        ex_all, _ = self.get_camera_params(preds=preds, squeezed=False, as_numpy=as_numpy)
        
        # Extract the specific frame
        ex = ex_all[:, frame_idx, :, :].reshape(3, 4)
        
        # Extract rotation matrix and translation vector
        rot_matrix = ex[:3, :3]  # (3, 3)
        t_vec = ex[:3, 3]        # (3,)
        
        # Calculate camera position in world coordinates
        # cam_pos = -R^T * t
        if isinstance(rot_matrix, np.ndarray):
            # Using numpy
            cam_position = -np.matmul(rot_matrix.T, t_vec)
            
            # Camera axes (columns of R^T)
            x_axis = rot_matrix.T[:, 0]  # First column of R^T
            y_axis = rot_matrix.T[:, 1]  # Second column of R^T
            z_axis = rot_matrix.T[:, 2]  # Third column of R^T
        else:
            # Using torch
            cam_position = -torch.matmul(rot_matrix.T, t_vec)
            
            # Camera axes (columns of R^T)
            x_axis = rot_matrix.T[:, 0]
            y_axis = rot_matrix.T[:, 1]
            z_axis = rot_matrix.T[:, 2]
        
        return cam_position, x_axis, y_axis, z_axis
    
    def get_points(
        self,
        preds: dict | None = None,
        squeezed: bool = False,
        as_numpy: bool = False,
        flatten: bool = False
        ) -> np.ndarray:
        """
        Extract the world points from preds. [1, N, H, W, 3]
        Apply squeezing and numpy conversion if requested.
        """
        preds = self._ensure_preds(preds)
        pts = preds["world_points"]
        pts = self._output_handler(pts, squeezed, as_numpy)
        if flatten:
            pts = pts.reshape(-1, 3)  # Flatten to (N, 3) if requested
        return pts
    
    def get_points_conf(
        self,
        preds: dict | None = None,
        squeezed: bool = False,
        as_numpy: bool = False,
        flatten: bool = False,
        ) -> np.ndarray:
        """
        Extract the world points confidence from preds. [1, N, H, W]
        Apply squeezing and numpy conversion if requested.
        """
        preds = self._ensure_preds(preds)
        conf = preds["world_points_conf"]
        conf = self._output_handler(conf, squeezed, as_numpy)
        if flatten:
            conf = conf.reshape(-1)
        return conf
    
    def get_points_colors(
        self,
        points: np.ndarray | torch.Tensor | None = None,
        imgs: np.ndarray | None = None,
        squeezed: bool = False,
        flatten: bool = False,
        as_hex: bool = False,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray | torch.Tensor:
        """
        Return per-frame RGB colors aligned to VGGT world_points.

        Uses original/cached RGB images, resized to the VGGT point-map resolution.
        This avoids using VGGT-preprocessed low-contrast colors.
        """
        if points is None:
            points = self.get_points(squeezed=False, as_numpy=False)

        imgs = self._ensure_imgs(imgs)
        imgs = np.asarray(imgs)

        if points.ndim == 2:
            raise ValueError("Passed flattened points array; use unflattened points.")

        has_batch = points.ndim == 5 and points.shape[0] == 1
        pts_shape = points.squeeze(0).shape if has_batch else points.shape

        if len(pts_shape) != 4 or pts_shape[-1] != 3:
            raise ValueError(f"Expected points shape (T,H,W,3) or (1,T,H,W,3), got {points.shape}")

        T_pts, H_pts, W_pts, _ = pts_shape
        T_img, H_img, W_img, C_img = imgs.shape

        if C_img != 3:
            raise ValueError(f"Expected RGB imgs with shape (T,H,W,3), got {imgs.shape}")
        if T_img != T_pts:
            raise ValueError(f"Number of images ({T_img}) != number of point frames ({T_pts})")

        if imgs.dtype != np.uint8:
            imgs = np.clip(imgs, 0, 255).astype(np.uint8)

        if (H_img, W_img) != (H_pts, W_pts):
            imgs = np.stack(
                [
                    cv2.resize(im, (W_pts, H_pts), interpolation=interpolation)
                    for im in imgs
                ],
                axis=0,
            )

        colors = imgs[None, ...] if has_batch else imgs

        if isinstance(points, torch.Tensor):
            colors = torch.from_numpy(colors).to(device=points.device)

        if squeezed:
            colors = colors.squeeze(0) if has_batch else colors

        if flatten:
            colors = colors.reshape(-1, 3)

        if as_hex:
            colors_np = colors.detach().cpu().numpy() if isinstance(colors, torch.Tensor) else colors
            colors_np = colors_np.astype(np.uint8)
            colors = np.apply_along_axis(
                lambda x: f"#{x[0]:02x}{x[1]:02x}{x[2]:02x}",
                axis=-1,
                arr=colors_np,
            )

        return colors


    def preds_to_numpy(self, preds: dict | None = None) -> dict:
        """
        Convert all tensor/array outputs in preds to NumPy arrays.
        Supports:
          - torch.Tensor → float32 NumPy
          - np.ndarray    → as-is
          - list of tensors/arrays → stacked into one array
        Skips any other types.
        """
        preds = self._ensure_preds(preds)
        np_preds = {}
        for k, v in preds.items():
            # 1) Single tensor → NumPy
            if isinstance(v, torch.Tensor):
                np_preds[k] = self._as_numpy(v)
            # 2) Already a NumPy array
            elif isinstance(v, np.ndarray):
                np_preds[k] = v
            # 3) List of tensors or arrays → stack
            elif isinstance(v, list):
                converted = []
                for elem in v:
                    if isinstance(elem, torch.Tensor):
                        converted.append(self._as_numpy(elem))
                    elif isinstance(elem, np.ndarray):
                        converted.append(elem)
                    else:
                        raise TypeError(f"Cannot convert element of type {type(elem)} in list for key '{k}'")
                # stack along first axis
                np_preds[k] = np.stack(converted, axis=0)
            # 4) Otherwise skip
            else:
                # you can log or silently ignore unsupported types
                continue

        return np_preds
    
    def save_preds_npz(self, path: str | Path, preds: dict | None = None, *, compressed: bool = True) -> None:
        """
        Save the whole preds dict to a compressed .npz file.
        Uses preds_to_numpy to convert tensors -> numpy.
        Non-numeric / unsupported entries are skipped.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        preds = self._ensure_preds(preds)
        np_preds = self.preds_to_numpy(preds)
        # np.savez_compressed expects key=array kwargs
        if compressed:
            np.savez_compressed(path, **np_preds)
        else:
            np.savez(path, **np_preds)

    def load_preds_npz(
        self,
        path: str | Path,
        as_torch: bool = True,
        device: str | None = None,
    ) -> dict:
        """
        Load a preds .npz file previously saved by save_preds_npz.

        Parameters:
            path:     path to the .npz file
            as_torch: if True, return torch tensors and loads them to buffer; otherwise NumPy arrays
            device:   device for torch tensors (default: self.device or 'cpu')

        Returns:
            dict mapping key -> np.ndarray or torch.Tensor
        """
        path = Path(path)
        data = np.load(path)
        

        if not as_torch:
            # plain numpy version
            return {k: data[k] for k in data.files}

        if device is None:
            device = getattr(self, "device", "cpu")

        out: dict[str, torch.Tensor] = {}
        for k in data.files:
            arr = data[k]
            t = torch.from_numpy(arr).to(device=device)
            out[k] = t
        self._last_preds = out
        return out
    
    def save_reconstruction_npz(
        self,
        path: str | Path,
        *,
        preds: dict | None = None,
        imgs: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        extr: np.ndarray | None = None,
        intr: np.ndarray | None = None,
        conf: np.ndarray | None = None,
        pts: np.ndarray | None = None,
        pts_flat: np.ndarray | None = None,
        confs_flat: np.ndarray | None = None,
        cols_flat: np.ndarray | None = None,
        cam_positions: np.ndarray | None = None,
        x_axes: np.ndarray | None = None,
        y_axes: np.ndarray | None = None,
        z_axes: np.ndarray | None = None,
        img_ids: np.ndarray | None = None,
        best_frame_masks: np.ndarray | None = None,
        compressed: bool = True,
    ) -> None:
        """
        Save a full reconstruction snapshot to .npz.

        If most arguments are None, the function reproduces the logic:

            depth        = reconstr_class.get_depth()
            ex, inn      = reconstr_class.get_camera_params()
            conf         = reconstr_class.get_points_conf()
            imgs         = np.stack(best_frames, axis=0)  # or self._last_imgs if imgs not passed
            pts          = reconstr_class.get_points(as_numpy=True)
            pts_flat     = pts.reshape(-1, 3)
            confs_flat   = reconstr_class.get_points_conf(as_numpy=True, flatten=True)
            cols_flat    = reconstr_class.get_points_colors(flatten=True)
            extr, intr   = reconstr_class.get_camera_params(squeezed=True, as_numpy=True)
            cam_positions, x_axes, y_axes, z_axes = reconstr_class.get_camera_poses(as_numpy=True)

        Notes:
          - If `imgs` is None, uses self._last_imgs.
          - Any argument you explicitly pass is used as-is and not recomputed.
          - Fields that remain None are simply not written to the .npz.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        preds = self._ensure_preds(preds)

        if imgs is None:
            try:
                imgs = self._ensure_imgs(None)
            except ValueError:
                imgs = None

        if depth is None:
            depth = self.get_depth(preds=preds, squeezed=False, as_numpy=True)

        if conf is None:
            conf = self.get_points_conf(
                preds=preds, squeezed=False, as_numpy=True, flatten=False
            )

        if pts is None:
            pts = self.get_points(
                preds=preds, squeezed=False, as_numpy=True, flatten=False
            )  # (N, H, W, 3)

        if pts_flat is None:
            pts_flat = pts.reshape(-1, 3)

        if confs_flat is None and conf is not None:
            confs_flat = self.get_points_conf(
                preds=preds, squeezed=False, as_numpy=True, flatten=True
            )

        if cols_flat is None and imgs is not None:
            cols_flat = self.get_points_colors(
                points=pts,
                imgs=imgs,
                squeezed=True,
                flatten=True,
            )
            if isinstance(cols_flat, np.ndarray) and cols_flat.dtype != np.uint8:
                cols_flat = np.clip(cols_flat, 0, 255).astype(np.uint8)

        if extr is None or intr is None:
            ex_np, in_np = self.get_camera_params(
                preds=preds, squeezed=True, as_numpy=True
            )
            if extr is None:
                extr = ex_np
            if intr is None:
                intr = in_np
                
        if cam_positions is None or x_axes is None or y_axes is None or z_axes is None:
            cam_pos_np, x_np, y_np, z_np = self.get_camera_poses(
                preds=preds, as_numpy=True
            )
            if cam_positions is None:
                cam_positions = cam_pos_np
            if x_axes is None:
                x_axes = x_np
            if y_axes is None:
                y_axes = y_np
            if z_axes is None:
                z_axes = z_np

        raw_dict = {
            "depth": depth,                 # (N, H, W)
            "extr": extr,                   # (N, 3, 4) or (N, 4, 4)
            "intr": intr,                   # (N, 3, 3)
            "conf": conf,                   # (N, H, W)
            "imgs": imgs,                   # (N, H, W, 3)
            "pts": pts,                     # (N, H, W, 3)
            "pts_flat": pts_flat,           # (M, 3)
            "confs_flat": confs_flat,       # (M,)
            "cols_flat": cols_flat,         # (M, 3)
            "cam_positions": cam_positions, # (N, 3)
            "x_axes": x_axes,               # (N, 3)
            "y_axes": y_axes,               # (N, 3)
            "z_axes": z_axes,               # (N, 3)
            "img_ids": img_ids,             # optional (N,)
            "best_frame_masks": best_frame_masks,  # optional (N, H, W)
        }

        save_dict = {k: v for k, v in raw_dict.items() if v is not None}
        if compressed:
            np.savez_compressed(path, **save_dict)
        else:
            np.savez(path, **save_dict)
        print(f"Reconstruction saved to {path}")

    @staticmethod
    def load_reconstruction_npz(path: str | Path) -> dict:
        """
        Load reconstruction results from a .npz file produced by save_reconstruction_npz().

        Returns a dict with keys:
          depth, extr, intr, conf, imgs,
          pts, pts_flat, confs_flat, cols_flat,
          cam_positions, x_axes, y_axes, z_axes,
          img_ids, best_frame_masks

        Any key that was not present in the file is returned as None.
        """
        path = Path(path)
        data = np.load(path)

        def get(name: str, default=None):
            return data[name] if name in data.files else default

        return {
            "depth":            get("depth"),
            "extr":             get("extr"),
            "intr":             get("intr"),
            "conf":             get("conf"),
            "imgs":             get("imgs"),
            "pts":              get("pts"),
            "pts_flat":         get("pts_flat"),
            "confs_flat":       get("confs_flat"),
            "cols_flat":        get("cols_flat"),
            "cam_positions":    get("cam_positions"),
            "x_axes":           get("x_axes"),
            "y_axes":           get("y_axes"),
            "z_axes":           get("z_axes"),
            "img_ids":          get("img_ids"),
            "best_frame_masks": get("best_frame_masks"),
        }


    def segment_pointcloud(
        self,
        masks: list[np.ndarray],
        preds: dict | None = None,
        imgs: np.ndarray | None = None,
        return_sel: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (pts, cols) for *one* object mask-list:
          - pts:  (N,3) world points
          - cols: (N,3) uint8 RGB
        """
        preds = self._ensure_preds(preds)
        imgs = self._ensure_imgs(imgs)
        
        # 1) grab world_points & colors
        pts = self.get_points(preds, squeezed=True, as_numpy=True)           # (T,H_d,W_d,3)
        cols = self.get_points_colors(points=pts, imgs=imgs, squeezed=True) # (T,H_d,W_d,3)

        T, H_d, W_d, _ = pts.shape
        pts_flat  = pts.reshape(-1, 3)
        cols_flat = cols.reshape(-1, 3).astype(np.uint8)

        # 2) build & resize mask stack → flatten
        m = np.stack(masks, axis=0).astype(np.uint8)  # (T, H_img, W_img)
        if (m.shape[1], m.shape[2]) != (H_d, W_d):
            m = np.stack([
                cv2.resize(frame, (W_d, H_d),
                           interpolation=cv2.INTER_NEAREST)
                for frame in m
            ], axis=0)
        sel = (m > 0).reshape(-1)
        if return_sel:
            return pts_flat, cols_flat, sel
        return pts_flat[sel], cols_flat[sel]
    


    def depth_to_point_cloud(self,
                             depth: np.ndarray | torch.Tensor | None = None,
                             extrinsic: np.ndarray | torch.Tensor | None = None,
                             intrinsic: np.ndarray | torch.Tensor | None = None,
                             squeezed: bool=False,
                             flatten: bool=False
                            ) -> np.ndarray:
        """
        Convert a depth map to a point cloud using the provided extrinsic and intrinsic matrices.
        depth: (H, W) or (1, H, W)
        extrinsic: (3, 4) or (N, 3, 4)
        intrinsic: (3, 3) or (N, 3, 3)
        Returns: (N, H*W, 3) point cloud in world coordinates.
        """
        preds = self._ensure_preds()
        if depth is None:
            depth = self.get_depth(preds, squeezed=False, as_numpy=True)
        
        if extrinsic is None or intrinsic is None:
            ex_list, in_list = self.get_camera_params(
                preds=preds,
                squeezed=False,
                as_numpy=True
            )
            if extrinsic is None:
                extrinsic = ex_list
            if intrinsic is None:
                intrinsic = in_list
            
        if depth.ndim == 2:
            depth = depth[None, ..., None]  # Add batch dimension
        if extrinsic.ndim == 2:
            extrinsic = extrinsic[None]
        if intrinsic.ndim == 2:
            intrinsic = intrinsic[None]
        
        # Unproject depth map to point map
        point_map = unproject_depth_map_to_point_map(
            depth,
            extrinsic,
            intrinsic
        )
        
        if squeezed:
            point_map = point_map.squeeze()
        if flatten:
            point_map = point_map.reshape(-1, 3)
            
        return point_map  # Flatten to (N, H*W, 3)
    
    @staticmethod
    def save_bones_as_glb(femur_pts: np.ndarray,
                          tibia_pts: np.ndarray,
                          output_path: str,
                          femur_colors: np.ndarray=None,
                          tibia_colors: np.ndarray=None):
        """
        Merge femur and tibia 3D points (and optional RGB colors) into a scene,
        then export the combined scene as a binary GLTF (.glb) file using trimesh.
        """
        # Create trimesh PointCloud objects for each bone
        # If color arrays are given, ensure they are RGBA (uint8)
        femur_pc = trimesh.points.PointCloud(vertices=femur_pts, colors=None)
        tibia_pc = trimesh.points.PointCloud(vertices=tibia_pts, colors=None)
        
        if femur_colors is not None:
            # Convert (N,3) RGB to (N,4) RGBA with full opacity
            rgba = np.hstack((femur_colors.astype(np.uint8),
                              255 * np.ones((femur_colors.shape[0],1), np.uint8)))
            femur_pc = trimesh.points.PointCloud(vertices=femur_pts, colors=rgba)
        if tibia_colors is not None:
            rgba = np.hstack((tibia_colors.astype(np.uint8),
                              255 * np.ones((tibia_colors.shape[0],1), np.uint8)))
            tibia_pc = trimesh.points.PointCloud(vertices=tibia_pts, colors=rgba)

        # Create a scene and add both point clouds
        scene = trimesh.Scene()
        scene.add_geometry(femur_pc, node_name='Femur')
        scene.add_geometry(tibia_pc, node_name='Tibia')

        # Export the scene as binary GLTF (GLB); this returns bytes or writes to file
        scene.export(file_obj=output_path, file_type='glb')
        
    @staticmethod
    def save_points_as_glb(pts: np.ndarray, cols: np.ndarray, output_path: str) -> None:
        """
        Save a point cloud with colors as a GLB file.
        pts: (N,3) array of 3D points.
        cols: (N,3) array of RGB colors.
        output_path: path to save the GLB file.
        """
        rgba = np.hstack((cols.astype(np.uint8),
                          255 * np.ones((cols.shape[0],1), np.uint8)))
        pc = trimesh.points.PointCloud(vertices=pts, colors=rgba)
        scene = trimesh.Scene()
        scene.add_geometry(pc)
        scene.export(file_obj=output_path, file_type='glb')
                              
        

