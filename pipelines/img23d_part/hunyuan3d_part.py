"""Mesh-to-parts pipeline backed by Tencent Hunyuan3D-Part (XPart + P3-SAM).

Wraps `tencent/Hunyuan3D-Part <https://huggingface.co/tencent/Hunyuan3D-Part>`_
— a separate model from Hunyuan3D-2.1 that takes a *whole* mesh and
decomposes it into semantically meaningful parts (e.g. chair → seat +
legs + back as distinct geometry).

Two stages run end-to-end inside the pipeline:

  1. **P3-SAM** (``p3sam.safetensors``) — predicts part bounding boxes
     from the input mesh.
  2. **XPart** (``model.safetensors`` + ``conditioner.safetensors`` +
     ``shapevae.safetensors``) — regenerates each detected part as a
     standalone, high-fidelity mesh and emits both an assembled
     "decomposed" output and an "exploded" visualization.

Returns paths to four ``.glb`` files on the runner pod:

  * ``mesh_path``       — assembled decomposed mesh (parts joined)
  * ``exploded_path``   — exploded view with parts spatially separated
  * ``bbox_path``       — the bounding-box wireframe only
  * ``gt_bbox_path``    — input mesh overlaid with predicted bboxes
                          (debug view)

The pipeline is **mesh-in / meshes-out** — NOT image-in.  The api wires
this so callers either upload a base64-encoded ``.glb`` directly or pass
a ``mesh_url`` from a prior Hunyuan3D-2.1 ``/v1/images/3d`` run.

Heavy dependencies (``torch``, ``partgen``, ``trimesh``, ``spconv``,
``torch_cluster``) are imported lazily inside :meth:`_load` so the rest
of the runner remains usable on machines that lack the
XPart-specific deps.  Missing-dep failures surface as a clean
``RuntimeError`` from ``_load`` describing how to install them.
"""

from __future__ import annotations

import base64
import binascii
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from config import SD_OUTPUT_DIR
from models import ModelTask
from pipelines.base import InProcessPipeline


# Yaml drives the on-disk path (``details.model_path`` in
# .models.yaml).  See ``pipelines/rembg/rmbg.py`` for the
# pattern; this is identical except the registry id is
# ``hunyuan3d-part``.
_MODEL_ID = "hunyuan3d-part"

# Where to persist generated .glb outputs for HTTP retrieval.  Same
# root the other 3D pipelines write into; the runner's
# ``/v1/pipelines/img23d_part/files/{filename}`` proxy serves from
# here.
_OUTPUT_DIR = os.environ.get(
    "HUNYUAN3D_PART_OUTPUT_DIR",
    os.path.join(SD_OUTPUT_DIR, "3d_parts"),
)

# Default octree resolution for marching cubes.  Higher = finer mesh
# detail but quadratically more memory + time.  512 matches the demo
# script; 256 is faster for iteration if needed.
_DEFAULT_OCTREE_RESOLUTION = int(
    os.environ.get("HUNYUAN3D_PART_OCTREE_RESOLUTION", "512")
)

# Hard cap on the number of parts the conditioner has to attend over.
# The cross-attention activation scales as roughly ``K * N_per_part *
# hidden_dim`` where ``N_per_part`` is the hardcoded 81920 surface
# sample count.  Measured peaks on a 24 GB 3090:
#
#   K=6   → ~3 GB activation, fits comfortably
#   K=8   → ~4 GB,            fits with headroom for VAE + DiT residue
#   K=16  → ~8 GB,            OOMs (17.8 GB resident + 8.1 GB alloc)
#   K=25  → ~12 GB,           OOMs hard
#
# Default cap at 8 — preserves the 6-10 part range that real
# Hunyuan3D-2.1 outputs produce while clipping pathological fixture
# meshes that detect 20-50 parts.  Override via env if you have more
# headroom; set to 0 to disable.
_DEFAULT_MAX_PARTS = int(os.environ.get("HUNYUAN3D_PART_MAX_PARTS", "8"))


_INSTALL_HINT = (
    "Hunyuan3D-Part dependencies are missing.  The pipeline needs the\n"
    "vendored XPart package, plus its CUDA-bound deps:\n"
    "  pip install spconv-cu124 fpsample addict easydict scikit-learn\n"
    "  pip install -e vendors/Hunyuan3D-Part/XPart\n"
    "and Sonata weights from facebook/sonata (downloaded automatically\n"
    "on first load if HF_TOKEN is set)."
)


class Hunyuan3DPartPipeline(InProcessPipeline):
    """Mesh-to-parts pipeline backed by Tencent Hunyuan3D-Part (XPart)."""

    name = "img23d_part"
    task = ModelTask.IMAGETO3D

    #: Identifier in ``.models.yaml`` — used to look up
    #: ``details.model_path`` lazily on first ``_load``.
    model_id: str = _MODEL_ID

    def __init__(self, model_path: Optional[str] = None) -> None:
        super().__init__()
        # ``model_path`` is the on-disk directory containing
        # ``model/``, ``conditioner/``, ``shapevae/``, ``p3sam/``,
        # ``scheduler/``.  When None, resolved from ``.models.yaml``
        # at first load.  Tests pass an explicit path.
        self._model_path: Optional[str] = model_path
        self._impl: Any = None
        # Set by ``_load`` to ``cuda:N`` (N = freest GPU) or ``cpu``.
        self._device: str = "cpu"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _resolve_model_path(self) -> str:
        """Look up ``details.model_path`` from the model registry."""
        from utils.model_loader import ModelLoader  # local — avoid import cycle

        loader = ModelLoader()
        model = loader.get_model_by_id(self.model_id)
        if model is None:
            raise RuntimeError(
                f"Hunyuan3D-Part pipeline could not find '{self.model_id}' "
                f"in the model registry (.models.yaml).  Add an entry with "
                f"``details.model_path`` pointing at the on-disk weights "
                f"directory."
            )
        path = getattr(model.details, "model_path", None)
        if not path:
            raise RuntimeError(
                f"Hunyuan3D-Part pipeline: model '{self.model_id}' has no "
                f"``details.model_path`` set in .models.yaml.  This field "
                f"is required for in_process pipelines."
            )
        return path

    async def _load(self) -> None:
        """Lazy-load XPart's PartFormerPipeline + torch."""
        try:
            import torch  # type: ignore[import-not-found]
            from partgen.partformer_pipeline import (  # type: ignore[import-not-found]
                PartFormerPipeline,
            )
        except ImportError as e:
            raise RuntimeError(_INSTALL_HINT) from e

        os.makedirs(_OUTPUT_DIR, exist_ok=True)

        if self._model_path is None:
            self._model_path = self._resolve_model_path()

        # XPart needs ~23 GB peak VRAM in a single-GPU layout, so we
        # shard across two cards when available: the diffusion DiT
        # (the 6 GB workspace consumer) gets its own card; the
        # conditioner + VAE + P3-SAM stay together on the primary
        # card.  Cross-card tensor transfer is handled by
        # accelerate's AlignDevicesHook (applied to ``self._impl.model``
        # further down).
        #
        # Device placement is yaml-driven via the shared
        # ``pipelines._gpu_select`` helper — same path
        # ``main_gpu`` / ``tensor_split`` plumbing as rembg + img23d.
        from pipelines._gpu_select import pick_device, device_hints_from_model

        try:
            from utils.model_loader import ModelLoader
            model = ModelLoader().get_model_by_id(self.model_id)
        except Exception:
            model = None
        choice = pick_device(
            **device_hints_from_model(model),
            min_vram_gb=20.0,  # XPart needs at least one 24 GB-class card
            logger=self._logger,
        )
        if not choice.primary.startswith("cuda"):
            raise RuntimeError(
                "No CUDA device meets XPart's 20 GB total-VRAM floor.  "
                "Adjust ``main_gpu``/``tensor_split`` in .models.yaml's "
                "``hunyuan3d-part`` entry or lower the floor in code."
            )
        device = choice.primary
        secondary_device: Optional[str] = choice.secondary

        # Precision knob.  Default fp16 — applied SELECTIVELY:
        #
        #   * DiT (``self.model``): cast to the chosen dtype.  This is
        #     the 6.6 GB module and the biggest memory win.  Plain
        #     transformer math, well-tested in fp16.
        #   * VAE, conditioner, P3-SAM/bbox_predictor: stay in fp32.
        #     The conditioner + P3-SAM contain Sonata which uses
        #     ``spconv``'s implicit_gemm — spconv 2.x has fp16 kernels
        #     for SOME shape configurations but not all, and our
        #     Sonata config trips ``can't find suitable algorithm for
        #     0`` in fp16.  Keeping them fp32 sidesteps the issue
        #     entirely.  Cross-module dtype mismatch is handled by
        #     XPart's __call__ which casts inputs to ``self.dtype``
        #     just before feeding the DiT.
        dtype_env = os.environ.get("HUNYUAN3D_PART_DTYPE", "fp16").lower()
        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        if dtype_env not in dtype_map:
            raise RuntimeError(
                f"HUNYUAN3D_PART_DTYPE={dtype_env!r} is not one of "
                f"{sorted(dtype_map)}.  Default is fp16."
            )
        target_dtype = dtype_map[dtype_env]

        self._logger.info(
            f"Loading Hunyuan3D-Part (XPart) from {self._model_path} "
            f"in fp32 (DiT will downcast to {dtype_env}) on {device}"
        )
        # Load EVERYTHING in fp32 first.  We selectively downcast the
        # DiT below.  XPart's smart_load_model joins HY3DGEN_MODELS +
        # model_path; we pass an absolute path so resolution is
        # independent of the env value.
        self._impl = PartFormerPipeline.from_pretrained(
            model_path=self._model_path,
            dtype=torch.float32,
            device=device,
        )

        try:
            if device.startswith("cuda"):
                # Move to device in fp32 (no dtype cast yet — see below
                # for the selective DiT cast).
                self._impl.to(device=device)
        except Exception as e:  # noqa: BLE001
            self._logger.warning(
                f"Could not move Hunyuan3D-Part to {device} ({e}); using CPU"
            )

        # XPart returns ``(out, None)`` for the debug tuple unless
        # ``verbose=True``.  We want the bbox / exploded / gt_bbox
        # outputs (the api wraps each with its own download URL), so
        # flip the flag post-init.
        if hasattr(self._impl, "verbose"):
            self._impl.verbose = True

        # XPart's __call__ reads ``self.dtype`` (line 588 of
        # partformer_pipeline.py).  ``from_pretrained`` doesn't set
        # it on the instance — it's expected to be assigned by the
        # caller via ``.to(dtype=)``.  Since we call ``.to`` without
        # a dtype above (in fp32 mode), the attribute never exists.
        # Set it explicitly to keep the upstream code path happy.
        self._impl.dtype = target_dtype

        # Selectively downcast the DiT (the 6.6 GB module).  XPart's
        # ``self.dtype`` is read by __call__ to cast DiT inputs just
        # before feeding the model — so we also update ``self.dtype``
        # to match the new DiT dtype, otherwise inputs stay fp32 and
        # we get a mixed-dtype error inside the model forward.
        if target_dtype != torch.float32 and hasattr(self._impl, "model"):
            try:
                self._impl.model = self._impl.model.to(dtype=target_dtype)
                self._impl.dtype = target_dtype
                self._logger.info(
                    f"DiT downcast to {dtype_env}; conditioner + VAE + "
                    f"P3-SAM kept in fp32 (spconv kernel coverage)"
                )
            except Exception as e:  # noqa: BLE001
                self._logger.warning(
                    f"Could not downcast DiT to {dtype_env} ({e}); "
                    f"falling back to fp32"
                )

        # Shard the diffusion DiT (``self.model``) onto the secondary
        # GPU when we have one.  This is the module that owns the
        # ~6 GB inference workspace allocation — putting it on its own
        # card means it has 24 GB to play in instead of competing with
        # the conditioner + VAE + P3-SAM (~17 GB resident) on a
        # shared card.
        #
        # ``AlignDevicesHook`` from accelerate auto-transfers the
        # forward call's args to ``execution_device`` before invoking
        # the module and moves the output back to the primary device
        # afterwards.  No XPart code change required.
        if (
            secondary_device is not None
            and secondary_device != device
            and self._impl is not None
            and hasattr(self._impl, "model")
        ):
            try:
                from accelerate.hooks import (  # type: ignore[import-not-found]
                    AlignDevicesHook,
                    add_hook_to_module,
                )

                self._impl.model.to(secondary_device)
                hook = AlignDevicesHook(
                    execution_device=torch.device(secondary_device),
                    io_same_device=True,  # output goes back to caller's device
                    offload=False,
                )
                add_hook_to_module(self._impl.model, hook)
                self._logger.info(
                    f"DiT submodule sharded onto {secondary_device} "
                    f"with accelerate.AlignDevicesHook; cross-card "
                    f"transfers handled by the hook"
                )
            except Exception as e:  # noqa: BLE001
                self._logger.warning(
                    f"Could not shard DiT to {secondary_device} "
                    f"({e}); falling back to single-card layout"
                )

        # Remember the chosen device so _run can move input tensors to it.
        self._device = device

    async def _run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run XPart on one input mesh.

        Payload schema::

            {
                "mesh_b64":      "<base64-encoded .glb>",   # required
                "octree_resolution": int,                     # optional, default 512
                "seed":          int,                         # optional
            }

        Response::

            {
                "id":             "<uuid>",
                "elapsed_sec":    <float>,
                "mesh_path":      "/data/sd-out/3d_parts/<id>_decomposed.glb",
                "exploded_path":  "/data/sd-out/3d_parts/<id>_exploded.glb",
                "bbox_path":      "/data/sd-out/3d_parts/<id>_bbox.glb",
                "gt_bbox_path":   "/data/sd-out/3d_parts/<id>_gt_bbox.glb",
            }

        The api wraps these with ``mesh_url`` / ``exploded_url`` /
        ``bbox_url`` / ``gt_bbox_url`` fields that route through
        ``GET /v1/images/3d/parts/{filename}``.
        """
        mesh_b64 = payload.get("mesh_b64")
        if not mesh_b64:
            raise ValueError("mesh_b64 is required")

        try:
            mesh_bytes = base64.b64decode(mesh_b64, validate=True)
        except (ValueError, binascii.Error) as e:
            raise ValueError(f"mesh_b64 is not valid base64: {e}") from e

        # XPart wants a file path, not bytes — its loader uses trimesh.load
        # under the hood which goes through the file extension.  Use a
        # temp file in the output dir so the mesh stays accessible if we
        # need to debug a failed run.
        gen_id = uuid.uuid4().hex[:12]
        input_path = os.path.join(_OUTPUT_DIR, f"{gen_id}_input.glb")
        with open(input_path, "wb") as f:
            f.write(mesh_bytes)

        octree_resolution = _DEFAULT_OCTREE_RESOLUTION
        try:
            if "octree_resolution" in payload:
                octree_resolution = int(payload["octree_resolution"])
        except (TypeError, ValueError):
            pass

        kwargs: Dict[str, Any] = {}
        seed = 42
        if "seed" in payload:
            try:
                seed = int(payload["seed"])
                kwargs["seed"] = seed
            except (TypeError, ValueError):
                pass

        # Resolve the per-call max-parts cap.  Payload override
        # wins (caller may know they have headroom for more parts);
        # otherwise use the env default.  Set to 0 to disable.
        max_parts = _DEFAULT_MAX_PARTS
        try:
            if "max_parts" in payload:
                max_parts = int(payload["max_parts"])
        except (TypeError, ValueError):
            pass

        started = time.perf_counter()

        # Cap the number of bbox-predicted parts so the conditioner's
        # cross-attention activation stays bounded.  P3-SAM happily
        # returns 20-50 parts for fixture meshes; the downstream
        # ``conditioner(part_surface_inbbox, object_surface)`` then
        # OOMs allocating ~7-8 GB of fp32 layer_norm intermediates
        # under autocast.  Pre-compute the aabb here, sort by box
        # volume descending, keep top ``max_parts``, then pass the
        # capped aabb back to ``self._impl(...)`` — XPart accepts
        # ``aabb=`` as a kwarg and uses it instead of running
        # ``predict_bbox`` itself.
        capped_aabb = None
        if max_parts > 0 and getattr(self._impl, "bbox_predictor", None) is not None:
            try:
                import trimesh  # local — keeps cold start fast

                _mesh_in = trimesh.load(input_path, force="mesh")
                aabb = self._impl.predict_bbox(  # type: ignore[union-attr]
                    _mesh_in, seed=seed
                )
                # aabb shape: [B, K, 2, 3]  (min/max corners per part)
                k = aabb.shape[1]
                if k > max_parts:
                    extents = aabb[:, :, 1, :] - aabb[:, :, 0, :]
                    volumes = (extents.abs() + 1e-9).prod(dim=-1)  # [B, K]
                    top_idx = volumes[0].argsort(descending=True)[:max_parts]
                    capped_aabb = aabb[:, top_idx, :, :]
                    self._logger.info(
                        f"Capped bbox count {k} → {max_parts} "
                        f"(sorted by volume desc) to keep "
                        f"conditioner cross-attention bounded"
                    )
                else:
                    capped_aabb = aabb
            except Exception as e:  # noqa: BLE001
                self._logger.warning(
                    f"Pre-bbox capping failed ({e}); letting XPart "
                    f"predict_bbox run in-line (may OOM on high-K meshes)"
                )
                capped_aabb = None

            # Drop the bbox-prediction workspace cache before the
            # conditioner runs.  predict_bbox's intermediate tensors
            # (Sonata serialization buffers, point-cloud features)
            # otherwise stay pooled on primary's allocator — adding
            # several GB on top of the conditioner's own 8 GB
            # activation peak and re-creating the OOM the cap was
            # meant to prevent.
            try:
                from utils.hardware_manager import hardware_manager  # local

                hardware_manager.release_vram()
            except Exception:  # noqa: BLE001
                pass

        try:
            # XPart returns (obj_mesh, (out_bbox, mesh_gt_bbox, explode_object)).
            if capped_aabb is not None:
                kwargs["aabb"] = capped_aabb
            obj_mesh, bbox_tuple = self._impl(  # type: ignore[misc]
                mesh_path=input_path,
                octree_resolution=octree_resolution,
                output_type="trimesh",
                **kwargs,
            )
            out_bbox, mesh_gt_bbox, explode_object = bbox_tuple
        finally:
            # Best-effort clean up of the temp input.  Keep it on
            # exception so the user can inspect what was sent.
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
            except OSError:
                pass

        # Persist whichever outputs XPart actually produced.  When
        # the diffusion VAE-decode + marching cubes fails on every
        # part (e.g. SDF doesn't cross ``mc_level``, common on
        # low-quality input meshes), ``obj_mesh`` is an empty Scene
        # and trimesh raises ValueError("Can't export empty scenes!").
        # Skip those export failures individually so the response
        # still surfaces what *did* succeed (typically the bbox /
        # gt_bbox views are non-empty even when the decomposed
        # mesh isn't).
        mesh_path = os.path.join(_OUTPUT_DIR, f"{gen_id}_decomposed.glb")
        exploded_path = os.path.join(_OUTPUT_DIR, f"{gen_id}_exploded.glb")
        bbox_path = os.path.join(_OUTPUT_DIR, f"{gen_id}_bbox.glb")
        gt_bbox_path = os.path.join(_OUTPUT_DIR, f"{gen_id}_gt_bbox.glb")

        def _try_export(scene: Any, path: str, label: str) -> Optional[str]:
            try:
                scene.export(path)
                return path
            except Exception as e:  # noqa: BLE001
                self._logger.warning(
                    f"XPart produced no usable {label} mesh for this "
                    f"input ({e}); skipping export"
                )
                return None

        mesh_path = _try_export(obj_mesh, mesh_path, "decomposed")
        exploded_path = _try_export(explode_object, exploded_path, "exploded")
        bbox_path = _try_export(out_bbox, bbox_path, "bbox")
        gt_bbox_path = _try_export(mesh_gt_bbox, gt_bbox_path, "gt_bbox")

        # Optional per-part split.  XPart's ``obj_mesh`` is a
        # ``trimesh.Scene`` with one Trimesh per detected part.
        # When ``split=true`` is in the payload, write each geometry
        # to its own ``<id>_part_NN.glb`` so the caller can import
        # parts individually in Blender / three.js / Unity without
        # an extra round-trip through trimesh.
        part_paths: List[str] = []
        if payload.get("split") and obj_mesh is not None:
            try:
                import trimesh  # noqa: WPS433 — local import keeps cold start fast
                if isinstance(obj_mesh, trimesh.Scene):
                    geometries = list(obj_mesh.geometry.values())
                else:
                    geometries = [obj_mesh]
                for idx, geom in enumerate(geometries):
                    part_path = os.path.join(
                        _OUTPUT_DIR, f"{gen_id}_part_{idx:02d}.glb"
                    )
                    try:
                        geom.export(part_path)
                        part_paths.append(part_path)
                    except Exception as e:  # noqa: BLE001
                        self._logger.warning(
                            f"Failed to export part {idx} of split "
                            f"({e}); skipping"
                        )
            except Exception as e:  # noqa: BLE001
                self._logger.warning(
                    f"Split failed ({e}); only the assembled "
                    f"decomposed.glb will be available"
                )

        # If literally every output failed, the run is unusable —
        # surface a clean error.  Otherwise return whatever did work.
        if not any((mesh_path, exploded_path, bbox_path, gt_bbox_path)):
            raise RuntimeError(
                "Hunyuan3D-Part produced no exportable geometry for "
                "this mesh.  XPart's VAE-decode + marching-cubes step "
                "failed on every diffusion-produced part — typically "
                "because the input mesh's surface complexity / scale "
                "doesn't match what XPart was trained on (it prefers "
                "AI-generated or scanned meshes from Hunyuan3D V2.5 "
                "or V3.0).  Try a different input mesh or check the "
                "runner logs for per-part export errors."
            )

        return {
            "id": gen_id,
            "elapsed_sec": round(time.perf_counter() - started, 2),
            "mesh_path": mesh_path,
            "exploded_path": exploded_path,
            "bbox_path": bbox_path,
            "gt_bbox_path": gt_bbox_path,
            "part_paths": part_paths,
        }

    async def unload(self) -> None:
        if self._impl is not None:
            try:
                self._impl = None
                try:
                    import torch  # type: ignore[import-not-found]

                    torch.cuda.empty_cache()
                except Exception:
                    pass
            finally:
                await super().unload()
