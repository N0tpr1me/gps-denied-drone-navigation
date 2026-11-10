"""The same scene and stereo rig rendered by Isaac Sim's RTX renderer (GPU server only).

What Isaac adds over the NumPy renderer (``raster.py``): real shading (shadows, inter-
reflection, the lights' falloff), the RTX texture filtering, and a renderer a buyer
recognises. What stays the same: the scene (``scene.py``: the quads and their textures, here
written to PNG and laid on USD meshes), the cameras (``camera.py``: the same pinhole
intrinsics and poses), and everything after the clean frame (``degrade.py``: fog, dust,
light level and the camera model), so a difference between the two renderers' VIO results
comes from how the clean scene looks and nothing else.

Output per camera: linear radiance from the ``HdrColor`` annotator (before tone mapping),
reduced to luminance and scaled once so its median matches the NumPy renderer's on the
first frame (both are then in the same relative units, 1 = a white surface lit at
``light_level`` 1, which the light levels and the camera model rely on), and the range from
the ``distance_to_camera`` annotator. The first frame is also a geometry check: Isaac's
range map must match the NumPy renderer's (median difference under ``GEOMETRY_TOL_M``) or
the run stops, because a camera-convention mistake would otherwise show up as a bad VIO.

Untested here: the cloud container has no GPU and cannot install Isaac Sim (docs/
isaac-lab.md). Written against Isaac Sim 5.1 (``isaacsim``, ``omni.replicator.core``, USD
``pxr``), the version pinned for the server; ``eval/vio_render/isaac_render.sh`` runs it in
the pinned Isaac Lab image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eval.vio_render.camera import RigConfig
from eval.vio_render.euroc import png_gray
from eval.vio_render.raster import RasterRenderer
from eval.vio_render.scene import Quad, Scene

GEOMETRY_TOL_M = 0.02
H_APERTURE_MM = 20.955  # USD's default horizontal aperture; only the ratio to focal matters
_APP: Any = None  # one SimulationApp per process


def _app(headless: bool = True) -> Any:
    global _APP
    if _APP is None:
        from isaacsim import SimulationApp

        _APP = SimulationApp({"headless": headless, "renderer": "RaytracedLighting"})
    return _APP


def usd_camera_matrix(rot_wc: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """4x4 USD transform (row-vector convention) for an OpenCV camera pose.

    USD cameras look along their -z with +y up; OpenCV's look along +z with +y down, so the
    USD camera's axes are the OpenCV ones with y and z flipped. USD writes a transform with
    the axes as rows and the translation in the last row."""
    r_usd = rot_wc @ np.diag([1.0, -1.0, -1.0])
    m = np.eye(4)
    m[:3, :3] = r_usd.T
    m[3, :3] = centre
    return m


def light_matrix(light_dir: np.ndarray) -> np.ndarray:
    """USD transform of a DistantLight that shines along ``light_dir`` (it emits along its
    own -z)."""
    z = -np.asarray(light_dir, dtype=float)
    z /= np.linalg.norm(z)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, :3] = np.stack([x, y, z])
    return m


class IsaacRenderer:
    """``render_pair(poses)`` renders both cameras in one RTX step."""

    def __init__(self, scene: Scene, rig: RigConfig, work_dir: Path, rt_subframes: int = 8):
        self.scene = scene
        self.rig = rig
        self.work = Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.rt_subframes = rt_subframes
        self.reference = RasterRenderer(scene, rig)
        self.scale: float | None = None
        _app()
        import carb
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

        self._rep, self._gf = rep, Gf
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        settings = carb.settings.get_settings()
        # No auto exposure, motion blur or temporal tricks: one clean linear frame per step.
        for key, value in (("/rtx/post/histogram/enabled", False),
                           ("/rtx/post/motionblur/enabled", False),
                           ("/rtx/post/dlss/execMode", 0),
                           ("/rtx/post/aa/op", 0)):
            settings.set(key, value)

        for name, tex in scene.textures.items():
            img = np.clip(np.rint(tex * 255.0), 0, 255).astype(np.uint8)
            (self.work / f"{name}.png").write_bytes(png_gray(img))
        for q in scene.quads:
            self._add_quad(stage, q, Gf, Sdf, UsdGeom, UsdShade)

        sun = UsdLux.DistantLight.Define(stage, "/World/sun")
        sun.CreateIntensityAttr(3000.0 * (1.0 - scene.ambient))
        sun.CreateAngleAttr(0.5)
        UsdGeom.Xformable(sun.GetPrim()).AddTransformOp().Set(
            Gf.Matrix4d(*light_matrix(np.asarray(scene.light_dir)).ravel().tolist()))
        dome = UsdLux.DomeLight.Define(stage, "/World/ambient")
        dome.CreateIntensityAttr(3000.0 * scene.ambient / np.pi)

        self._cams, self._ops, self._rgb, self._dist = [], [], [], []
        focal = rig.fx * H_APERTURE_MM / rig.width
        for i in (0, 1):
            path = f"/World/cam{i}"
            cam = UsdGeom.Camera.Define(stage, path)
            cam.CreateFocalLengthAttr(focal)
            cam.CreateHorizontalApertureAttr(H_APERTURE_MM)
            cam.CreateVerticalApertureAttr(H_APERTURE_MM * rig.height / rig.width)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
            self._ops.append(UsdGeom.Xformable(cam.GetPrim()).AddTransformOp())
            rp = rep.create.render_product(path, (rig.width, rig.height))
            rgb = rep.AnnotatorRegistry.get_annotator("HdrColor")
            dist = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
            rgb.attach(rp)
            dist.attach(rp)
            self._rgb.append(rgb)
            self._dist.append(dist)
        rep.orchestrator.set_capture_on_play(False)

    def _add_quad(self, stage: Any, q: Quad, Gf: Any, Sdf: Any, UsdGeom: Any,
                  UsdShade: Any) -> None:
        o, eu, ev = (np.asarray(v, dtype=float) for v in (q.origin, q.edge_u, q.edge_v))
        pts = [o, o + eu, o + eu + ev, o + ev]
        mesh = UsdGeom.Mesh.Define(stage, f"/World/{q.name}")
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateDoubleSidedAttr(True)
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                     UsdGeom.Tokens.vertex)
        st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
        mat = UsdShade.Material.Define(stage, f"/World/mat_{q.name}")
        shader = UsdShade.Shader.Define(stage, f"/World/mat_{q.name}/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        reader = UsdShade.Shader.Define(stage, f"/World/mat_{q.name}/st")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"/World/mat_{q.name}/tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(self.work / f"{q.texture}.png"))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb")
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

    def _step(self) -> None:
        try:
            self._rep.orchestrator.step(rt_subframes=self.rt_subframes, delta_time=0.0,
                                        pause_timeline=True)
        except TypeError:  # older signature
            self._rep.orchestrator.step(rt_subframes=self.rt_subframes)

    def render_pair(self, poses: list[tuple[np.ndarray, np.ndarray]],
                    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """[(rot_wc, centre)] for cam0 and cam1 -> [(radiance, range)] for each."""
        for op, (rot_wc, centre) in zip(self._ops, poses, strict=True):
            op.Set(self._gf.Matrix4d(*usd_camera_matrix(rot_wc, centre).ravel().tolist()))
        self._step()
        out = []
        for rgb_a, dist_a in zip(self._rgb, self._dist, strict=True):
            rgb = np.asarray(rgb_a.get_data(), dtype=np.float32)[..., :3]
            lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            rng = np.asarray(dist_a.get_data(), dtype=np.float32).reshape(lum.shape)
            rng = np.where(np.isfinite(rng) & (rng > 0) & (rng < 1e5), rng, np.inf)
            out.append((lum, rng.astype(np.float32)))
        if self.scale is None:
            self._calibrate(poses[0], *out[0])
        return [(lum * self.scale, rng) for lum, rng in out]

    def _calibrate(self, pose: tuple[np.ndarray, np.ndarray], lum: np.ndarray,
                   rng: np.ndarray) -> None:
        ref_rad, ref_rng = self.reference.render(*pose)
        both = np.isfinite(rng) & np.isfinite(ref_rng)
        if both.mean() < 0.5:
            raise RuntimeError("Isaac and the NumPy renderer disagree on what the first frame "
                               f"sees ({both.mean():.0%} of pixels hit in both)")
        geo = float(np.median(np.abs(rng[both] - ref_rng[both])))
        if geo > GEOMETRY_TOL_M:
            raise RuntimeError(f"Isaac's range map is {geo:.3f} m off the NumPy renderer's "
                               f"(limit {GEOMETRY_TOL_M} m): a camera pose or intrinsics "
                               "convention differs; fix before trusting any VIO result")
        self.scale = float(np.median(ref_rad[both]) / max(np.median(lum[both]), 1e-9))
        (self.work / "calibration.txt").write_text(
            f"geometry median |range diff| {geo:.4f} m\nradiance scale {self.scale:.6g}\n")

    def render(self, rot_wc: np.ndarray, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One camera (both are rendered; the second is thrown away)."""
        return self.render_pair([(rot_wc, centre), (rot_wc, centre)])[0]
