from __future__ import annotations

import sys
import warnings
from collections.abc import Iterable
from operator import attrgetter, methodcaller
from typing import TYPE_CHECKING, Any

import numpy as np

from manim.utils.hashing import get_hash_from_play_call

from .. import config, logger
from .._config.video_encoder import video_encoder_fingerprint
from ..camera.camera import Camera
from ..mobject.mobject import Mobject, _AnimationBuilder
from ..mobject.types.image_mobject import AbstractImageMobject
from ..mobject.types.point_cloud_mobject import PMobject
from ..mobject.types.vectorized_mobject import VMobject
from ..scene.scene_file_writer import SceneFileWriter
from ..utils.exceptions import EndSceneEarlyException
from ..utils.family import extract_mobject_family_members
from ..utils.iterables import list_update
from .protocol import RendererCapabilities

if TYPE_CHECKING:
    from manim._config.render_session import RenderSessionSpec
    from manim.animation.animation import Animation
    from manim.scene.scene import Scene
    from manim.scene.scene_file_writer import _SceneFileWriterSettings

    from ..typing import PixelArray

__all__ = ["CairoRenderer"]


class CairoRenderer:
    """A renderer using Cairo.

    Attributes
    ----------
    num_plays : int
        Number of play() functions in the scene.

    time : float
        Time elapsed since initialisation of scene.
    """

    capabilities = RendererCapabilities(live_preview=False)

    def __init__(
        self,
        file_writer_class: type[SceneFileWriter] = SceneFileWriter,
        camera_class: type[Camera] | None = None,
        skip_animations: bool = False,
        **kwargs: Any,
    ):
        # All of the following are set to EITHER the value passed via kwargs,
        # OR the value stored in the global config dict at the time of
        # _instance construction_.
        self._file_writer_class = file_writer_class
        camera_cls = camera_class if camera_class is not None else Camera
        self.camera = camera_cls()
        self._original_skipping_status = skip_animations
        self.skip_animations = skip_animations
        self.animations_hashes: list[str | None] = []
        self.num_plays = 0
        self.time = 0.0
        self.static_image: PixelArray | None = None
        self._static_members: set[Mobject] = set()
        self._static_watch: _StaticFrameWatch | None = None
        self._resplits = 0

    def init_scene(
        self,
        scene: Scene,
        session_spec: RenderSessionSpec,
        file_writer_settings: _SceneFileWriterSettings,
    ) -> None:
        self.file_writer: Any = self._file_writer_class(file_writer_settings)

    def play(
        self,
        scene: Scene,
        *args: Animation | Mobject | _AnimationBuilder,
        **kwargs: Any,
    ) -> None:
        # Reset skip_animations to the original state.
        # Needed when rendering only some animations, and skipping others.
        self.skip_animations = self._original_skipping_status
        self.update_skipping_status()

        scene.compile_animation_data(*args, **kwargs)

        if self.skip_animations:
            logger.debug(f"Skipping animation {self.num_plays}")
            hash_current_animation = None
            self.time += scene.duration
        else:
            if config["disable_caching"]:
                logger.info("Caching disabled.")
                hash_current_animation = f"uncached_{self.num_plays:05}"
            else:
                assert scene.animations is not None
                hash_current_animation = get_hash_from_play_call(
                    scene,
                    self.camera,
                    scene.animations,
                    scene.mobjects,
                    backend="cairo",
                    encoder_fingerprint=video_encoder_fingerprint(
                        scene.session_spec.video_encoder,
                    ),
                    renderer_state=(),
                )
                if self.file_writer.is_already_cached(hash_current_animation):
                    logger.info(
                        f"Animation {self.num_plays} : Using cached data (hash : %(hash_current_animation)s)",
                        {"hash_current_animation": hash_current_animation},
                    )
                    self.skip_animations = True
                    self.time += scene.duration
        # adding None as a partial movie file will make file_writer ignore the latter.
        self.file_writer.add_partial_movie_file(hash_current_animation)
        self.animations_hashes.append(hash_current_animation)
        logger.debug(
            "List of the first few animation hashes of the scene: %(h)s",
            {"h": str(self.animations_hashes[:5])},
        )

        self.file_writer.begin_animation(
            not self.skip_animations,
            animation_index=self.num_plays,
        )
        scene.begin_animations()

        # Save a static image, to avoid rendering non moving objects.
        self.save_static_frame_data(scene, scene.static_mobjects)

        if scene.is_current_animation_frozen_frame():
            self._update_play_frame(scene)
            # self.duration stands for the total run time of all the animations.
            # In this case, as there is only a wait, it will be the length of the wait.
            self.freeze_current_frame(scene.duration)
            if scene.render_finished_frame:
                self.add_frame(self.get_frame())
        else:
            self._resplits = 0
            if not self.skip_animations:
                self._static_watch = _StaticFrameWatch.for_play(scene)
            try:
                scene.play_internal()
            finally:
                if self._static_watch is not None:
                    self._static_watch.close()
                    self._static_watch = None
        self.file_writer.end_animation(not self.skip_animations)

        self.num_plays += 1

    def update_frame(  # TODO Description in Docstring
        self,
        scene: Scene,
        mobjects: Iterable[Mobject] | None = None,
        include_submobjects: bool = True,
        ignore_skipping: bool = True,
        **kwargs: Any,
    ) -> None:
        """Update the frame.

        Parameters
        ----------
        scene

        mobjects
            The mobjects to draw, over the static frame if there is one.
            ``None`` draws the whole scene.

        include_submobjects

        ignore_skipping

        **kwargs
        """
        if self.skip_animations and not ignore_skipping:
            return
        if mobjects is None:
            mobjects = list_update(
                scene.mobjects,
                scene.foreground_mobjects,
            )
        if self.static_image is not None:
            self.camera.set_frame_to_background(self.static_image)
        else:
            self.camera.reset()

        kwargs["include_submobjects"] = include_submobjects
        self.camera.capture_mobjects(mobjects, **kwargs)

    def render(
        self,
        scene: Scene,
        time: float,
        moving_mobjects: Iterable[Mobject] | None = None,
    ) -> None:
        # Each frame is drawn from the scene as it is, so moving_mobjects is not needed.
        if self._static_watch is not None:
            self._refresh_static_frame(scene)
        self._update_play_frame(scene)
        self.add_frame(self.get_frame())

    def _update_play_frame(self, scene: Scene) -> None:
        """Draw a frame of a play: the static frame, then every mobject it
        does not hold, in the scene's draw order as it is now, so that
        mobjects added during the play are drawn too. A mobject sorted
        before one the static frame holds splits the scene again.
        """
        order = self._draw_order(scene)
        if self.static_image is not None and not self._follows_static_frame(order):
            self._resplit(scene)
            order = self._draw_order(scene)
        if self.static_image is not None:
            order = [m for m in order if m not in self._static_members]
        self.update_frame(scene, mobjects=order, include_submobjects=False)

    def _draw_order(self, scene: Scene) -> list[Mobject]:
        return self.camera.get_mobjects_to_display(
            list_update(scene.mobjects, scene.foreground_mobjects)
        )

    def _follows_static_frame(self, order: list[Mobject]) -> bool:
        """Whether everything the static frame holds comes first."""
        drawing = False
        for mobject in order:
            if mobject not in self._static_members:
                drawing = True
            elif drawing:
                return False
        return True

    def _refresh_static_frame(self, scene: Scene) -> None:
        """Split the scene again if something painted into the static
        frame changed since.
        """
        watch = self._static_watch
        assert watch is not None
        changed = watch.changed()
        if changed is None:
            return
        if not changed:
            moving = watch.resettle()
            if moving is not None:
                scene.moving_mobjects = moving
                return
        self._resplit(scene, changed)

    def _resplit(self, scene: Scene, changed: Iterable[Mobject] = ()) -> None:
        """Split the scene into static and moving mobjects again, the
        changed ones moving, and repaint the static frame if it changed.
        """
        assert scene.animations is not None
        self._resplits += 1
        if self._resplits > _MAX_RESPLITS:
            changed = scene.get_mobject_family_members()
        watch = self._static_watch
        if watch is not None:
            watch.close()
        scene._changed_static_mobjects.update(changed)
        moving, static = scene.get_moving_and_static_mobjects(scene.animations)
        if static != scene.static_mobjects:
            self.save_static_frame_data(scene, static)
        scene.moving_mobjects, scene.static_mobjects = moving, static
        if watch is not None:
            self._static_watch = _StaticFrameWatch(scene)

    def get_frame(self) -> PixelArray:
        """Gets the current frame as NumPy array.

        Returns
        -------
        PixelArray
            NumPy array of pixel values of each pixel in screen.
            The shape of the array is height x width x 3.
        """
        return self.camera.pixel_array.copy()

    def add_frame(self, frame: PixelArray, num_frames: int = 1) -> None:
        """Adds a frame to the video_file_stream

        Parameters
        ----------
        frame
            The frame to add, as a pixel array.
        num_frames
            The number of times to add frame.
        """
        dt = 1 / self.camera.frame_rate
        if self.skip_animations:
            return
        self.time += num_frames * dt
        self.file_writer.write_frame(frame, repeat=num_frames)

    def freeze_current_frame(self, duration: float) -> None:
        """Adds a static frame to the movie for a given duration. The static frame is the current frame.

        Parameters
        ----------
        duration
            [description]
        """
        dt = 1 / self.camera.frame_rate
        self.add_frame(
            self.get_frame(),
            num_frames=int(duration / dt),
        )

    def show_frame(self, scene: Scene) -> None:
        """Opens the current frame in the Default Image Viewer
        of your system.
        """
        self.update_frame(scene, ignore_skipping=True)
        self.camera.get_image().show()

    def save_static_frame_data(
        self,
        scene: Scene,
        static_mobjects: Iterable[Mobject],
    ) -> PixelArray | None:
        """Compute and save the static frame, that will be reused at each frame
        to avoid unnecessarily computing static mobjects.

        Parameters
        ----------
        scene
            The scene played.
        static_mobjects
            Static mobjects of the scene. If None, self.static_image is set to None.

        Returns
        -------
        PixelArray | None
            The static image computed. The return value is None if there are no static mobjects in the scene.
        """
        self.static_image = None
        self._static_members = set(static_mobjects or ())
        if not static_mobjects:
            return None
        # Already each mobject of the family, in draw order.
        self.update_frame(scene, mobjects=static_mobjects, include_submobjects=False)
        self.static_image = self.get_frame()
        return self.static_image

    def update_skipping_status(self) -> None:
        """This method is used internally to check if the current
        animation needs to be skipped or not. It also checks if
        the number of animations that were played correspond to
        the number of animations that need to be played, and
        raises an EndSceneEarlyException if they don't correspond.
        """
        # there is always at least one section -> no out of bounds here
        if self.file_writer.sections[-1].skip_animations:
            self.skip_animations = True
        if self.file_writer.output_spec.is_still:
            self.skip_animations = True
        if (
            config.from_animation_number > 0
            and self.num_plays < config.from_animation_number
        ):
            self.skip_animations = True
        if (
            config.upto_animation_number >= 0
            and self.num_plays > config.upto_animation_number
        ):
            self.skip_animations = True
            raise EndSceneEarlyException()

    def scene_finished(self, scene: Scene) -> None:
        output = self.file_writer.output_spec
        if self.num_plays and (output.is_video or output.is_image_sequence):
            self.file_writer.finish()
        elif not self.num_plays:
            self.static_image = None
            self.update_frame(scene)

        # Automatically selected video output falls back to a last-frame PNG
        # when a scene has no play calls.
        if output.is_still or (not self.num_plays and output.fallback_to_still):
            if self.num_plays:
                self.static_image = None
                self.update_frame(scene)
            self.file_writer.save_image(self.get_frame())


_tobytes = methodcaller("tobytes")

# What the camera draws of each kind of mobject: its arrays, and its plain
# values.
_DRAWN: list[tuple[type[Mobject], tuple[str, ...], tuple[str, ...]]] = [
    (
        VMobject,
        ("points", "fill_rgbas", "stroke_rgbas", "background_stroke_rgbas"),
        ("stroke_width", "background_stroke_width", "joint_type", "cap_style"),
    ),
    (PMobject, ("points", "rgbas"), ("stroke_width",)),
    (AbstractImageMobject, ("points",), ("resampling_algorithm",)),
    (Mobject, ("points",), ()),
]

# Resplitting more often than this in one play gives up on the static frame.
_MAX_RESPLITS = 3


def _references(namespace: dict[str, Any], name: str) -> int:
    return sys.getrefcount(namespace[name])


# The count :func:`_references` gives for a value held by its namespace alone.
_ALONE = _references({"value": object()}, "value")


class _Kind:
    """The watched mobjects of one kind in :data:`_DRAWN`, and a snapshot of
    their structure, their values and which arrays they hold.
    """

    def __init__(
        self, mobjects: list[Mobject], arrays: tuple[str, ...], values: tuple[str, ...]
    ) -> None:
        self.arrays = arrays
        self.get = attrgetter("submobjects", "z_index", *values, *arrays)
        self.mobjects = [m for m in mobjects if self._readable(m)]
        self.state: list[tuple[Any, ...]] = []

    def _readable(self, mobject: Mobject) -> bool:
        try:
            _, _, *rest = self.get(mobject)
            values = rest[: len(rest) - len(self.arrays)]
            arrays = rest[len(values) :]
            return all(isinstance(a, np.ndarray) for a in arrays) and bool(
                values == values
            )
        except Exception:
            return False

    def snapshot(self) -> None:
        self.state = [(list(sub), *rest) for sub, *rest in map(self.get, self.mobjects)]

    def changed(self) -> list[Mobject]:
        try:
            # Arrays compare as themselves: identical, or not at all.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if list(map(self.get, self.mobjects)) == self.state:
                    return []
        except ValueError:
            pass
        return [
            m
            for m, then in zip(self.mobjects, self.state, strict=True)
            if not all(map(_same, self.get(m), then))
        ]


def _same(now: Any, then: Any) -> bool:
    if isinstance(now, np.ndarray) or isinstance(then, np.ndarray):
        return now is then
    return bool(now == then)


class _Pool:
    """Arrays moved into one buffer, each attribute left holding a view of
    its part, so a single comparison sees a write to any of them.

    Only arrays nothing else refers to are moved, so no reference to the
    old array is left behind, and :meth:`release` gives them back their
    own memory when nothing else refers to the view either.
    """

    def __init__(self, entries: list[tuple[Mobject, str]]) -> None:
        self.entries = entries
        arrays = [m.__dict__[name] for m, name in entries]
        sizes = np.array([a.size for a in arrays], dtype=np.intp)
        self.ends = np.cumsum(sizes)
        self.data = (
            np.concatenate([a.ravel() for a in arrays]) if arrays else np.empty(0)
        )
        self.saved = self.data.copy()
        self.views = [
            self.data[end - size : end].reshape(a.shape)
            for a, size, end in zip(arrays, sizes, self.ends, strict=True)
        ]
        del arrays
        for (m, name), view in zip(entries, self.views, strict=True):
            m.__dict__[name] = view

    @staticmethod
    def takes(mobject: Mobject, name: str) -> bool:
        namespace = mobject.__dict__
        if name not in namespace or _references(namespace, name) != _ALONE:
            return False
        a = namespace[name]
        return (
            type(a) is np.ndarray
            and a.base is None
            and a.dtype == np.float64
            and a.flags.c_contiguous
            and a.flags.writeable
        )

    def changed(self) -> list[Mobject]:
        # Bitwise, so that a NaN left as it was is no change.
        differ = self.data.view(np.int64) != self.saved.view(np.int64)
        if not differ.any():
            return []
        parts = np.unique(
            np.searchsorted(self.ends, np.flatnonzero(differ), side="right")
        )
        return [self.entries[i][0] for i in parts]

    def release(self) -> None:
        views, self.views = self.views, []
        for i, (m, name) in enumerate(self.entries):
            namespace = m.__dict__
            if (
                namespace.get(name) is views[i]
                and _references(namespace, name) == _ALONE + 1  # and views
            ):
                namespace[name] = namespace[name].copy()


class _StaticFrameWatch:
    """Watches, during a play, the mobjects painted once into the static frame.

    They are the mobjects before the first one that is animated or has
    updaters, so nothing is expected to change them. Yet code that runs
    every frame, like an updater, can change any mobject, not only its
    own, and add or remove mobjects. :meth:`changed` notices, comparing
    what the camera draws of each against a snapshot. The camera's own
    mobjects are watched too: moving the camera moves everything.
    """

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.roots = _roots(scene)
        moving = set(scene.moving_mobjects)
        self.members = [
            m for m in extract_mobject_family_members(self.roots) if m not in moving
        ]
        members = set(self.members)
        self.static_roots = [
            r for r in self.roots if members.intersection(r.get_family())
        ]
        # Static roots wholly static and first: then adding or removing
        # others leaves the static frame as it is.
        first = self.roots[: len(self.static_roots)] == self.static_roots
        self.clean = first and not any(
            moving.intersection(r.get_family()) for r in self.static_roots
        )
        camera = scene.renderer.camera
        assert isinstance(camera, Camera)
        self.camera_mobjects = [
            m for m in _camera_mobjects(camera) if self.members and m not in moving
        ]
        # Each mobject goes to the first kind it is and whose snapshot can
        # hold it; the last kind, points only, takes nearly anything.
        unkind = list(dict.fromkeys([*self.members, *self.camera_mobjects]))
        self.kinds = []
        for t, arrays, values in _DRAWN:
            kind = _Kind([m for m in unkind if isinstance(m, t)], arrays, values)
            taken = set(kind.mobjects)
            unkind = [m for m in unkind if m not in taken]
            self.kinds.append(kind)
        entries = [
            (m, name) for k in self.kinds for m in k.mobjects for name in k.arrays
        ]
        pooled: list[tuple[Mobject, str]] = []
        self.loose: list[tuple[Mobject, str]] = []
        for entry in entries:
            (pooled if _Pool.takes(*entry) else self.loose).append(entry)
        self.pool = _Pool(pooled)
        self.loose_bytes = self._loose_bytes()
        for kind in self.kinds:
            kind.snapshot()
        self.images = [
            (m, pixels.copy())
            for m in self.members
            if isinstance(pixels := vars(m).get("pixel_array"), np.ndarray)
        ]

    @classmethod
    def for_play(cls, scene: Scene) -> _StaticFrameWatch | None:
        """A watch over the current play, or ``None`` if the play runs no
        code but manim's own animations, which only change their mobjects.
        """
        assert scene.animations is not None
        if (
            scene.updaters
            or any(m.updaters for m in scene.moving_mobjects)
            or any(map(_runs_foreign_code, scene.animations))
        ):
            return cls(scene)
        return None

    def close(self) -> None:
        for kind in self.kinds:
            kind.state = []
        self.pool.release()

    def changed(self) -> list[Mobject] | None:
        """The watched mobjects changed since the snapshot, all of them if
        the camera moved; an empty list if only which mobjects are in the
        scene changed; ``None`` if nothing did.
        """
        try:
            found = self._changed()
        except Exception:
            # Something is unreadable now: assume everything changed
            # rather than guess.
            found = self.members
        if any(m in found for m in self.camera_mobjects):
            return self.members
        if found:
            return found
        if _roots(self.scene) != self.roots:
            return []
        return None

    def resettle(self) -> list[Mobject] | None:
        """After mobjects were only added or removed: the new moving
        mobjects if the static frame still holds, else ``None``.
        """
        roots = _roots(self.scene)
        n = len(self.static_roots)
        if not self.clean or roots[:n] != self.static_roots:
            return None
        self.roots = roots
        return extract_mobject_family_members(
            roots[n:], use_z_index=self.scene.renderer.camera.use_z_index
        )

    def _changed(self) -> list[Mobject]:
        found = [m for kind in self.kinds for m in kind.changed()]
        found += self.pool.changed()
        if self._loose_bytes() != self.loose_bytes:
            found += [
                m
                for (m, name), then in zip(self.loose, self.loose_bytes, strict=True)
                if _tobytes(getattr(m, name)) != then
            ]
        found += [
            m
            for m, pixels in self.images
            if not np.array_equal(vars(m)["pixel_array"], pixels)
        ]
        return found

    def _loose_bytes(self) -> list[bytes]:
        return [_tobytes(getattr(m, name)) for m, name in self.loose]


def _roots(scene: Scene) -> list[Mobject]:
    return list_update(scene.mobjects, scene.foreground_mobjects)


def _camera_mobjects(camera: Camera) -> list[Mobject]:
    """The mobjects whose change moves the camera."""
    from ..camera.moving_camera import MovingCamera
    from ..camera.three_d_camera import ThreeDCamera

    if isinstance(camera, MovingCamera):
        return camera.get_mobjects_indicating_movement()
    if isinstance(camera, ThreeDCamera):
        return [*camera.get_value_trackers(), camera._frame_center]
    return []


def _runs_foreign_code(animation: Animation) -> bool:
    """Whether ``animation`` runs code other than manim's each frame, which
    may change mobjects other than the animated ones.
    """
    from ..animation.updaters.update import UpdateFromFunc

    if isinstance(animation, UpdateFromFunc):
        return True
    if not type(animation).__module__.startswith("manim."):
        return True
    inner = [*getattr(animation, "animations", []), getattr(animation, "anim", None)]
    return any(_runs_foreign_code(a) for a in inner if a is not None)
