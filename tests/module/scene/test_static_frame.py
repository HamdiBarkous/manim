"""Mobjects painted once into the static frame of a play must still show
what the code running each frame, like updaters, does to them, and every
frame must keep the draw order of the whole scene.
"""

from __future__ import annotations

import numpy as np
import pytest

from manim import (
    Circle,
    Create,
    Dot,
    Mobject,
    MovingCameraScene,
    Rectangle,
    Scene,
    Square,
    UpdateFromAlphaFunc,
    ValueTracker,
    VGroup,
    VMobject,
)
from manim.constants import DOWN, LEFT, RIGHT, UP
from manim.renderer.cairo_renderer import CairoRenderer, _StaticFrameWatch


@pytest.fixture
def render(config, tmp_path, monkeypatch):
    """Render a scene, checking that every frame is what drawing the whole
    scene afresh gives.
    """
    config.media_dir = str(tmp_path)
    config.pixel_width, config.pixel_height, config.frame_rate = 160, 90, 15
    config.disable_caching = True
    config.write_to_movie = True
    add_frame = CairoRenderer.add_frame

    def checked(self, frame, num_frames=1):
        static_image, self.static_image = self.static_image, None
        self.update_frame(self.scene)
        np.testing.assert_array_equal(frame, self.get_frame())
        self.static_image = static_image
        add_frame(self, frame, num_frames)

    monkeypatch.setattr(CairoRenderer, "add_frame", checked)

    def render(scene_class):
        scene = scene_class()
        scene.renderer.scene = scene
        scene.render()
        return scene

    return render


class OtherMobjectsUpdater(Scene):
    def construct(self):
        square = Square()
        self.add(square)
        self.add(Mobject().add_updater(lambda m, dt: square.rotate(dt)))
        self.wait(1)


class SceneUpdater(Scene):
    def construct(self):
        square = Square(fill_opacity=0.5)
        self.add(square)
        self.add_updater(lambda dt: square.shift(RIGHT * dt))
        self.wait(1)


class FunctionAnimation(Scene):
    def construct(self):
        square = Square()
        self.add(square)
        self.play(UpdateFromAlphaFunc(Mobject(), lambda m, a: square.set_x(a)))


class Recolor(Scene):
    def construct(self):
        square = Square(fill_opacity=1)
        self.add(square)
        tracker = ValueTracker(0).add_updater(lambda m, dt: m.increment_value(dt))
        tracker.add_updater(lambda m: square.set_fill(opacity=1 - m.get_value()))
        self.add(tracker)
        self.wait(1)


class PathDrawnByOther(Scene):
    def construct(self):
        path = VMobject()
        self.add(path)

        def move(dot, dt):
            dot.shift(RIGHT * dt)
            if path.has_points():
                path.add_points_as_corners([dot.get_center()])
            else:
                path.set_points_as_corners([dot.get_center(), dot.get_center()])

        self.add(Dot().add_updater(move))
        self.wait(1)


class GroupGrows(Scene):
    def construct(self):
        group = VGroup(Dot())
        self.add(group)

        def grow(m, dt):
            group.add(Dot(RIGHT * len(group) * 0.5))

        self.add(Mobject().add_updater(grow))
        self.wait(1)


class RemovedByUpdater(Scene):
    def construct(self):
        static, moving = Circle(), Square()
        self.add(static)

        def remove(m, dt):
            if self.renderer.time > 0.5:
                self.remove(static, m)

        self.add(moving.add_updater(remove))
        self.wait(1)


class CameraFollows(MovingCameraScene):
    def construct(self):
        self.add(Square())
        dot = Dot()

        def follow(d, dt):
            d.shift(UP * dt)
            self.camera.frame.move_to(d)

        self.add(dot.add_updater(follow))
        self.wait(1)


@pytest.mark.parametrize(
    "scene_class",
    [
        OtherMobjectsUpdater,
        SceneUpdater,
        FunctionAnimation,
        Recolor,
        PathDrawnByOther,
        GroupGrows,
        RemovedByUpdater,
        CameraFollows,
    ],
)
def test_static_mobjects_changed_during_a_play_are_drawn(render, scene_class):
    render(scene_class)


def test_a_frame_rendered_without_moving_mobjects_draws_everything(render):
    class FinishedFrame(Scene):
        """Renders the finished state as one more frame of the play, passing
        no moving mobjects: the whole scene.
        """

        def play_internal(self, skip_rendering=False):
            super().play_internal(skip_rendering)
            self.renderer.render(self, self.duration)

        def construct(self):
            self.add(Square(fill_opacity=1))
            self.add(Dot().add_updater(lambda m, dt: m.shift(RIGHT * dt)))
            self.wait(0.2)

    render(FinishedFrame)


def test_the_static_frame_is_repainted_once(render, monkeypatch):
    paints = []
    save = CairoRenderer.save_static_frame_data

    def counted(self, scene, static_mobjects):
        paints.append(len(static_mobjects))
        return save(self, scene, static_mobjects)

    monkeypatch.setattr(CairoRenderer, "save_static_frame_data", counted)
    render(OtherMobjectsUpdater)
    assert len(paints) == 2


def test_nothing_is_watched_when_only_manims_animations_run(render, monkeypatch):
    watches = []
    monkeypatch.setattr(
        _StaticFrameWatch, "__init__", lambda self, scene: watches.append(scene)
    )

    class Plain(Scene):
        def construct(self):
            self.add(Square())
            self.play(Create(Circle()))

    render(Plain)
    assert watches == []


def test_arrays_are_given_back_after_the_play(render):
    class Kept(Scene):
        def construct(self):
            self.square, self.circle = Square(), Circle()
            self.add(self.square, self.circle)
            self.kept = []

            def keep(m, dt):
                self.kept.append(self.circle.points)

            self.add(Mobject().add_updater(keep))
            self.wait(0.2)

    scene = render(Kept)
    for name in ("points", "fill_rgbas", "stroke_rgbas", "background_stroke_rgbas"):
        assert getattr(scene.square, name).base is None
    # A reference taken during the play still is the circle's.
    assert scene.circle.points is scene.kept[0]


def hill():
    return Rectangle(width=15, height=4, fill_opacity=1, stroke_width=0).shift(DOWN * 2)


def tree():
    return (
        Circle(fill_opacity=1, stroke_width=0).shift(LEFT * 4 + DOWN).set_z_index(-2.5)
    )


def spinning():
    return Square().shift(RIGHT * 4 + UP).add_updater(lambda m, dt: m.rotate(dt))


class LowChildOfAGroupAfterTheMovingOne(Scene):
    def construct(self):
        hills = VGroup(hill())
        hills[0].set_z_index(-3)
        self.add(spinning(), hills, tree())
        self.wait(0.5)


class GroupUpdaterMovesItsLowChild(Scene):
    def construct(self):
        hills = VGroup(hill())
        hills[0].set_z_index(-3)
        hills.add_updater(lambda m, dt: m.shift(UP * dt))
        self.add(spinning(), hills, tree())
        self.wait(0.5)


class LowChildOfAForegroundGroup(Scene):
    def construct(self):
        hills = VGroup(hill())
        hills[0].set_z_index(-3)
        self.add(spinning(), tree())
        self.add_foreground_mobject(hills)
        self.wait(0.5)


class LowChildOfANestedGroup(Scene):
    def construct(self):
        outer = VGroup(VGroup(hill().set_z_index(-3)), Square().shift(UP * 3))
        self.add(spinning(), outer, tree())
        self.wait(0.3)
        outer.add_updater(lambda m, dt: m.shift(RIGHT * dt))
        self.wait(0.3)


class LowChildSharedByTwoGroups(Scene):
    def construct(self):
        low = hill().set_z_index(-3)
        self.add(spinning(), VGroup(low), tree(), VGroup(low, Square().shift(UP * 3)))
        self.wait(0.5)


class LowMobjectAddedDuringAPlay(Scene):
    def construct(self):
        low = hill().set_z_index(-3)

        def add_once(m, dt):
            if self.renderer.time > 0.2 and low not in self.mobjects:
                self.add(low)

        self.add(tree(), spinning().add_updater(add_once))
        self.wait(0.5)


@pytest.mark.parametrize(
    "scene_class",
    [
        LowChildOfAGroupAfterTheMovingOne,
        GroupUpdaterMovesItsLowChild,
        LowChildOfAForegroundGroup,
        LowChildOfANestedGroup,
        LowChildSharedByTwoGroups,
        LowMobjectAddedDuringAPlay,
    ],
)
def test_frames_keep_the_draw_order(render, scene_class):
    render(scene_class)
