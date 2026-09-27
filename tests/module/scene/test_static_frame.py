"""Mobjects painted once into the static frame of a play must still show
what the code running each frame, like updaters, does to them, and every
frame must keep the draw order of the whole scene.
"""

from __future__ import annotations

import numpy as np
import pytest

from manim import (
    BLUE,
    RED,
    CapStyleType,
    Circle,
    Create,
    Dot,
    ImageMobject,
    Line,
    ManimColor,
    Mobject,
    MovingCameraScene,
    Polygon,
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


@pytest.mark.parametrize("finished", [False, True])
def test_a_play_can_end_on_its_finished_state(render, monkeypatch, finished):
    """Three plays of a second each, at 15 frames per second: an animation,
    a wait with nothing moving and a wait with an updater. Each can end
    with one more frame, of its finished state.
    """
    frames = []
    add_frame = CairoRenderer.add_frame

    def counted(self, frame, num_frames=1):
        frames.extend([frame] * num_frames)
        add_frame(self, frame, num_frames)

    monkeypatch.setattr(CairoRenderer, "add_frame", counted)

    class ThreePlays(Scene):
        def __init__(self):
            super().__init__(render_finished_frame=finished)

        def construct(self):
            self.play(Square().animate.shift(RIGHT), run_time=1)
            self.wait(1)
            self.add(Dot().add_updater(lambda m, dt: m.shift(UP * dt)))
            self.wait(1)

    render(ThreePlays)
    assert len(frames) == 3 * (15 + finished)


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


def circles(n=40, fill=0.6):
    """Still circles on the left half."""
    return VGroup(
        *[
            Circle(0.18, fill_opacity=fill, stroke_width=4).move_to([x, y, 0])
            for x in np.arange(-6.5, -0.4, 0.45)
            for y in np.arange(-3.4, 3.5, 0.5)
        ][:n]
    )


def wheel(hub=RIGHT * 3.5):
    spokes = VGroup(
        *[
            Line(hub, hub + 1.5 * np.array([np.cos(a), np.sin(a), 0]))
            for a in np.arange(0, 2 * np.pi, np.pi / 6)
        ]
    )
    spokes.add_updater(lambda m, dt: m.rotate(-2 * dt, about_point=hub))
    return VGroup(Circle(1.5).move_to(hub), spokes)


class StillInFrontOfTheMovingOne(Scene):
    def construct(self):
        self.add(wheel(), circles(), Line(RIGHT * 2, RIGHT * 5, stroke_width=8))
        self.wait(0.5)


class MovingOnBothSides(Scene):
    def construct(self):
        dot = Dot(LEFT * 3).add_updater(lambda m, dt: m.shift(RIGHT * 2 * dt))
        squares = VGroup(
            *[Square(0.5, fill_opacity=0.8).move_to([x, 0, 0]) for x in range(-4, 4)]
        )
        self.add(wheel(), circles(20), dot, squares)
        self.wait(0.5)


class StillChangedByAnotherUpdater(Scene):
    def construct(self):
        still = circles(20)
        w = wheel()
        w.add_updater(lambda m, dt: still[5].shift(UP * dt))
        self.add(w, still)
        self.wait(0.5)


class TranslucentStillStacked(Scene):
    def construct(self):
        halo = VGroup(
            *[
                Circle(0.65 + 0.18 * k, stroke_width=0, fill_opacity=0.1).shift(
                    RIGHT * 3.5
                )
                for k in range(6)
            ]
        )
        self.add(
            Rectangle(width=15, height=9, fill_opacity=1, stroke_width=0), wheel(), halo
        )
        self.wait(0.5)


class MovingRebuiltEveryFrame(Scene):
    def construct(self):
        clock = [0.0]

        def flap(m, dt):
            clock[0] += dt
            m.become(Square(fill_opacity=0.5).rotate(clock[0]).shift(RIGHT * 3))

        self.add(Square().add_updater(flap), circles(20))
        self.wait(0.5)


class ImageAndGradientInFront(Scene):
    def construct(self):
        pixels = np.uint8(
            [
                [[255, 0, 0, 255], [0, 255, 0, 128]],
                [[0, 0, 255, 200], [255, 255, 0, 60]],
            ]
        )
        image = ImageMobject(pixels).scale(2).shift(RIGHT * 3)
        gradient = (
            Square(2, fill_opacity=1)
            .set_color([RED, BLUE])
            .set_sheen(0.4, UP)
            .shift(RIGHT * 3)
        )
        self.add(wheel(), image, gradient)
        self.wait(0.5)


class ThickMiterStroke(Scene):
    def construct(self):
        spike = Polygon(
            [-0.05, -0.6, 0],
            [0.05, -0.6, 0],
            [0, 0.9, 0],
            stroke_width=24,
            fill_opacity=0.3,
        )
        spike.shift(RIGHT * 2).add_updater(lambda m, dt: m.rotate(1.3 * dt))
        self.add(spike, circles())
        self.wait(0.5)


class SquareCapsAndBackgroundStroke(Scene):
    def construct(self):
        line = (
            Line(LEFT, RIGHT, stroke_width=30)
            .set_cap_style(CapStyleType.SQUARE)
            .shift(RIGHT * 2)
        )
        line.set_stroke(width=40, background=True).add_updater(
            lambda m, dt: m.rotate(dt)
        )
        self.add(line, circles())
        self.wait(0.5)


class HalfOffScreen(Scene):
    def construct(self):
        disc = Circle(1.5, fill_opacity=0.7).move_to(RIGHT * 7 + UP * 3.5)
        self.add(disc.add_updater(lambda m, dt: m.shift(LEFT * dt)), circles())
        self.wait(0.5)


class MovingBehindTranslucent(Scene):
    def construct(self):
        dot = Dot(LEFT * 3, radius=0.3).add_updater(
            lambda m, dt: m.shift(RIGHT * 3 * dt)
        )
        cover = Rectangle(width=8, height=2, fill_opacity=0.35, stroke_width=6)
        self.add(circles(20), dot, cover)
        self.wait(0.5)


class MovingImageAndHuge(Scene):
    def construct(self):
        pixels = np.uint8(
            [
                [[255, 0, 0, 255], [0, 255, 0, 128]],
                [[0, 0, 255, 200], [255, 255, 0, 60]],
            ]
        )
        image = ImageMobject(pixels).scale(1.5).shift(RIGHT * 3)
        huge = Square(7.5, fill_opacity=0.2).add_updater(
            lambda m, dt: m.rotate(0.3 * dt)
        )
        self.add(
            image.add_updater(lambda m, dt: m.shift(DOWN * dt)),
            circles(20),
            huge,
            Square(0.5).shift(UP * 3),
        )
        self.wait(0.5)


class StillChangedBetweenPlays(Scene):
    def construct(self):
        still = circles(20)
        self.add(wheel(), still)
        self.wait(0.3)
        still[3].shift(UP * 0.4).set_color(RED)
        still[4].become(Square(0.3, fill_opacity=1).move_to(still[4]))
        self.wait(0.3)


class CameraBackgroundChanges(Scene):
    def construct(self):
        clock = [0.0]

        def dusk(m, dt):
            clock[0] += dt
            if 0.2 < clock[0] < 0.3:
                self.camera.background_color = ManimColor("#203040")
                self.camera.init_background()

        self.add(wheel(), circles(20), Mobject().add_updater(dusk))
        self.wait(0.5)


class CameraMovesOverStill(MovingCameraScene):
    def construct(self):
        self.add(wheel(), circles(20))
        self.camera.frame.add_updater(lambda m, dt: m.shift(RIGHT * 0.5 * dt))
        self.wait(0.5)


@pytest.mark.parametrize(
    "scene_class",
    [
        StillInFrontOfTheMovingOne,
        MovingOnBothSides,
        StillChangedByAnotherUpdater,
        TranslucentStillStacked,
        MovingRebuiltEveryFrame,
        ImageAndGradientInFront,
        ThickMiterStroke,
        SquareCapsAndBackgroundStroke,
        HalfOffScreen,
        MovingBehindTranslucent,
        MovingImageAndHuge,
        StillChangedBetweenPlays,
        CameraBackgroundChanges,
        CameraMovesOverStill,
    ],
)
def test_frames_repaint_only_what_moves_and_what_is_drawn_over_it(render, scene_class):
    render(scene_class)


def test_a_frame_draws_what_moves_and_the_still_mobjects_it_reaches(
    render, monkeypatch
):
    """A spinning square on the right, 40 still circles on the left and one
    still line over the square: after the first frame, which paints the still
    ones once, a frame draws the square and the line.
    """
    from manim.camera.camera import Camera

    drawn: list[int] = []
    fill = Camera.apply_fill

    def counted(self, ctx, vmobject):
        if drawn and drawn[-1] is not None:  # inside a frame, not the fixture's check
            drawn[-1] += 1
        return fill(self, ctx, vmobject)

    monkeypatch.setattr(Camera, "apply_fill", counted)
    paint = CairoRenderer._update_play_frame

    def framed(self, scene):
        drawn.append(0)
        paint(self, scene)
        drawn.append(None)

    monkeypatch.setattr(CairoRenderer, "_update_play_frame", framed)

    class Spinning(Scene):
        def construct(self):
            square = Square().shift(RIGHT * 3).add_updater(lambda m, dt: m.rotate(dt))
            self.add(square, circles(), Line(RIGHT * 2, RIGHT * 4))
            self.wait(0.5)

    render(Spinning)
    frames = [n for n in drawn if n is not None]
    assert (
        frames[0] == 41 + 2
    )  # the still frame once (40 circles, the line), and the box
    assert set(frames[1:]) == {2}
