"""
Articulation, checked against closed-form truth rather than against itself.

A kinematics test that only asserts "the solver converged" proves nothing: a
solver can converge confidently onto the wrong point. So the geometry here is
chosen to have an answer that can be written down. A planar two-link arm has an
elbow angle given exactly by the law of cosines, a hinge sweeps a circle of
known radius, and a target beyond the chain misses by exactly
`distance - chain_length`. Those are the assertions.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

import lathe.build as B
from lathe import pose as P


def _arm(upper: float = 0.30, fore: float = 0.28, elbow_limit=None):
    """A planar two-link arm hinging about Y, hanging down -Z from (0,0,0).

    Straight down at rest, so the tip sits at -(upper + fore) in Z and the whole
    thing stays in the XZ plane -- which is what makes the closed form apply.
    """
    limit = elbow_limit if elbow_limit is not None else (0.0, P.deg(150))
    return P.Skeleton(
        (
            P.Joint("shoulder", offset=(0.0, 0.0, 0.0), axis=(0.0, 1.0, 0.0)),
            P.Joint("elbow", parent="shoulder", offset=(0.0, 0.0, -upper),
                    axis=(0.0, 1.0, 0.0), limit=limit),
            P.Joint("hand", parent="elbow", offset=(0.0, 0.0, -fore),
                    axis=(0.0, 1.0, 0.0), limit=(0.0, 0.0)),
        ),
        name="arm",
    )


class TestSkeleton(unittest.TestCase):
    def test_parents_come_before_children(self):
        sk = _arm()
        self.assertEqual(sk.names, ("shoulder", "elbow", "hand"))

    def test_declaration_order_does_not_matter(self):
        """Joints are a tree, not a list. Writing a child first is legal."""
        sk = P.Skeleton((
            P.Joint("hand", parent="elbow", offset=(0.0, 0.0, -0.28)),
            P.Joint("elbow", parent="shoulder", offset=(0.0, 0.0, -0.30)),
            P.Joint("shoulder"),
        ))
        self.assertEqual(sk.names, ("shoulder", "elbow", "hand"))

    def test_a_cycle_is_named_not_hung_on(self):
        with self.assertRaises(P.RigError) as cm:
            P.Skeleton((
                P.Joint("a", parent="b"),
                P.Joint("b", parent="a"),
            ))
        self.assertIn("cycle", str(cm.exception))

    def test_unknown_parent_lists_what_exists(self):
        with self.assertRaises(P.RigError) as cm:
            P.Skeleton((P.Joint("elbow", parent="shoulder"),))
        self.assertIn("shoulder", str(cm.exception))

    def test_duplicate_names_are_rejected(self):
        """Names are the addressing scheme, so a collision is unaddressable."""
        with self.assertRaises(P.RigError):
            P.Skeleton((P.Joint("a"), P.Joint("a")))

    def test_chain_walks_up_from_the_tip(self):
        self.assertEqual(_arm().chain("shoulder", "hand"),
                         ("shoulder", "elbow", "hand"))

    def test_a_chain_that_is_not_a_chain_says_so(self):
        sk = P.Skeleton((
            P.Joint("root"),
            P.Joint("left", parent="root"),
            P.Joint("right", parent="root"),
        ))
        with self.assertRaises(P.RigError) as cm:
            sk.chain("left", "right")
        self.assertIn("ancestor", str(cm.exception))

    def test_rest_clamps_into_the_declared_limit(self):
        """A joint whose range excludes zero must not report zero as its rest."""
        sk = P.Skeleton((P.Joint("j", limit=(P.deg(10), P.deg(90))),))
        self.assertAlmostEqual(sk.rest()["j"], P.deg(10), places=12)

    def test_zero_length_axis_is_rejected(self):
        sk = P.Skeleton((P.Joint("j", axis=(0.0, 0.0, 0.0)),))
        with self.assertRaises(P.RigError):
            P.forward(sk)


class TestForwardKinematics(unittest.TestCase):
    def test_rest_pose_hangs_straight_down(self):
        pos = P.positions(_arm())
        np.testing.assert_allclose(pos["hand"], [0.0, 0.0, -0.58], atol=1e-12)

    def test_a_hinge_sweeps_a_circle_of_known_radius(self):
        """Right-hand rule about +Y carries +Z to +X, so it carries the
        downward-hanging tip to -X. The distance from the pivot is unchanged --
        a hinge rotates its subtree and never stretches it."""
        pos = P.positions(_arm(), {"shoulder": P.deg(90)})
        np.testing.assert_allclose(pos["hand"], [-0.58, 0.0, 0.0], atol=1e-12)

    def test_the_elbow_moves_the_hand_and_not_the_shoulder(self):
        pos = P.positions(_arm(), {"elbow": P.deg(90)})
        np.testing.assert_allclose(pos["shoulder"], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(pos["elbow"], [0.0, 0.0, -0.30], atol=1e-12)
        # Forearm now points -X from the elbow, and the elbow itself has not moved.
        np.testing.assert_allclose(pos["hand"], [-0.28, 0.0, -0.30], atol=1e-12)

    def test_rotations_compose_down_the_chain(self):
        """Two 45 degree hinges equal one 90 degree bend at the tip."""
        a = P.positions(_arm(), {"shoulder": P.deg(45), "elbow": P.deg(45)})
        b = P.positions(_arm(), {"shoulder": P.deg(90)})
        # Same total rotation of the forearm segment: both put the hand at the
        # same distance from the origin, and the forearm horizontal.
        self.assertAlmostEqual(float(a["hand"][2]), float(a["elbow"][2]), places=12)
        self.assertAlmostEqual(float(np.linalg.norm(b["hand"])), 0.58, places=12)

    def test_a_partial_pose_leaves_the_rest_alone(self):
        pos = P.positions(_arm(), {"elbow": P.deg(30)})
        np.testing.assert_allclose(pos["elbow"], [0.0, 0.0, -0.30], atol=1e-12)

    def test_an_unknown_joint_in_a_pose_is_an_error(self):
        """Silently ignoring it would make a typo invisible."""
        with self.assertRaises(P.RigError) as cm:
            P.forward(_arm(), {"knee": 0.5})
        self.assertIn("knee", str(cm.exception))

    def test_chain_length_is_the_sum_of_offsets(self):
        sk = _arm()
        self.assertAlmostEqual(P.chain_length(sk, sk.chain("shoulder", "hand")),
                               0.58, places=12)

    def test_forward_kinematics_is_pure(self):
        a = P.positions(_arm(), {"shoulder": 0.4, "elbow": 0.9})
        b = P.positions(_arm(), {"shoulder": 0.4, "elbow": 0.9})
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])


class TestInverseKinematics(unittest.TestCase):
    def test_elbow_angle_matches_the_law_of_cosines(self):
        """The closed form. For links a and b reaching distance d, the interior
        elbow angle is acos((a^2 + b^2 - d^2) / 2ab), and the joint angle is
        its supplement because the arm is straight at zero."""
        a, b = 0.30, 0.28
        sk = _arm(a, b)
        d = 0.40
        # -X, because a positive elbow angle folds the forearm that way and the
        # default limit is positive-only. See the workspace test below.
        #
        # Solved to 1e-12 m rather than the 1e-5 default: the point here is that
        # the solver lands on the analytic answer and not merely near it, and at
        # the default tolerance the residual *angle* is 2e-5 rad -- which is the
        # position tolerance faithfully converted, not an error.
        hit = P.solve(sk, "shoulder", "hand", (-d, 0.0, 0.0),
                      tolerance=1e-12, max_passes=512)

        self.assertTrue(hit.converged, hit.describe())
        interior = math.acos((a * a + b * b - d * d) / (2 * a * b))
        expected = math.pi - interior
        self.assertAlmostEqual(abs(hit.pose["elbow"]), expected, places=9)

    def test_the_tip_lands_on_the_target(self):
        sk = _arm()
        target = (-0.35, 0.0, -0.20)
        hit = P.solve(sk, "shoulder", "hand", target)
        self.assertTrue(hit.converged, hit.describe())
        self.assertLess(hit.error, P.DEFAULT_TOLERANCE)
        np.testing.assert_allclose(
            P.positions(sk, hit.pose)["hand"], target, atol=1e-4)

    def test_a_straight_reach_needs_no_elbow(self):
        """Target at exactly full extension: the elbow must stay straight."""
        sk = _arm()
        hit = P.solve(sk, "shoulder", "hand", (0.0, 0.0, -0.58))
        self.assertTrue(hit.converged, hit.describe())
        self.assertAlmostEqual(hit.pose["elbow"], 0.0, places=6)

    def test_an_unreachable_target_reports_the_exact_deficit(self):
        """0.58 m of arm cannot reach 0.90 m. The miss is 0.32 m, not 'failed'."""
        sk = _arm()
        hit = P.solve(sk, "shoulder", "hand", (0.90, 0.0, 0.0))
        self.assertFalse(hit.reachable)
        self.assertFalse(hit.converged)
        self.assertAlmostEqual(hit.reach_deficit, 0.32, places=9)
        # Best effort points straight at the target, so the miss equals the deficit.
        self.assertAlmostEqual(hit.error, 0.32, places=5)
        self.assertIn("beyond the chain", hit.describe())

    def test_a_limit_is_respected_even_when_it_costs_the_solve(self):
        """An elbow capped at 30 degrees cannot fold far enough to reach a near
        target on the side it *can* bend toward. The solver must miss rather
        than exceed the declared limit."""
        sk = _arm(elbow_limit=(0.0, P.deg(30)))
        hit = P.solve(sk, "shoulder", "hand", (-0.05, 0.0, -0.15))
        self.assertLessEqual(hit.pose["elbow"], P.deg(30) + 1e-9)
        self.assertGreaterEqual(hit.pose["elbow"], -1e-9)
        self.assertIn("elbow", hit.at_limit)
        self.assertFalse(hit.converged)

    def test_every_solved_pose_is_inside_every_limit(self):
        sk = _arm()
        for target in [(-0.2, 0.0, -0.3), (-0.4, 0.0, 0.1), (-0.3, 0.0, -0.2)]:
            with self.subTest(target=target):
                hit = P.solve(sk, "shoulder", "hand", target)
                self.assertEqual(P.violations(sk, hit.pose), ())

    def test_a_locked_joint_never_moves(self):
        sk = _arm()
        hit = P.solve(sk, "shoulder", "hand", (0.3, 0.0, -0.2))
        self.assertAlmostEqual(hit.pose["hand"], 0.0, places=12)

    def test_a_one_sided_limit_halves_the_workspace(self):
        """The most practical consequence of a hinge limit, and the easiest
        thing to be surprised by.

        A positive elbow angle folds the forearm toward -X, so an elbow that may
        only open positively reaches -X targets and cannot reach their mirror.
        The far side is not a solver failure -- it is the declared limit doing
        exactly what it says, and the report names it rather than shrugging.
        """
        sk = _arm(elbow_limit=(0.0, P.deg(150)))
        near = P.solve(sk, "shoulder", "hand", (-0.40, 0.0, 0.0))
        far = P.solve(sk, "shoulder", "hand", (0.40, 0.0, 0.0))

        self.assertTrue(near.converged, near.describe())
        self.assertFalse(far.converged)
        self.assertTrue(far.reachable)        # within arm's length, just not bendable
        self.assertIn("elbow", far.at_limit)
        self.assertEqual(P.violations(sk, far.pose), ())

    def test_a_decisive_seed_breaks_the_elbow_up_or_down_tie(self):
        """Two-link IK has two solutions. The solver does not guess -- it keeps
        whichever branch the seed sits in.

        The seed has to be decisive to matter: +-20 degrees lands in the same
        basin either way, and only a seed past the fold selects the other
        branch. Measured, not assumed -- both of these converge onto the same
        point by genuinely different poses.
        """
        sk = _arm(elbow_limit=(-P.deg(150), P.deg(150)))
        target = (0.40, 0.0, 0.0)
        up = P.solve(sk, "shoulder", "hand", target, seed={"elbow": P.deg(90)})
        down = P.solve(sk, "shoulder", "hand", target, seed={"elbow": -P.deg(90)})

        self.assertTrue(up.converged and down.converged)
        self.assertGreater(up.pose["elbow"], 0.0)
        self.assertLess(down.pose["elbow"], 0.0)
        self.assertNotAlmostEqual(up.pose["shoulder"], down.pose["shoulder"], places=3)
        for hit in (up, down):
            np.testing.assert_allclose(
                P.positions(sk, hit.pose)["hand"], target, atol=1e-4)

    def test_solving_is_deterministic(self):
        """Principle 1. Byte-identical, not merely close."""
        sk = _arm()
        a = P.solve(sk, "shoulder", "hand", (0.31, 0.0, -0.22))
        b = P.solve(sk, "shoulder", "hand", (0.31, 0.0, -0.22))
        self.assertEqual(a.pose, b.pose)
        self.assertEqual(a.error, b.error)
        self.assertEqual(a.passes, b.passes)

    def test_solving_does_not_disturb_joints_outside_the_chain(self):
        sk = P.Skeleton((
            P.Joint("root"),
            P.Joint("shoulder", parent="root", offset=(0.0, 0.0, 1.4)),
            P.Joint("elbow", parent="shoulder", offset=(0.0, 0.0, -0.30)),
            P.Joint("hand", parent="elbow", offset=(0.0, 0.0, -0.28)),
            P.Joint("tail", parent="root", offset=(0.0, 0.0, -0.20)),
        ))
        hit = P.solve(sk, "shoulder", "hand", (0.30, 0.0, 1.10),
                      seed={"tail": P.deg(33)})
        self.assertAlmostEqual(hit.pose["tail"], P.deg(33), places=12)

    def test_a_one_joint_chain_is_refused_with_a_reason(self):
        with self.assertRaises(P.RigError) as cm:
            P.solve(_arm(), "hand", "hand", (0.0, 0.0, 0.0))
        self.assertIn("nothing to solve", str(cm.exception))

    def test_reach_error_measures_a_pose_it_did_not_solve(self):
        """The assertable number, usable on a hand-authored pose."""
        sk = _arm()
        self.assertAlmostEqual(
            P.reach_error(sk, "hand", (0.0, 0.0, 0.0)), 0.58, places=12)

    def test_violations_names_an_out_of_range_hand_authored_pose(self):
        """`solve` clamps. Nothing clamps a pose typed by a human or lifted off
        a photograph, which is exactly why this exists."""
        sk = _arm()
        bad = P.violations(sk, {"elbow": P.deg(200)})
        self.assertEqual(len(bad), 1)
        name, angle, lo, hi = bad[0]
        self.assertEqual(name, "elbow")
        self.assertAlmostEqual(hi, P.deg(150), places=12)


class TestRig(unittest.TestCase):
    def _rig(self):
        sk = _arm()
        upper = B.box(0.06, 0.06, 0.32, tag="upper").at(z=-0.15)
        fore = B.box(0.05, 0.05, 0.30, tag="fore").at(z=-0.44)
        return P.Rig(sk, name="SM_Arm_Test_LOD0").bind("shoulder", upper).bind("elbow", fore)

    def test_the_rest_pose_is_the_geometry_as_authored(self):
        """Rigging an existing model must not move it. Otherwise every asset
        would need rebuilding to gain a skeleton."""
        rig = self._rig()
        lo, hi = rig.at_rest().bbox()
        self.assertAlmostEqual(lo[2], -0.59, places=9)
        self.assertAlmostEqual(hi[2], 0.01, places=9)

    def test_posing_moves_the_bound_part(self):
        rig = self._rig()
        rested = rig.part("elbow").bbox()
        turned = rig.part("elbow", {"elbow": P.deg(90)}).bbox()
        self.assertNotAlmostEqual(rested[1][0], turned[1][0], places=3)

    def test_a_hinge_preserves_the_volume_of_each_part(self):
        """A rotation is rigid. Any volume change means the transform was not.

        Measured per part rather than on the assembly: the parts overlap across
        the pivot on purpose, and bending the joint changes how much of that
        overlap the union double-counts. The assembly's volume is *supposed* to
        move. A single part's is not.
        """
        rig = self._rig()
        bent = {"shoulder": P.deg(35), "elbow": P.deg(70)}
        for joint in rig.bound:
            with self.subTest(joint=joint):
                self.assertAlmostEqual(rig.part(joint).volume(),
                                       rig.part(joint, bent).volume(), places=12)

    def test_the_shoulder_carries_the_forearm_with_it(self):
        """The forearm is bound to the elbow, but the shoulder is its parent, so
        rotating the shoulder must move it. This is the whole point of a chain
        and the easiest thing to get wrong by applying only the local matrix."""
        rig = self._rig()
        rest = P.positions(rig.skeleton)["hand"]
        turned = P.positions(rig.skeleton, {"shoulder": P.deg(90)})["hand"]
        self.assertAlmostEqual(float(rest[2]), -0.58, places=12)
        self.assertAlmostEqual(float(turned[0]), -0.58, places=12)

        lo, _ = rig.part("elbow", {"shoulder": P.deg(90)}).bbox()
        self.assertLess(lo[0], -0.2)  # forearm swung out along -X

    def test_posed_output_is_an_ordinary_solid(self):
        """It must drop straight into verify, perceive, render and export with
        no special case, or the subsystem has bought itself an island."""
        posed = self._rig().posed({"elbow": P.deg(45)})
        self.assertEqual(posed.name, "SM_Arm_Test_LOD0")
        self.assertTrue(posed.status_ok())
        self.assertGreater(posed.n_tris, 0)
        self.assertEqual(sorted(set(posed.tags.values())), ["fore", "upper"])

    def test_posing_is_pure(self):
        rig = self._rig()
        p = {"shoulder": P.deg(20), "elbow": P.deg(40)}
        self.assertAlmostEqual(rig.posed(p).volume(), rig.posed(p).volume(), places=15)

    def test_binding_twice_unions_rather_than_replaces(self):
        """Placed clear of the arm, so the added volume cannot hide inside
        geometry that was already there."""
        rig = self._rig()
        extra = B.box(0.04, 0.04, 0.04, tag="pad").at(x=0.30, z=-0.15)
        grown = rig.bind("shoulder", extra)
        self.assertAlmostEqual(grown.at_rest().volume() - rig.at_rest().volume(),
                               0.04 ** 3, places=9)
        self.assertIn("pad", set(grown.at_rest().tags.values()))

    def test_binding_to_an_unknown_joint_is_refused(self):
        with self.assertRaises(P.RigError):
            P.Rig(_arm()).bind("knee", B.box(0.1, 0.1, 0.1))

    def test_a_rig_with_no_geometry_says_what_to_do(self):
        with self.assertRaises(P.RigError) as cm:
            P.Rig(_arm()).posed()
        self.assertIn("bind", str(cm.exception))

    def test_unbound_joints_are_legal(self):
        """A tip marker usually carries nothing -- that is how IK targets a toe
        without inventing an end-effector concept."""
        self.assertIn("hand", self._rig().unbound)

    def test_component_count_is_pose_dependent_and_worth_asserting(self):
        """Why `rig.py` says to overlap across the pivot, and why E108 belongs
        on a posed rig rather than only on the rest shape.

        These two parts sit 10 mm apart and read as two components at rest. Turn
        the hinge and a corner swings into contact, welding them into one. The
        count is a function of the pose, so a rig that declares `components: 1`
        has made a claim about its whole range of motion, not about one shape.
        """
        sk = P.Skeleton((
            P.Joint("a"),
            P.Joint("b", parent="a", offset=(0.0, 0.0, -0.2)),
        ))
        rig = (P.Rig(sk)
               .bind("a", B.box(0.1, 0.1, 0.2, tag="a").at(z=-0.1))
               .bind("b", B.box(0.1, 0.1, 0.18, tag="b").at(z=-0.31)))
        self.assertEqual(rig.at_rest().components(), 2)
        self.assertEqual(rig.posed({"b": P.deg(25)}).components(), 1)

    def test_overlapping_parts_stay_one_component_through_a_pose(self):
        """The fix: overlap across the pivot deliberately."""
        sk = P.Skeleton((
            P.Joint("a"),
            P.Joint("b", parent="a", offset=(0.0, 0.0, -0.2)),
        ))
        rig = (P.Rig(sk)
               .bind("a", B.box(0.1, 0.1, 0.24, tag="a").at(z=-0.1))
               .bind("b", B.box(0.08, 0.08, 0.24, tag="b").at(z=-0.28)))
        self.assertEqual(rig.at_rest().components(), 1)
        self.assertEqual(rig.posed({"b": P.deg(25)}).components(), 1)


class TestUnits(unittest.TestCase):
    def test_deg_and_degrees_round_trip(self):
        self.assertAlmostEqual(P.degrees(P.deg(37.5)), 37.5, places=12)

    def test_ninety_degrees_is_a_quarter_turn(self):
        self.assertAlmostEqual(P.deg(90), math.pi / 2, places=15)


if __name__ == "__main__":
    unittest.main(verbosity=1)
