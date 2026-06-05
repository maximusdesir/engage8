"""Rules-based defensive play-call recommendation engine.

Given a run/pass prediction and the situation, this returns a ranked list of
defensive calls. There is no ML here: it is deliberately a transparent,
football-literate ruleset a coach can read and trust.

Vocabulary
----------
Fronts:    Bear, Over, Under, Tite, Even
Coverages: Cover 1, Cover 3, Cover 3 Buzz, Quarters, Cover 6
Pressures: None, Sim, Creeper, Fire Zone, Cross Dog
Stunts:    None, TEX, ET, Twist
"""
from __future__ import annotations


def _explosive_risk(prediction: dict, game_state: dict) -> float:
    """Heuristic 0-1 estimate of explosive-play risk.

    The ml model does not return an explosive probability, so we derive one
    from the situation: long-developing pass downs with a strong pass lean and
    space to throw into are the most explosive; the offense's confidence in the
    predicted call amplifies the risk.
    """
    down = game_state["down"]
    distance = game_state["distance"]
    yardline_100 = game_state["yardline_100"]
    pass_prob = prediction["pass_prob"]

    risk = 0.18  # baseline

    # Pass plays carry the bulk of explosive risk; weight by how likely.
    risk += 0.35 * pass_prob

    # 2nd/3rd & long = shot-play territory.
    if distance >= 8:
        risk += 0.18
    elif distance >= 5:
        risk += 0.08

    # Open field (not backed up, not in the red zone) gives room to run after
    # catch / break a long run.
    if 25 <= yardline_100 <= 80:
        risk += 0.10
    if yardline_100 <= 10:
        risk -= 0.10  # compressed field caps explosive length

    # Early downs let the offense take vertical shots more freely.
    if down <= 2 and distance >= 8:
        risk += 0.05

    return max(0.05, min(0.95, round(risk, 3)))


def _epa_prevented(confidence: float, explosive_risk: float, base: float) -> float:
    """Heuristic expected EPA prevented by getting the call right.

    Scales with how confident we are in the read and how much explosive damage
    is on the table. ``base`` differentiates aggressive vs. conservative calls.
    """
    value = base * (0.5 + 0.5 * confidence) * (0.6 + 0.8 * explosive_risk)
    return round(value, 3)


def recommend(prediction: dict, game_state: dict) -> list[dict]:
    """Return the top 2-3 defensive calls, ranked by confidence."""
    down = game_state["down"]
    distance = game_state["distance"]
    yardline_100 = game_state["yardline_100"]

    pass_prob = prediction["pass_prob"]
    run_prob = prediction["run_prob"]
    explosive_risk = _explosive_risk(prediction, game_state)

    red_zone = yardline_100 <= 20
    goal_to_go = yardline_100 <= distance
    obvious_pass = down == 3 and distance >= 7
    short_yardage = distance <= 2
    third_or_fourth = down >= 3

    calls: list[dict] = []

    # --- Obvious passing down, strong pass lean ---------------------------
    if (obvious_pass or distance >= 8) and pass_prob >= 0.6:
        conf = round(min(0.92, 0.55 + 0.35 * pass_prob), 3)
        calls.append(
            {
                "front": "Even",
                "coverage": "Cover 3 Buzz",
                "pressure": "Fire Zone",
                "stunt": "TEX",
                "confidence": conf,
                "expected_epa_prevented": _epa_prevented(conf, explosive_risk, 0.9),
                "rationale": (
                    f"{down} & {distance}, {int(pass_prob * 100)}% pass: drop into a "
                    "Cover 3 Buzz with the buzz defender robbing the intermediate, "
                    "five-man Fire Zone pressure with a TEX twist to muddy the launch "
                    "point without exposing the deep thirds."
                ),
            }
        )
        sim_conf = round(min(0.88, 0.50 + 0.32 * pass_prob), 3)
        calls.append(
            {
                "front": "Tite",
                "coverage": "Quarters",
                "pressure": "Sim",
                "stunt": "ET",
                "confidence": sim_conf,
                "expected_epa_prevented": _epa_prevented(sim_conf, explosive_risk, 0.75),
                "rationale": (
                    "Simulated pressure out of Quarters: show heat, drop to a 4-man "
                    "rush behind quarters to cap explosives over the top while the ET "
                    "stunt wins one-on-one inside."
                ),
            }
        )
        creeper_conf = round(min(0.82, 0.46 + 0.30 * pass_prob), 3)
        calls.append(
            {
                "front": "Even",
                "coverage": "Cover 6",
                "pressure": "Creeper",
                "stunt": "Twist",
                "confidence": creeper_conf,
                "expected_epa_prevented": _epa_prevented(creeper_conf, explosive_risk, 0.65),
                "rationale": (
                    "Cover 6 (quarter-quarter-half) with a zone creeper: replace a "
                    "rusher with an off-ball defender to keep four in coverage and "
                    "split-field leverage against the boundary shot."
                ),
            }
        )

    # --- Short yardage / heavy run lean -----------------------------------
    elif short_yardage and run_prob >= 0.55:
        conf = round(min(0.93, 0.58 + 0.32 * run_prob), 3)
        calls.append(
            {
                "front": "Bear",
                "coverage": "Cover 1",
                "pressure": "Cross Dog",
                "stunt": "None",
                "confidence": conf,
                "expected_epa_prevented": _epa_prevented(conf, explosive_risk, 0.85),
                "rationale": (
                    f"{down} & {distance}, {int(run_prob * 100)}% run: load a Bear front "
                    "to cover all the gaps, Cover 1 man behind it and a Cross Dog "
                    "linebacker blitz to spill the ball and win the short-yardage "
                    "down."
                ),
            }
        )
        over_conf = round(min(0.88, 0.54 + 0.30 * run_prob), 3)
        calls.append(
            {
                "front": "Over",
                "coverage": "Cover 3",
                "pressure": "None",
                "stunt": "ET",
                "confidence": over_conf,
                "expected_epa_prevented": _epa_prevented(over_conf, explosive_risk, 0.7),
                "rationale": (
                    "Over front into Cover 3 keeps an extra hat in the box with a "
                    "safety rolled down; the ET stunt knifes the run-side gap while "
                    "staying sound against play-action over the top."
                ),
            }
        )

    # --- Red zone (tight coverage, compressed field) ----------------------
    elif red_zone or goal_to_go:
        if pass_prob >= 0.5:
            conf = round(min(0.85, 0.52 + 0.28 * pass_prob), 3)
            calls.append(
                {
                    "front": "Even",
                    "coverage": "Cover 1",
                    "pressure": "Sim",
                    "stunt": "Twist",
                    "confidence": conf,
                    "expected_epa_prevented": _epa_prevented(conf, explosive_risk, 0.75),
                    "rationale": (
                        "Red zone pass look: tight Cover 1 man with a robber to wall "
                        "off crossers, simulated pressure to speed up the throw in a "
                        "compressed field where there is no deep grass to defend."
                    ),
                }
            )
            calls.append(
                {
                    "front": "Tite",
                    "coverage": "Quarters",
                    "pressure": "None",
                    "stunt": "TEX",
                    "confidence": round(min(0.80, 0.48 + 0.26 * pass_prob), 3),
                    "expected_epa_prevented": _epa_prevented(
                        round(min(0.80, 0.48 + 0.26 * pass_prob), 3),
                        explosive_risk,
                        0.6,
                    ),
                    "rationale": (
                        "Tite front, Quarters coverage: pattern-match the route "
                        "distributions and keep eyes on the quarterback for the scramble "
                        "while the TEX stunt pressures with four."
                    ),
                }
            )
        else:
            conf = round(min(0.88, 0.55 + 0.30 * run_prob), 3)
            calls.append(
                {
                    "front": "Bear",
                    "coverage": "Cover 1",
                    "pressure": "Cross Dog",
                    "stunt": "None",
                    "confidence": conf,
                    "expected_epa_prevented": _epa_prevented(conf, explosive_risk, 0.78),
                    "rationale": (
                        "Goal-line run look: Bear front to plug every gap, Cover 1 man "
                        "behind it and a Cross Dog to penetrate before the back hits "
                        "the line."
                    ),
                }
            )
            calls.append(
                {
                    "front": "Over",
                    "coverage": "Cover 3",
                    "pressure": "None",
                    "stunt": "ET",
                    "confidence": round(min(0.82, 0.50 + 0.28 * run_prob), 3),
                    "expected_epa_prevented": _epa_prevented(
                        round(min(0.82, 0.50 + 0.28 * run_prob), 3),
                        explosive_risk,
                        0.62,
                    ),
                    "rationale": (
                        "Over front, Cover 3 with the down safety as the extra run "
                        "fitter; ET stunt squeezes the play-side B gap."
                    ),
                }
            )

    # --- Balanced / early down --------------------------------------------
    else:
        if pass_prob >= 0.5:
            conf = round(min(0.78, 0.48 + 0.24 * pass_prob), 3)
            calls.append(
                {
                    "front": "Even",
                    "coverage": "Quarters",
                    "pressure": "None",
                    "stunt": "Twist",
                    "confidence": conf,
                    "expected_epa_prevented": _epa_prevented(conf, explosive_risk, 0.62),
                    "rationale": (
                        f"{down} & {distance}, slight pass lean: stay sound in Quarters "
                        "to cap explosives, rush four with a Twist to win up front "
                        "without committing extra defenders."
                    ),
                }
            )
            calls.append(
                {
                    "front": "Under",
                    "coverage": "Cover 3 Buzz",
                    "pressure": "Creeper",
                    "stunt": "ET",
                    "confidence": round(min(0.74, 0.45 + 0.22 * pass_prob), 3),
                    "expected_epa_prevented": _epa_prevented(
                        round(min(0.74, 0.45 + 0.22 * pass_prob), 3),
                        explosive_risk,
                        0.55,
                    ),
                    "rationale": (
                        "Under front, Cover 3 Buzz with a zone creeper to disguise the "
                        "rush while keeping a defender on the intermediate hole."
                    ),
                }
            )
        else:
            conf = round(min(0.80, 0.50 + 0.24 * run_prob), 3)
            calls.append(
                {
                    "front": "Over",
                    "coverage": "Cover 3",
                    "pressure": "None",
                    "stunt": "ET",
                    "confidence": conf,
                    "expected_epa_prevented": _epa_prevented(conf, explosive_risk, 0.6),
                    "rationale": (
                        f"{down} & {distance}, run lean: Over front into Cover 3 keeps "
                        "the box sound with a rolled-down safety; ET stunt attacks the "
                        "front-side gap."
                    ),
                }
            )
            calls.append(
                {
                    "front": "Tite",
                    "coverage": "Quarters",
                    "pressure": "None",
                    "stunt": "TEX",
                    "confidence": round(min(0.76, 0.47 + 0.22 * run_prob), 3),
                    "expected_epa_prevented": _epa_prevented(
                        round(min(0.76, 0.47 + 0.22 * run_prob), 3),
                        explosive_risk,
                        0.55,
                    ),
                    "rationale": (
                        "Tite front spills the ball outside to unblocked overhang "
                        "defenders; Quarters keeps two-high integrity against "
                        "play-action."
                    ),
                }
            )

    # Rank by confidence, then by expected EPA prevented; cap at top 3.
    calls.sort(
        key=lambda c: (c["confidence"], c["expected_epa_prevented"]),
        reverse=True,
    )
    return calls[:3]
