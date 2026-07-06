import argparse
import unittest
from unittest.mock import patch

from run_pipeline import (
  aggregate_candidate_wins,
  build_parser,
  eval_enabled,
  should_promote,
  validate_eval_args,
)


class ShouldPromoteTests(unittest.TestCase):
  def test_threshold_boundary_fails(self):
    self.assertFalse(should_promote(0.55, 0.55))

  def test_just_above_threshold_passes(self):
    self.assertTrue(should_promote(0.5501, 0.55))

  def test_below_threshold_fails(self):
    self.assertFalse(should_promote(0.54, 0.55))


class EvalEnabledTests(unittest.TestCase):
  def test_disabled_when_zero(self):
    args = argparse.Namespace(num_eval_games=0)
    self.assertFalse(eval_enabled(args))

  def test_enabled_when_positive(self):
    args = argparse.Namespace(num_eval_games=10)
    self.assertTrue(eval_enabled(args))


class ValidateEvalArgsTests(unittest.TestCase):
  def _args(self, **overrides):
    defaults = {
      "num_eval_games": 0,
      "winrate_threshold": 0.55,
      "player1_type": "nmcts",
      "player2_type": "nmcts",
      "player1_model": None,
      "player2_model": None,
      "accumulate_records": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)

  def test_rejects_negative_eval_games(self):
    with self.assertRaises(ValueError):
      validate_eval_args(self._args(num_eval_games=-1))

  def test_rejects_invalid_threshold(self):
    with self.assertRaises(ValueError):
      validate_eval_args(self._args(winrate_threshold=0.0))
    with self.assertRaises(ValueError):
      validate_eval_args(self._args(winrate_threshold=1.0))

  def test_requires_dynamic_nmcts_when_eval_enabled(self):
    with self.assertRaises(ValueError):
      validate_eval_args(self._args(
        num_eval_games=10,
        player1_type="nmcts",
        player1_model="fixed.pt",
        player2_type="nmcts",
        player2_model="fixed.pt",
      ))

  def test_allows_eval_with_one_dynamic_nmcts_player(self):
    validate_eval_args(self._args(
      num_eval_games=10,
      player1_type="nmcts",
      player1_model=None,
      player2_type="nmcts",
      player2_model="fixed.pt",
    ))

  def test_warns_when_accumulate_records_with_eval_enabled(self):
    import warnings
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      validate_eval_args(self._args(num_eval_games=10, accumulate_records=True))
    self.assertTrue(any("--accumulate-records is ignored" in str(w.message) for w in caught))


class AggregateCandidateWinsTests(unittest.TestCase):
  def test_candidate_wins_from_both_seat_orderings(self):
    candidate_as_p1 = {
      "player_one_wins": 7,
      "draws": 2,
      "player_two_wins": 1,
    }
    candidate_as_p2 = {
      "player_one_wins": 3,
      "draws": 1,
      "player_two_wins": 6,
    }
    candidate_wins, candidate_win_rate, num_games = aggregate_candidate_wins(candidate_as_p1, candidate_as_p2)
    self.assertEqual(candidate_wins, 13)
    self.assertEqual(num_games, 20)
    self.assertAlmostEqual(candidate_win_rate, 0.65)

  def test_draws_do_not_count_as_candidate_wins(self):
    candidate_as_p1 = {
      "player_one_wins": 0,
      "draws": 5,
      "player_two_wins": 0,
    }
    candidate_as_p2 = {
      "player_one_wins": 0,
      "draws": 5,
      "player_two_wins": 0,
    }
    candidate_wins, candidate_win_rate, num_games = aggregate_candidate_wins(candidate_as_p1, candidate_as_p2)
    self.assertEqual(candidate_wins, 0)
    self.assertEqual(num_games, 10)
    self.assertAlmostEqual(candidate_win_rate, 0.0)


class BuildParserTests(unittest.TestCase):
  def test_eval_defaults(self):
    parser = build_parser()
    args = parser.parse_args([
      "--game-type", "TTT",
      "--rounds", "1",
      "--games-per-round", "1",
      "--output-dir", "output",
      "--player1-type", "mcts",
      "--player2-type", "mcts",
    ])
    self.assertEqual(args.num_eval_games, 0)
    self.assertEqual(args.winrate_threshold, 0.55)


class RunCandidateEvalTests(unittest.TestCase):
  @patch("run_pipeline.run_matches")
  def test_seat_swapping_and_win_rate(self, mock_run_matches):
    from run_pipeline import run_candidate_eval

    mock_run_matches.side_effect = [
      {"player_one_wins": 6, "draws": 1, "player_two_wins": 3},
      {"player_one_wins": 2, "draws": 0, "player_two_wins": 8},
    ]

    result = run_candidate_eval(
      game_type="TTT",
      num_games=20,
      candidate_ckpt="candidate.pt",
      champion_ckpt="champion.pt",
      iters=25,
      play_device="cpu",
      workers=1,
      non_interactive_logging=False,
    )

    self.assertEqual(mock_run_matches.call_count, 2)
    first_call = mock_run_matches.call_args_list[0].kwargs
    second_call = mock_run_matches.call_args_list[1].kwargs
    self.assertEqual(first_call["num_games"], 10)
    self.assertEqual(first_call["player_one_model"], "candidate.pt")
    self.assertEqual(first_call["player_two_model"], "champion.pt")
    self.assertEqual(second_call["num_games"], 10)
    self.assertEqual(second_call["player_one_model"], "champion.pt")
    self.assertEqual(second_call["player_two_model"], "candidate.pt")

    self.assertEqual(result["candidate_wins"], 14)
    self.assertEqual(result["num_games"], 20)
    self.assertAlmostEqual(result["candidate_win_rate"], 0.7)


if __name__ == "__main__":
  unittest.main()
