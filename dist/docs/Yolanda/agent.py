from collections.abc import Callable
from time import time
from typing import Deque, Dict, List, Optional, Set, Tuple
from collections import deque

import numpy as np
from game import board
from game.enums import Direction, MoveType, Result


class PlayerAgent:
    """
    Heuristic-first agent with shallow alpha-beta search, trapdoor beliefs,
    and stall/anti-oscillation guards.
    """

    HIGH_TRAP_THRESHOLD = 0.35
    MIN_SAFE_AREA_AFTER_EGG = 6

    def __init__(self, board: board.Board, time_left: Callable):
        self.map_size = board.game_map.MAP_SIZE

        # Trapdoor beliefs (parity-separated)
        self.trapdoor_beliefs = [
            np.ones((self.map_size, self.map_size), dtype=np.float32),
            np.ones((self.map_size, self.map_size), dtype=np.float32),
        ]
        for i in range(self.map_size):
            for j in range(self.map_size):
                parity = (i + j) % 2
                if parity == 0:
                    self.trapdoor_beliefs[1][i, j] = 0.0
                else:
                    self.trapdoor_beliefs[0][i, j] = 0.0
        for belief_map in self.trapdoor_beliefs:
            total = float(belief_map.sum())
            if total > 0:
                belief_map /= total

        self.found_trapdoors: Set[Tuple[int, int]] = set()
        self._risk_cache: Optional[np.ndarray] = None

        # Search parameters
        self.max_depth = 6
        self.q_depth = 2
        self.time_safety_factor = 0.9
        self.tt_size = 60000
        self.transposition_table: Dict[Tuple, Tuple[int, float, str, float, float]] = {}
        self.killer_moves: Dict[int, List[Tuple[Direction, MoveType]]] = {}
        self.history_table: Dict[Tuple[Direction, MoveType], int] = {}

        # Anti-oscillation/stall tracking
        self.recent_positions: List[Tuple[int, int]] = []
        self.turns_since_egg = 0
        self.last_egg_count = 0

    # ------------------------------------------------------------------
    # Trapdoor handling
    # ------------------------------------------------------------------

    def _mark_tile_safe(self, location: Tuple[int, int]):
        """We know we are standing on a safe tile; zero its belief mass."""
        x, y = location
        if not (0 <= x < self.map_size and 0 <= y < self.map_size):
            return
        for idx in range(2):
            self.trapdoor_beliefs[idx][x, y] = 0.0
            total = float(self.trapdoor_beliefs[idx].sum())
            if total > 0:
                self.trapdoor_beliefs[idx] /= total
        self._risk_cache = None

    def update_trapdoor_beliefs(
        self,
        board_obj: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        location: Tuple[int, int],
    ):
        """Bayesian update using (heard, felt) for each trapdoor parity."""
        chicken = board_obj.chicken_player
        for trap_idx in range(2):
            did_hear, did_feel = sensor_data[trap_idx]
            for i in range(self.map_size):
                for j in range(self.map_size):
                    parity = (i + j) % 2
                    if (trap_idx == 0 and parity != 0) or (trap_idx == 1 and parity != 1):
                        continue
                    hear_prob, feel_prob = chicken.prob_senses_if_trapdoor_were_at(
                        did_hear, did_feel, i, j
                    )
                    self.trapdoor_beliefs[trap_idx][i, j] *= hear_prob * feel_prob
            total = float(self.trapdoor_beliefs[trap_idx].sum())
            if total > 0:
                self.trapdoor_beliefs[trap_idx] /= total
        self._risk_cache = None

    def get_trapdoor_risk(self, location: Tuple[int, int]) -> float:
        """Risk of trapdoor at a tile."""
        if location in self.found_trapdoors:
            return 1.0
        if self._risk_cache is None:
            self._risk_cache = self.trapdoor_beliefs[0] + self.trapdoor_beliefs[1]
        x, y = location
        if 0 <= x < self.map_size and 0 <= y < self.map_size:
            return float(self._risk_cache[x, y])
        return 0.0

    # ------------------------------------------------------------------
    # Territory estimation
    # ------------------------------------------------------------------

    def _is_blocked_for(
        self,
        pos: Tuple[int, int],
        my_eggs: Set[Tuple[int, int]],
        my_turds: Set[Tuple[int, int]],
        opp_eggs: Set[Tuple[int, int]],
        opp_turds: Set[Tuple[int, int]],
        avoid_adjacent_to_opp_turd: bool,
        planning: bool,
    ) -> bool:
        x, y = pos
        if self.get_trapdoor_risk(pos) >= self.HIGH_TRAP_THRESHOLD:
            return True
        if pos in my_turds or pos in opp_turds:
            return True
        if pos in opp_eggs:
            return True
        if (not planning) and pos in my_eggs:
            return True
        if avoid_adjacent_to_opp_turd:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in opp_turds:
                    return True
        return False

    def _reachable_area(
        self,
        board_obj: board.Board,
        for_enemy: bool,
        max_steps: int = 7,
        planning: bool = True,
    ) -> int:
        """BFS area estimate; allows own eggs if planning=True to avoid fake self-traps."""
        if for_enemy:
            start = board_obj.chicken_enemy.get_location()
            my_eggs = set(board_obj.eggs_enemy)
            my_turds = set(board_obj.turds_enemy)
            opp_eggs = set(board_obj.eggs_player)
            opp_turds = set(board_obj.turds_player)
        else:
            start = board_obj.chicken_player.get_location()
            my_eggs = set(board_obj.eggs_player)
            my_turds = set(board_obj.turds_player)
            opp_eggs = set(board_obj.eggs_enemy)
            opp_turds = set(board_obj.turds_enemy)

        visited: Set[Tuple[int, int]] = set()
        q: Deque[Tuple[Tuple[int, int], int]] = deque()
        q.append((start, 0))
        visited.add(start)
        area = 0

        while q:
            (x, y), d = q.popleft()
            if d > max_steps:
                continue
            if self._is_blocked_for(
                (x, y), my_eggs, my_turds, opp_eggs, opp_turds, True, planning
            ):
                continue
            area += 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.map_size and 0 <= ny < self.map_size):
                    continue
                nxt = (nx, ny)
                if nxt in visited:
                    continue
                if self._is_blocked_for(
                    nxt, my_eggs, my_turds, opp_eggs, opp_turds, True, planning
                ):
                    continue
                visited.add(nxt)
                q.append((nxt, d + 1))
        return area

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_position(self, board_obj: board.Board) -> float:
        if board_obj.is_game_over():
            winner = board_obj.get_winner()
            if winner == Result.PLAYER:
                return 10_000.0
            if winner == Result.ENEMY:
                return -10_000.0
            return 0.0

        my_chick = board_obj.chicken_player
        enemy_chick = board_obj.chicken_enemy

        my_eggs = my_chick.get_eggs_laid()
        enemy_eggs = enemy_chick.get_eggs_laid()
        egg_diff = my_eggs - enemy_eggs

        my_turds_left = my_chick.get_turds_left()
        enemy_turds_left = enemy_chick.get_turds_left()

        my_moves = len(board_obj.get_valid_moves())
        enemy_moves = len(board_obj.get_valid_moves(enemy=True))
        mobility_diff = my_moves - enemy_moves

        my_loc = my_chick.get_location()
        trap_risk_here = self.get_trapdoor_risk(my_loc)

        corners = [
            (0, 0),
            (0, self.map_size - 1),
            (self.map_size - 1, 0),
            (self.map_size - 1, self.map_size - 1),
        ]
        my_corner_dist = min(abs(my_loc[0] - cx) + abs(my_loc[1] - cy) for (cx, cy) in corners)
        enemy_loc = enemy_chick.get_location()
        enemy_corner_dist = min(
            abs(enemy_loc[0] - cx) + abs(enemy_loc[1] - cy) for (cx, cy) in corners
        )
        corner_advantage = enemy_corner_dist - my_corner_dist

        my_area = self._reachable_area(board_obj, for_enemy=False, max_steps=7, planning=True)
        enemy_area = self._reachable_area(board_obj, for_enemy=True, max_steps=7, planning=True)
        area_diff = my_area - enemy_area

        turns_left = board_obj.turns_left_player
        score = 0.0

        if turns_left <= 10:
            score += egg_diff * 80.0
            if egg_diff <= 0:
                score += 200.0
            score += area_diff * 0.5
            score += mobility_diff * 1.0
            score += (my_turds_left - enemy_turds_left) * 2.0
            score += corner_advantage * 1.0
            score -= trap_risk_here * 3.0
        else:
            score += egg_diff * 70.0
            score += area_diff * 2.5
            score += mobility_diff * 3.0
            score += (my_turds_left - enemy_turds_left) * 2.0
            score += corner_advantage * 1.2
            if egg_diff >= 0:
                score -= trap_risk_here * 12.0
            else:
                score -= trap_risk_here * 7.0
            if egg_diff < 0:
                score += min(25.0, abs(egg_diff) * 7.0)
                score += mobility_diff * 1.0

        if enemy_moves == 0 and board_obj.turns_left_enemy > 0:
            score += 150.0

        return score

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------

    def order_moves(
        self, board_obj: board.Board, moves: List[Tuple[Direction, MoveType]], depth: int
    ) -> List[Tuple[Direction, MoveType]]:
        my_loc = board_obj.chicken_player.get_location()
        center = (self.map_size - 1) / 2.0
        egg_diff = board_obj.chicken_player.get_eggs_laid() - board_obj.chicken_enemy.get_eggs_laid()
        egg_moves_exist = any(m[1] == MoveType.EGG for m in moves)
        stall_penalty = max(0, (self.turns_since_egg - 1) * 5)

        scored: List[Tuple[float, Tuple[Direction, MoveType]]] = []
        for move in moves:
            direction, move_type = move
            new_loc = board_obj.chicken_player.get_next_loc(direction, my_loc)
            if new_loc is None:
                scored.append((-1e9, move))
                continue

            trap_risk = self.get_trapdoor_risk(new_loc)
            if trap_risk >= self.HIGH_TRAP_THRESHOLD:
                scored.append((-1e8, move))
                continue

            score = 0.0
            if move_type == MoveType.EGG:
                score += 22.0
                if egg_diff < 0:
                    score += min(24.0, abs(egg_diff) * 5.0)
                elif egg_diff == 0:
                    score += 8.0
                score += self.turns_since_egg * 4.0
            elif move_type == MoveType.TURD:
                score += 7.0
            else:
                score -= 3.0
                score -= stall_penalty
                if egg_moves_exist:
                    score -= 4.0

            risk_weight = 8.0 if move_type == MoveType.EGG else 10.0
            score -= trap_risk * risk_weight

            dist_to_center = abs(new_loc[0] - center) + abs(new_loc[1] - center)
            score += max(0.0, 3.0 - dist_to_center)

            if self.recent_positions:
                if new_loc == self.recent_positions[-1]:
                    score -= 6.0
                elif new_loc in self.recent_positions[-4:]:
                    score -= 4.0
                elif new_loc in self.recent_positions:
                    score -= 2.0
                else:
                    score += 1.0

            score += self.history_table.get(move, 0) * 0.05
            if depth in self.killer_moves and move in self.killer_moves[depth]:
                score += 5.0

            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _board_hash(self, board_obj: board.Board, maximizing: bool) -> Tuple:
        return (
            board_obj.chicken_player.get_location(),
            board_obj.chicken_enemy.get_location(),
            tuple(sorted(board_obj.eggs_player)),
            tuple(sorted(board_obj.eggs_enemy)),
            tuple(sorted(board_obj.turds_player)),
            tuple(sorted(board_obj.turds_enemy)),
            board_obj.turns_left_player,
            board_obj.turns_left_enemy,
            maximizing,
        )

    def quiescence(
        self,
        board_obj: board.Board,
        alpha: float,
        beta: float,
        maximizing: bool,
        depth: int,
        deadline: float,
    ) -> float:
        stand_pat = self.evaluate_position(board_obj)
        if time() > deadline or depth == 0:
            return stand_pat

        if maximizing:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)

        moves = board_obj.get_valid_moves() if maximizing else board_obj.get_valid_moves(enemy=True)
        tactical = [m for m in moves if m[1] != MoveType.PLAIN]
        if not tactical:
            return stand_pat

        for direction, move_type in tactical[:6]:
            if time() > deadline:
                break
            new_board = board_obj.forecast_move(direction, move_type, check_ok=False)
            if new_board is None:
                continue
            score = self.quiescence(new_board, alpha, beta, not maximizing, depth - 1, deadline)
            if maximizing:
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            else:
                beta = min(beta, score)
                if beta <= alpha:
                    break
        return alpha if maximizing else beta

    def minimax(
        self,
        board_obj: board.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        deadline: float,
    ) -> float:
        if time() > deadline:
            return self.evaluate_position(board_obj)

        tt_key = self._board_hash(board_obj, maximizing)
        if tt_key in self.transposition_table:
            tt_depth, tt_value, tt_flag, tt_alpha, tt_beta = self.transposition_table[tt_key]
            if tt_depth >= depth:
                if tt_flag == "EXACT":
                    return tt_value
                if tt_flag == "LOWER":
                    alpha = max(alpha, tt_value)
                elif tt_flag == "UPPER":
                    beta = min(beta, tt_value)
                if alpha >= beta:
                    return tt_value

        if depth == 0 or board_obj.is_game_over():
            return self.quiescence(board_obj, alpha, beta, maximizing, self.q_depth, deadline)

        orig_alpha, orig_beta = alpha, beta

        if maximizing:
            best = float("-inf")
            moves = board_obj.get_valid_moves()
            if not moves:
                return self.evaluate_position(board_obj)
            ordered = self.order_moves(board_obj, moves, depth)
            for move in ordered:
                if time() > deadline:
                    break
                direction, move_type = move
                new_board = board_obj.forecast_move(direction, move_type, check_ok=False)
                if new_board is None:
                    continue
                score = self.minimax(new_board, depth - 1, alpha, beta, False, deadline)
                if score > best:
                    best = score
                if score > alpha:
                    alpha = score
                self.history_table[move] = self.history_table.get(move, 0) + depth * depth
                if beta <= alpha:
                    self.killer_moves.setdefault(depth, [])
                    if move not in self.killer_moves[depth]:
                        self.killer_moves[depth].append(move)
                    break
        else:
            best = float("inf")
            moves = board_obj.get_valid_moves(enemy=True)
            if not moves:
                return self.evaluate_position(board_obj)
            for move in moves:
                if time() > deadline:
                    break
                direction, move_type = move
                new_board = board_obj.forecast_move(direction, move_type, check_ok=False)
                if new_board is None:
                    continue
                score = self.minimax(new_board, depth - 1, alpha, beta, True, deadline)
                if score < best:
                    best = score
                if score < beta:
                    beta = score
                if beta <= alpha:
                    break

        flag = "EXACT"
        if best <= orig_alpha:
            flag = "UPPER"
        elif best >= orig_beta:
            flag = "LOWER"

        if len(self.transposition_table) >= self.tt_size:
            self.transposition_table.pop(next(iter(self.transposition_table)))
        self.transposition_table[tt_key] = (depth, best, flag, orig_alpha, orig_beta)
        return best

    # ------------------------------------------------------------------
    # Main play
    # ------------------------------------------------------------------

    def play(
        self,
        board_obj: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[Direction, MoveType]:
        start_time = time()
        self.found_trapdoors = board_obj.found_trapdoors.copy()
        self._risk_cache = None

        self._mark_tile_safe(board_obj.chicken_player.get_location())
        self.update_trapdoor_beliefs(
            board_obj, sensor_data, board_obj.chicken_player.get_location()
        )

        moves = board_obj.get_valid_moves()
        if not moves:
            return (Direction.UP, MoveType.PLAIN)

        current_eggs = board_obj.chicken_player.get_eggs_laid()
        if current_eggs > self.last_egg_count:
            self.turns_since_egg = 0
        else:
            self.turns_since_egg += 1
        self.last_egg_count = current_eggs

        remaining_time = time_left()
        remaining_turns = board_obj.turns_left_player
        if remaining_turns <= 0:
            per_move_budget = remaining_time * self.time_safety_factor
        else:
            per_move_budget = (
                remaining_time / max(1, remaining_turns + 2)
            ) * self.time_safety_factor
        deadline = start_time + per_move_budget

        cur_loc = board_obj.chicken_player.get_location()
        self.recent_positions.append(cur_loc)
        if len(self.recent_positions) > 8:
            self.recent_positions = self.recent_positions[-8:]

        best_move = moves[0]
        best_score = float("-inf")
        achieved_depth = 1

        for depth in range(1, self.max_depth + 1):
            if time() > deadline:
                break

            current_best: Optional[Tuple[Direction, MoveType]] = None
            current_score = float("-inf")
            ordered_moves = self.order_moves(board_obj, moves, depth)

            for move in ordered_moves:
                if time() > deadline:
                    break
                direction, move_type = move
                new_board = board_obj.forecast_move(direction, move_type, check_ok=False)
                if new_board is None:
                    continue

                # Reject self-trapping egg moves early
                if move_type == MoveType.EGG and new_board.turns_left_player > 5:
                    area_after = self._reachable_area(
                        new_board, for_enemy=False, max_steps=7, planning=False
                    )
                    if area_after < self.MIN_SAFE_AREA_AFTER_EGG:
                        continue

                score = self.minimax(
                    new_board,
                    depth - 1,
                    alpha=float("-inf"),
                    beta=float("inf"),
                    maximizing=False,
                    deadline=deadline,
                )

                if score > current_score:
                    current_score = score
                    current_best = move

            if current_best is not None:
                best_move = current_best
                best_score = current_score
                achieved_depth = depth

            if best_score > 5000:
                break

        # Update recent positions with intended destination
        next_loc = board_obj.chicken_player.get_next_loc(
            best_move[0], board_obj.chicken_player.get_location()
        )
        if next_loc:
            self.recent_positions.append(next_loc)
            if len(self.recent_positions) > 8:
                self.recent_positions = self.recent_positions[-8:]

        return best_move
