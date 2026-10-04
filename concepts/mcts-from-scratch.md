---
title: Monte Carlo tree search (UCT) from scratch (tic-tac-toe vs random and perfect play)
category: concepts
tags: [mcts, monte-carlo-tree-search, uct, ucb, game-tree-search, planning, reinforcement-learning, alphazero, minimax, python, from-scratch, ml-basics]
use_cases:
  - "implement Monte Carlo tree search (selection, expansion, rollout, backpropagation) in plain Python"
  - "see how MCTS strength grows with the simulation budget against a random and a perfect opponent"
  - "explain how UCT applies the UCB1 bandit rule at every tree node"
  - "explain how AlphaZero replaces random rollouts with a policy/value network"
status: draft
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1007/11871842_29
  - https://doi.org/10.1109/TCIAIG.2012.2186810
  - https://doi.org/10.1126/science.aar6404
---

# Monte Carlo tree search (UCT) from scratch (tic-tac-toe vs random and perfect play)

## Summary
Monte Carlo tree search (MCTS) picks a move by growing a search tree from the current position, one simulation at a time. Each simulation walks down the tree using a bandit rule, adds one new node, plays random moves to the end of the game, and credits the result to every node on the path. It needs no evaluation function, only a simulator of the rules, which is why it took over Go before deep learning and is the search inside AlphaZero. Below, UCT (MCTS with the UCB1 rule) plays 100 games of tic-tac-toe per row. Against a random opponent, 10 simulations per move already win 72 games, and 1,000 win 95 with no losses. Against a perfect minimax opponent, where the best result is a draw, it goes from 21 draws at 10 simulations to 97 at 1,000.

## Key concepts
- **The four steps.** *Selection*: from the root, repeatedly go to the child with the highest UCT score until reaching a node with untried moves. *Expansion*: add one child for an untried move. *Simulation (rollout)*: play random moves to a terminal state. *Backpropagation*: add 1 visit to every node on the path, and add the result (1 win, 0.5 draw, 0 loss) from the point of view of the player who made the move into that node.
- **UCT score.** `W/N + c·√(ln N_parent / N)`. This is UCB1 from [[thompson-sampling-and-ucb-from-scratch]], treating each node's children as arms of a bandit. The first term exploits moves that win, and the second explores moves that have few visits. `c = √2 ≈ 1.4` is the textbook value for rewards in [0, 1].
- **Perspective flips.** In a two-player game a node's value belongs to the player who moved into it, so the parent picks the child that is best for the player to move. Getting this sign wrong is the most common bug: the agent then plays for its opponent.
- **Final move choice.** Play the most-visited root child, not the one with the best mean. Visit counts are more robust, because a move with 3 lucky wins out of 3 has a perfect mean but almost no evidence.
- **Anytime algorithm.** Stop whenever the time budget runs out; more simulations give a better estimate. The tree can be kept between moves by reusing the subtree under the move that was played.

## When to use / scenarios
- Learning: the shortest path from bandits to planning, and the search half of AlphaZero ([[reinforcement-learning]]).
- Interviews: "explain MCTS", "how does AlphaGo choose a move", "UCT vs minimax", "why play the most-visited move".
- Practice: board and card games, puzzle solvers, scheduling and combinatorial planning where a simulator exists but a good heuristic does not, and tree search over LLM reasoning steps (each node a partial solution, rollouts scored by a verifier or reward model: [[reasoning-models]]).
- Not for: problems with a cheap, accurate evaluation function and a small branching factor, where alpha-beta minimax is stronger per CPU second (classic chess engines); problems without a simulator; or single-step decisions (use a bandit).

## Setup & code
Standard library only. The whole table takes about 8 seconds.

```python
import math
import random
from functools import lru_cache

random.seed(0)
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner(b):
    for i, j, k in LINES:
        if b[i] != "." and b[i] == b[j] == b[k]:
            return b[i]
    return "draw" if "." not in b else None


def moves(b):
    return [i for i, c in enumerate(b) if c == "."]


def play(b, m, p):
    return b[:m] + p + b[m + 1:]


other = {"X": "O", "O": "X"}


@lru_cache(maxsize=None)
def minimax(b, p):
    """Value of board b for player p to move: +1 win, 0 draw, -1 loss."""
    w = winner(b)
    if w:
        return 0 if w == "draw" else (1 if w == p else -1)
    return max(-minimax(play(b, m, p), other[p]) for m in moves(b))


def perfect_move(b, p):
    best = max(-minimax(play(b, m, p), other[p]) for m in moves(b))
    return random.choice([m for m in moves(b) if -minimax(play(b, m, p), other[p]) == best])


class Node:
    def __init__(self, board, to_move, parent=None, move=None):
        self.board, self.to_move, self.parent, self.move = board, to_move, parent, move
        self.children, self.untried = [], moves(board) if not winner(board) else []
        self.N, self.W = 0, 0.0          # visits, wins for the player who moved INTO this node


def uct_select(node, c):
    return max(node.children, key=lambda ch: ch.W / ch.N + c * math.sqrt(math.log(node.N) / ch.N))


def mcts(board, to_move, iters, c=1.4):
    root = Node(board, to_move)
    for _ in range(iters):
        node = root
        while not node.untried and node.children:                   # 1. selection
            node = uct_select(node, c)
        if node.untried:                                            # 2. expansion
            m = node.untried.pop(random.randrange(len(node.untried)))
            child = Node(play(node.board, m, node.to_move), other[node.to_move], node, m)
            node.children.append(child)
            node = child
        b, p = node.board, node.to_move                             # 3. random rollout
        while not winner(b):
            b, p = play(b, random.choice(moves(b)), p), other[p]
        w = winner(b)
        while node:                                                 # 4. backpropagation
            node.N += 1
            mover = other[node.to_move]                             # player who made node.move
            node.W += 1.0 if w == mover else 0.5 if w == "draw" else 0.0
            node = node.parent
    return max(root.children, key=lambda ch: ch.N).move            # most-visited, not best mean


def game(mcts_player, opponent, iters):
    b, p = "." * 9, "X"
    while not winner(b):
        if p == mcts_player:
            m = mcts(b, p, iters)
        else:
            m = random.choice(moves(b)) if opponent == "random" else perfect_move(b, p)
        b, p = play(b, m, p), other[p]
    w = winner(b)
    return "draw" if w == "draw" else ("win" if w == mcts_player else "loss")


GAMES = 100
print(f"MCTS (UCT, c=1.4) at tic-tac-toe, {GAMES} games per row, MCTS alternates X/O")
print("opponent   iters   win  draw  loss")
for opp in ["random", "perfect"]:
    for iters in [10, 50, 200, 1000]:
        r = {"win": 0, "draw": 0, "loss": 0}
        for g in range(GAMES):
            r[game("X" if g % 2 == 0 else "O", opp, iters)] += 1
        print(f"{opp:<9} {iters:>6}  {r['win']:>4}  {r['draw']:>4}  {r['loss']:>4}")
```

Output (Python 3.14):
```
MCTS (UCT, c=1.4) at tic-tac-toe, 100 games per row, MCTS alternates X/O
opponent   iters   win  draw  loss
random        10    72    12    16
random        50    91     7     2
random       200    92     7     1
random      1000    95     5     0
perfect       10     0    21    79
perfect       50     0    66    34
perfect      200     0    73    27
perfect     1000     0    97     3
```

Tic-tac-toe has 9 opening moves, so 10 simulations barely visit each one once and the agent is close to random. It still beats a random player most of the time, because rollouts already favour moves that complete or block a line. The perfect opponent exposes the remaining errors: every loss is a missed block that only appears two plies deep. Going from 200 to 1,000 simulations cuts losses from 27 to 3, which is the usual MCTS profile of steady gains with log-scale compute. A full minimax search is cheap here (the game has 5,478 legal positions), which is why MCTS matters for games like Go, where the tree is far too large to enumerate.

## Choosing / trade-offs
- **UCT vs alpha-beta minimax.** Minimax needs a heuristic evaluation at the depth cutoff and prunes well only with good move ordering. MCTS needs only legal moves and terminal outcomes, and it spends its budget unevenly on the promising lines. With a strong evaluation function and a branching factor of about 35 (chess), alpha-beta was long stronger. With a branching factor of about 250 and no good evaluation (Go), MCTS won.
- **Random rollouts vs a value network.** Random rollouts are unbiased but noisy and can be badly wrong in tactical positions where only one move survives. AlphaZero drops rollouts entirely: a network gives a value estimate at the leaf and a policy prior P that enters the selection rule (PUCT: `Q + c·P·√N_parent / (1 + N)`).
- **Exploration constant c.** Larger c explores more evenly and is safer with few simulations; smaller c goes deeper into the best line. Tune it per game together with the budget.
- **Enhancements.** Transpositions (share nodes for identical positions reached by different move orders), RAVE/AMAF (share move statistics across the tree in early iterations), progressive widening (for huge or continuous action spaces) and tree reuse between moves.
- **Libraries.** OpenSpiel has MCTS and AlphaZero for many games; `mctx` (JAX) implements batched MuZero/AlphaZero-style search on accelerators.

## Gotchas
- Store wins from the perspective of the player who moved into the node. Storing them for a fixed player, or for the player to move, makes the selection maximize the opponent's result.
- Do not pick the final move by best mean value. Use visit count (or require both the best mean and the most visits).
- `ln(N_parent) / N` divides by zero for unvisited children. Expand all children before UCT selection applies, or give unvisited children an infinite score.
- Random rollouts mislead in "only move" positions (a forced block). Light domain knowledge in the rollout policy, such as "win if you can, block if you must", often matters more than more simulations.
- In stochastic or imperfect-information games, plain MCTS over the true state "cheats" by seeing hidden information. Use chance nodes for dice and information-set MCTS for hidden cards.
- Python is slow for rollouts. Real engines use bitboards, compiled code or batched GPU evaluation; budget in simulations per second before choosing c and depth.

## Related
- [[thompson-sampling-and-ucb-from-scratch]] - the UCB1 bandit rule that UCT applies at every node.
- [[reinforcement-learning]] - planning vs model-free learning, AlphaZero and MuZero.
- [[q-learning-from-scratch]] - learning values from experience instead of searching with a simulator.
- [[ppo-from-scratch]] - the policy-gradient side of game-playing agents.
- [[reasoning-models]] - tree search over reasoning steps with verifiers.

## References
- Kocsis and Szepesvári (2006), "Bandit Based Monte-Carlo Planning" (UCT), ECML: https://doi.org/10.1007/11871842_29
- Browne et al. (2012), "A Survey of Monte Carlo Tree Search Methods", IEEE TCIAIG: https://doi.org/10.1109/TCIAIG.2012.2186810
- Silver et al. (2018), "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" (AlphaZero), Science: https://doi.org/10.1126/science.aar6404
