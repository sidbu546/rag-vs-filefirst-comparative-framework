# Think64 system design: who builds what

Oct 30, 2026 · @Siddhanth Kalyanaraman

## At a glance

Think64 splits into two services joined by one shared contract. Person 1 builds the chess truth service, which knows what is true on the board. Person 2 builds the coach and product, which knows the player and decides how to teach. Each person owns five components, and the only thing that blocks either of you is the shared contract you write together in week 1.

&#91;embedded content: Think64 architecture · 2 owners, 10 components, 1 shared contract\]

Every arrow between the two sides passes through the shared contract, which is why each side can be built on its own and connected later.

Open question: who takes Person 1 and who takes Person 2 between the two of us (me and Jitarth)?

## Person 2: coach and product

Person 2 owns everything the player sees and everything that decides how to teach. All of it can be built against the mock server until Person 1's API is ready.

| Component | What it does |
| --- | --- |
| Web app | Interactive board, chat panel, position setup, PGN upload and game viewer, arrows and highlights from coach answers |
| Coach agent | Turns a user question into tool calls to the chess API, builds the context, calls the LLM, sends every answer to `/verify` before showing it |
| Player memory | Rating, experience level, preferred explanation depth, recurring mistakes, past sessions |
| LLM adapter | One wrapper around the model provider with structured JSON output, so the model can be swapped |
| Evaluation harness | A fixed set of test positions and questions, scoring answers for chess correctness and level-appropriate explanation |

## Shared contract: analysis-schema

We write this together in week 1, and it is the only thing either side waits on. Once it is fixed, Person 2 builds against the mock and Person 1's tests check that real output matches the same examples.

It contains:

- **Types** for Position, Move, CandidateLine, EngineAnalysis, ChessFact, VerifyRequest/Result, PlayerProfile, CoachRequest and CoachResponse.
- **5–10 real example responses**, including a mate, a pin, a stalemate and a game with threefold repetition.
- **A mock server** that returns those examples, so Person 2 is never blocked.
- **Contract tests** that both sides run against the same examples.

Decisions the contract must settle:

- Move notation: UCI for input, both SAN and UCI in every response.
- Evaluation format: `{type: "cp" | "mate", value, pov: "white"}`, so mate scores and point of view are explicit.
- Game history: requests carry the move list or a game ID, not only a FEN, so repetition and draws can be detected.
- Engine timing: whether analysis is synchronous, streamed or polled, and which depth and MultiPV parameters the agent may set per player level.

## Person 1: chess truth service

Person 1 owns everything that decides what is objectively true on the board. It must work fully without any LLM.

| Component | What it does |
| --- | --- |
| Chess API | The endpoints both sides agreed on: analyze, move, compare, verify, game |
| Rules core | Legal moves, FEN and PGN parsing, the move tree with variations, check, mate, stalemate and draw detection |
| Engine service | Runs Stockfish with depth or time limits and MultiPV, returns evaluations and principal variations, caches results |
| Facts and verifier | Attackers, defenders, pins, x-rays, checks and material; checks claims from the coach, such as "the knight on f3 is pinned" |
| Line comparison | Compares candidate lines and returns structured differences, such as material won or a defender removed |

## One owner for every overlap

Each place where the two sides touch has exactly one owner, so neither of us builds the same thing twice.

| Overlap | Owner | Rule |
| --- | --- | --- |
| Move legality | Person 1 | Person 2's board uses chess.js only for instant drag feedback; the rules core is the final authority on every move |
| Move tree and PGN | Person 1 | Person 1 parses games and stores variations; Person 2 only displays and navigates what the API returns |
| Claim verification | Person 1 builds, Person 2 calls | The agent calls `/verify` on every LLM answer before the user sees it |
| Line comparison | Person 1 computes, Person 2 explains | Person 1 returns structured differences; Person 2 decides how to explain them |
| Concept attribution | Both, later | A joint milestone, since it needs both engine evidence and coaching judgment |

## Working rhythm

We build separately but connect weekly, so mismatches surface early instead of all at once at the end.

1. **Week 1, together:** write the shared contract, example JSON and mock server.
2. **Weeks 2–3, separately:** Person 1 gets FEN → Stockfish → top 3 moves working; Person 2 gets board → chat → mock analysis → LLM explanation working.
3. **First joint demo:** swap the mock for the real API and run one loop end to end: load a position, analyze it, get a grounded explanation.
4. **Then weekly:** reconnect every week as new endpoints land (what-if moves, PGN review, verification).
5. **Later, together:** concept attribution.
