/* Iterative deepening alpha-beta in negamax fashion */
#include "search.h"

#include <cmath>
#include <iomanip>
#include <algorithm>

#include "eval.h"
#include "threads.h"
#include "order.h"

// Global evaluation struct (for multithreaded, we'll want to have a separate one for
// each thread)
eval_t eval;


namespace {

// For maintaining the principal variation in the triangular array
// Copies up to n moves from p_src to p_tgt, kind of like memcpy
// Adapted from https://www.chessprogramming.org/Triangular_PV-Table
void movcpy(move_t *p_tgt, move_t *p_src, int n) {
    while (n-- && (*p_tgt++ = *p_src++));
}

/* Principal Variation

Triangular table layout:

ply  maxLengthPV
    +--------------------------------------------+
0   |N                                           |
    +------------------------------------------+-+
1   |N-1                                       |
    +----------------------------------------+-+
2   |N-2                                     |
    +--------------------------------------+-+
3   |N-3                                   |
    +------------------------------------+-+
4   |N-4                                 |
...                        /
N-4 |4      |
    +-----+-+
N-3 |3    |
    +---+-+
N-2 |2  |
    +-+-+
N-1 |1|
    +-+
*/

typedef struct pv_line {
    move_t moves[MAX_DEPTH] = {};
    size_t size = 0;
    void clear() { last = moves; size = 0; memset(moves, 0, sizeof(moves)); }

    move_t operator[](int i) const { return moves[i]; }
    move_t& operator[](int i)      { return moves[i]; }

    // Print the principal variation line
    void print() const {
        for (size_t i = 0; i < size; ++i) {
            std::cout << move_to_str(moves[i]) << " ";
        }
    }

    private:
        move_t *last = moves;
} pv_line;

// Global PV table (quadratic approach)
// - indexed by [ply]
// - pv[ply] is the principal variation line for the search at depth 'ply'
pv_line pv_tb[MAX_DEPTH+1];

// TODO: Use Unicode chars in source code? Compiler compatibility?
// int α = alpha;
// int β = beta;

/**
 @brief Alpha-Beta search in negamax fashion.
 @param alpha the lowerbound
 @param beta the upperbound
 @param board the board position to search
 @param info search info: time, depth to search, etc.
 @param pv reference to a table storing the (depth - 1) principal variation
*/
int negamax(int α, int β, int depth, board_t *board, searchinfo_t *info, stack_t *stack) {
    assert(check(board));
    assert(α < β);
    assert(depth >= 0);

    // PV for the current search ply
    pv_line &pv = pv_tb[board->ply];
    // PV for the next search ply
    pv_line &next_pv = pv_tb[board->ply + 1];

    // Set principal variation line size for the current search ply
    pv.size = board->ply;

    /* Recursion base case */
    if (depth <= 0) {
        return quiescence(α, β, board, info, stack);
    }

    ++info->nodes;

    // If not at root of the search, check for repetitions
    if (board->ply && (is_repetition(board) || board->fifty_move >= 100)) {
        //return 0;
        // Randomized draw score
        return -2 + (info->nodes & 0x3);
    }

    // Are we too deep into the search tree?
    if (board->ply >= MAX_DEPTH - 1) {
        return evaluate(board, &eval);
    }

    int score = -oo;

    /* Get a static evaluation of the current position */
    stack[board->ply].score = score = evaluate(board, &eval);


    /* Move generation, ordering, and move loop */

    // Generate pseudolegal moves
    movelist_t moves;
    generate_moves(board, &moves);

    // If following the principal variation (from a previous search at a smaller
    // depth), order the PV move higher
    score_moves(board, &moves, NULLMV, stack[board->ply].killer);

    int moves_searched = 0;
    int bestscore = score = -oo;

    // Iterate over the pseudolegal moves in the current position
    // for (const auto& move : moves) {
    move_t move;
    move_t bestmove = NULLMV;
    while ((move = next_best(&moves, board->ply)) != NULLMV) {

        // Pseudo-legal move generation
        if (!make_move(board, move))
            continue;

        score = -negamax(-β, -α, depth - 1, board, info, stack);
        undo_move(board, move);

        if (search_stopped(info))
            return 0;

        ++moves_searched;

        assert(info->state == ENGINE_SEARCHING);

        if (score > bestscore) {
            bestscore = score;
            bestmove = move;
            // Check if PV or fail-high node
            if (score > α) {
                if (score >= β) { // Fail-high node
                    if (moves_searched == 1) {
                        info->fail_high_first++;
                    }
                    info->fail_high++;

                    /* The move caused a beta cutoff, hence we get a lowerbound score */
                    return β;
                }

                /* Otherwise if no fail-high occured but we beat alpha, we are in a PV node */

                // Update the PV
                pv[board->ply] = bestmove;
                movcpy(&pv[board->ply + 1], &next_pv[board->ply + 1], next_pv.size);
                pv.size = next_pv.size;

                // Update the search window lowerbound
                α = score;
            }
        }
        /* The move failed low */
    }

    // If no legal moves could be performed, then check if we're in check:
    // if not, it's a stalemate. Otherwise we've been mated!
    if (!moves_searched) {
                                                 // Mate score      // Stalemate
        return is_in_check(board, board->turn) ? -oo + board->ply : 0;
    }

    assert(check(board));

    return α;
}

inline void print_search_info(int s, int d, int sd, uint64_t n, uint64_t t,
                              const pv_line &pv, [[maybe_unused]] board_t *board) {

  // Print the info line
  std::cout << "info depth " << d << " seldepth " << sd \
            << " score ";

  // Print mate distance info if a player is being mated
  if (std::abs(s) >= +oo - MAX_DEPTH) {
     std::cout << "mate " \
          << (s > 0 ? +oo - s + 1 : -oo - s + 1) / 2;
  } else {
      std::cout << "cp " << s;
  }
  std::cout << " nodes " << n << " time " << t \
            << " pv ";

  pv.print();
  std::cout << std::endl;
}


void init_search(board_t *board, searchinfo_t *info, stack_t *s) {

    // Scale tables used for the history heuristic
    for (piece_t p = NO_PIECE; p < PIECE_NO; ++p) {
        for (square_t sq = A1; sq <= H8; ++sq) {
            for (int colour : {BLACK, WHITE}) {
                board->history_h[colour][p][sq] /= 16;
            }
        }
    }

    // Clear the global pv table
    for (int i = 0; i < MAX_DEPTH; ++i) {
        pv_tb[i].clear();
    }

    // Clear search info, like # nodes searched
    info->clear();

    // Clear the search stack
    // - killers
    // - scores
    for (int i = 0; i < MAX_DEPTH; ++i) {
        s->killer[0] = s->killer[1] = NULLMV;
        s->score = 0;
    }

    // The ply at the root of the search is 0
    board->ply = 0;
}

} // namespace


/**
 @brief Quiescence search - we only search 'quiet' (non-tactical)
 positions to get a reliable score from our static evaluation function
 @param alpha the lowerbound
 @param beta the upperbound
 @param board the board position to search
 @param info search info: time, depth to search, etc.
 @param stack the search stack
*/
int quiescence(int α, int β, board_t *board, searchinfo_t *info, stack_t *stack) {
    assert(check(board));
    assert(α < β);

    ++info->nodes;

    //int pv_node = α + 1 < β;

    if (board->ply > info->seldepth)
        info->seldepth = board->ply - 1;

    int score = -oo;

    /* Stand-pat score */
    stack[board->ply].score = score = evaluate(board, &eval);

    assert(-oo < score && score < +oo);

    // Are we too deep into the search tree?
    if (board->ply >= MAX_DEPTH - 1) {
        return score;
    }

    if (score >= β) { // fail-high
        return β;
    }

    if (score > α) { // PV-node
        α = score;
    }

    movelist_t noisy;
    generate_noisy(board, &noisy);

    // Move ordering
    score_moves(board, &noisy, NULLMV, nullptr);

    #ifdef DEBUG
    int moves_searched = 0;
    #endif

    // Iterate over the pseudolegal moves in the current position
    move_t move = NULLMV;
    while ((move = next_best(&noisy, board->ply)) != NULLMV) {

        // Pseudo-legal move generation
        if (!make_move(board, move))
            continue;

        #ifdef DEBUG
        ++moves_searched;
        #endif
        score = -quiescence(-β, -α, board, info, stack);

        undo_move(board, move);

        if (search_stopped(info)) {
            return 0;
        }

        if (score >= β) { // fail-high
            #ifdef DEBUG
            if (moves_searched == 1) {
                info->fail_high_first++;
            }
            info->fail_high++;
            #endif
            return β;
        }

        if (score > α) { // PV-node
            α = score;
        }
    }

    return α;
}

/* Search the tree starting from the root node (current board state) */
void search(board_t *board, searchinfo_t *info) {
    assert(check(board));

    move_t best_move = NULLMV;
    int best_score = 0;
    stack_t stack[MAX_DEPTH+1] = {};

    // Clear for search
    init_search(board, info, stack);

    int curr_depth_nodes = 0;
    int curr_depth_time = 0;

    /*
    std::cout << "Starting search: ";
    std::cout << "time allocated: " << info->end - now();
    std::cout << " time start: " << info->start;
    std::cout << " time end: " << info->end << std::endl;
    */

    // Iterative deepening
    for (int depth = 1; depth <= info->depth; ++depth) {
        // For calculating the branching factor
        curr_depth_nodes = info->nodes;

        // For time management
        curr_depth_time = now();

        stack[0].score = best_score = negamax(-oo, +oo, depth, board, info, stack);

        curr_depth_nodes = info->nodes - curr_depth_nodes;
        curr_depth_time = now() - curr_depth_time;

        if (search_stopped(info)) {
            break;
        }

        assert(info->state == ENGINE_SEARCHING);

        best_move = pv_tb[0][0];

        print_search_info(best_score,
                          depth,
                          info->seldepth,
                          info->nodes,
                          now() - info->start,
                          pv_tb[0], board);

        LOG("info string depth " << depth \
            << std::setprecision(4) \
            << " branchf " << std::pow(curr_depth_nodes, 1.0/depth) \
            << std::setprecision(2) \
            << " ordering " << (static_cast<double>(info->fail_high_first) / info->fail_high) \
        );

        // We try to estimate if we have enough time to search the next depth,
        // and if not, we cut the search short to not waste the time
        // REVIEW: Need to tune the coefficient here
        //if (info->time_set && 3.5 * (now() - info->start) >= info->end - info->start) {
            //std::cout << "info string Engine won't have enough time to search the next depth!\n";
            //break;
        //}
    }

    std::cout << "bestmove " << move_to_str(best_move) << std::endl;

    assert(check(board));

    // After the search is stopped, the thread sets the status to stopped
    info->state = ENGINE_STOPPED;
}
