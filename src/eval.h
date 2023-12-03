#ifndef EVAL_H_
#define EVAL_H_

#include "types.h"
#include "board.h"

/**
 @brief This struct scores and stores all relevant data for the evaluation of
 given board position
 */
typedef struct eval_t {
    // Game phase (0, 256)
    int phase = 0;
    // Middlegame score
    int middlegame = 0;
    // Endgame score
    int endgame = 0;
    // Tapered score
    int score = 0;

    /**
    @brief Calculates the current game phase and sets the corresponding field in eval

    Phase can range from 0 to 256 (no pieces except kings -> all pieces)
    The closer to the endgame we are, the more heavily endgame psqts
    are weighted. We use min(max(0, 1.5x - 64), 256) to scale between game phases

    @param board boards state to calculate the game phase for
    @param eval eval_t struct to store the game phase in
    */
    inline void set_phase(const board_t *board) {
        phase = CNT(board->bitboards[p] | board->bitboards[P]) << 1;
        phase += 6 * CNT(board->bitboards[k] | board->bitboards[K]);
        phase += 12 * CNT(board->bitboards[b] | board->bitboards[B]);
        phase += 18 * CNT(board->bitboards[r] | board->bitboards[R]);
        phase += 40 * CNT(board->bitboards[q] | board->bitboards[Q]);
        phase *= 3;
        phase -= 128;
        phase >>= 1;
        phase = MIN(MAX(0, phase), 256);
    }

    inline int get_tapered_score() {
        return score = (middlegame * phase + endgame * (256 - phase)) / 256;
    }

    inline void print() {
        std::cout << "Phase: " << phase \
                  << " Middlegame score: " << middlegame \
                  << " Endgame score: " << endgame \
                  << " Final score: " << score << std::endl;
    }
} eval_t;

/**
 @brief Static tapered evaluation of the current board state

 Uses: piece values, piece-square tables, passed pawns, isolated pawns,
 rook/queen on open/semi-open files, basic king safety, bishop pairs,

 @param board boards state to evaluate
 @param eval eval_t struct storing evaluation data, used for the 'eval' command
 among others
 */
int evaluate(const board_t *board, eval_t *eval);

void mirror_test(board_t *board);

//Material
extern int value_mg[PIECE_NO];
extern int value_eg[PIECE_NO];
// PSQTs
extern int pawn_table_mg[SQUARE_NO];
extern int pawn_table_eg[SQUARE_NO];
extern int knight_table_mg[SQUARE_NO];
extern int knight_table_eg[SQUARE_NO];
extern int bishop_table_mg[SQUARE_NO];
extern int bishop_table_eg[SQUARE_NO];
extern int rook_table_mg[SQUARE_NO];
extern int rook_table_eg[SQUARE_NO];
extern int queen_table_mg[SQUARE_NO];
extern int queen_table_eg[SQUARE_NO];
extern int king_table_mg[SQUARE_NO];
extern int king_table_eg[SQUARE_NO];

// Tempo score (a small bonus for the side to move)
extern int tempo_bonus_mg;
extern int tempo_bonus_eg;
// Pass and isolated pawn
extern int isolated_pawn;
// Doubled pawn penalty
extern int doubled_pawn;
// Bonus for supported pawns
extern int pawn_supported;
extern int pawn_protected_bonus;
// Indexed by rank, i.e. the closer to promoting, the higher the bonus
extern int passed_pawn[RANK_NO];
// Indexed by rank, bonus for good pawn structure
extern int pawn_bonuses[RANK_NO];
// Bonus for having two bishops on board
extern int bishop_pair_mg;
extern int bishop_pair_eg;
// Bonuses for rooks/queens on open/semi-open files
extern int rook_open_file;
extern int rook_semiopen_file;
extern int queen_open_file;
extern int queen_semiopen_file;
// Mobility weights
extern int mobility_weights[PIECE_NO];
// King safety parameters
extern int PAWN_SHIELD1_BONUS;
extern int PAWN_SHIELD2_BONUS;
extern int PAWN_STORM_PENALTY;
extern int KING_PAWN_DIST_BONUS;
extern int SAFE_PAWN_ATTACK;


#endif // EVAL_H_
