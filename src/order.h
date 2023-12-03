#ifndef ORDER_H_
#define ORDER_H_

#include "types.h"
#include "board.h"

/**
 @brief Scores the movelist for move ordering purposes
 @param board position for which the movelist was generated
 @param moves the movelist to score
 @param pv_move principal variation move to order first, if any
 @param killers killer moves that caused a cutoff, if any
 */
void score_moves(const board_t *board, movelist_t *moves, move_t pv_move, move_t *killers);

// Returns the next best move
move_t next_best(movelist_t *moves, int ply);

// Prints the movescores (useful for debugging)
void movescore(const board_t *board, movelist_t *moves, int n = 5);

#endif // ORDER_H_
