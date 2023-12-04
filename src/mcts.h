// Core MCTS Structures like Nodes etc.
#include "types.h"
#include "board.h"

// For clarity, we alias the names
typedef board_t State;
typedef move_t Action;

typedef struct Node Node;

/**
 * Find a region of interest in the tree, insert a leaf node, and return.
 */
Node *insert_node_with_tree_policy(Node *root, State *s);


/**
 * Perform the Monte Carlo Rollout:
 * Simulate the game from the state of 'node' to the end, and return the
 * terminal state reward.
 */
double rollout(Node* node, State *s);


/**
 * Propogate the reward information backward along the path from node_i to the
 * root, updating the utilities for all nodes on the path.
 */
void backprop(int reward, Node* node);


/**
 * The MCTS search function
 */
void MCTS_Search(board_t* board, searchinfo_t *info);
