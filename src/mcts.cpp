#include "mcts.h"

#include <random>
#include <vector>
#include <cmath>
#include <climits>

#include "eval.h"
#include "threads.h"
#include "sgd.h"
#include "board.h"
#include "arena.h"
#include "categorical.h"

// Global evaluator
extern eval_t eval;

// Constants (TODO: Tune with self-play?)
constexpr double UCB_CONST = 2.7;
constexpr int ROLLOUT_BUDGET = 3;
constexpr size_t DEFAULT_ARENA_MB = 2048; /* Default size of the arena in MB */

// Arena allocator 
Arena arena(DEFAULT_ARENA_MB);

class Node {

    // Search should have access to all private members
    friend void MCTS_Search(board_t* board, searchinfo_t *info);

public:
    // Constructor
    Node(const board_t* board, move_t mv, Node* parent_node)
        : parent(parent_node)
        , a(mv)
        , total_reward(0)
        , avg(0)
        , visits(0)
    {
        // Generate all possible actions (chess moves) from this node
        generate_moves(board, &this->untried_moves);
    }

    // Destructor
    ~Node() {
        for (Node* child : this->children) {
            child->~Node();
        }
    }

    Node* insert_child(move_t mv, const board_t* board);
    Node* best_child(bool exploration_mode = true);
    double UCB(bool exploration_mode = true);
    void update(double res); // backprop update (increment visits etc.)
    inline bool is_terminal() {
        return children.size() == 0 && is_fully_expanded();
    }

    inline bool is_fully_expanded() {
        // REVIEW: Are leaves considered fully expanded?
        return this->untried_moves.size() == 0;
    }

    Node *parent;
    std::vector<Node *> children;
    // action that got us to this node (for performance reasons only the root
    // stores the actual board state)
    Action a;
    movelist_t untried_moves;
private:
    double total_reward;
    double avg;
    int visits;
};


/* TODO:
std::ostream& operator << (std::ostream &o, const Node* node) {
    return o << "Node; " << node->children.size() << " children; " \
    << node->visits << " visits; " << node->total_reward << " reward";
}
*/


// REVIEW: Here, we could experiment with multiple rollout policies
// and report the results?

[[__always_inline__]] 
static inline Action random_policy(movelist_t& actions, State *s = NULL) {
    (void) s; // Ignore the state if our policy is random
    return actions[rand_uint64() % actions.size()];
}

static inline Action evaluation_based_policy(movelist_t& actions, State* s) {

    // Get an evaluation score for each child and treat it as a weight
    std::vector<double> weights;
    weights.reserve(actions.size());
    for (const move_t move : actions) {
        if (!make_move(s, move)) { // Pseudolegal move generation
            weights.push_back(0.0);
            continue;
        }
        // NOTE: After making the move, the evaluation score will be w.r.t.
        // the opponent!
        double weight = (1.0 - winning_prob(evaluate(s, &eval)));
        weights.push_back(100 * (weight * weight * weight));
        undo_move(s); 
    }
    LOG("Categorical weights:");
    for (size_t i = 0; i < weights.size(); ++i) {
        LOG(move_to_str(actions[i]) << ": " << weights[i]);
    }

    // Sample from a categorical distribution
    std::default_random_engine generator;
    fast_discrete_distribution<int> distribution(weights);
    size_t sampled = distribution(generator);
    LOG("Sampled move " << move_to_str(actions[sampled]));
    return actions[sampled];
}


/**
 * @brief Given a node, select a *legal* action according to policy,
 * perform the action, and insert a child node with the resulting state
 * 
 * @return Node* or nullptr if no legal action could be taken
 */
Node* select_and_insert(Node* node, State* s, Action (*policy)(movelist_t&, State*)) {

    assert(node != nullptr);
    assert(s != nullptr);
    assert(!node->is_terminal());

    // We sample actions until we find a valid one
    LOG("Board before making move:");
    print(s);
    Action a = policy(node->untried_moves, s);
    LOG("Making move " << move_to_str(a));
    LOG("Remaining moves are: ");
    print_moves(node->untried_moves);
    while (!make_move(s, a)) {
        // Remove the action from untried_moves, as it is illegal
        node->untried_moves.erase(node->untried_moves.find(a));
        if (node->untried_moves.size() == 0) { /* Edge case: if ran out of moves */
            LOG("We ran out of moves in this state!");
            return nullptr; 
        } 
        a = policy(node->untried_moves, s);
        LOG("Failed! Making move " << move_to_str(a));
    }
    node = node->insert_child(a, s);
    LOG("Board after making move:");
    print(s);
    return node;
}

/**
 * Returns an action to play during rollout.
 * This could be parametrized w.r.t the current state? NN?
 * For Pure MCTS we use random rollouts
*/
inline Action rollout_policy(movelist_t& actions) {
    return random_policy(actions);
}

// Ignore
double rollout(Node *node, State *s) {

    assert(node != nullptr);
    assert(s != nullptr);

    // We limit the number of rollouts (tree height)
    int budget = ROLLOUT_BUDGET;
    while (!node->is_terminal() && budget --> 0) {
        // Note: select_and_insert can return a nullptr
        Node *child = select_and_insert(node, s, &random_policy);
        if (child == nullptr) {
            break;
        }
        node = child;
    }

    /* Leaf state's reward */

    // 1) If terminal, check who won the rollout 
    if (node->is_terminal()) {
        // If side to turn (us?) is in check and node is terminal (no moves),
        // we've been mated (we get a centipawn score of negative infinity,
        // equivalent to a zero probability of winning)
        if (is_in_check(s, s->turn)) {
            return -oo;
        
        // REVIEW: Will this condition *ever* hold?
        } else if (is_in_check(s, s->turn ^ 1)) { // if opponent got mated
            return +oo;
        } else {
            return 0; // stalemate (i.e. draw)
        }
    }

    // 2) Otherwise, use the static evaluation function as a heuristic
    // Note: We convert this into a winning probability estimate with sigmoid
    int static_eval_score = evaluate(s, &eval);
    return winning_prob(static_eval_score);
}


/** 
 * @brief Backpropagate the result of a playout up to the root of the tree
 * @param double reward achieved during last rollout
 * @param Node* Pointer to the selected node from which the rollout was performed
 * @param int Color of the root player
 * @param int Color of the node player
 */
void backprop(double reward, Node *node, int root_color, int color) {
    assert(node != nullptr);

    // Same color: +
    // Diff color: -

    // Flip the reward if node is not the side to move
    // reward *= 2*(color == root_color)-1;

    Node *curr = node;
    while (curr != nullptr) {
        reward *= -1.0;
        curr->update(reward);
        curr = curr->parent;
    }
}

double Node::UCB(bool exploration_mode) {
    double ucb = static_cast<double>(total_reward) / (visits + 1);
    // double ucb = this->avg;
    if (exploration_mode)
        // Avoid div-by-zero
        ucb += UCB_CONST * std::sqrt(std::log(parent->visits) / (visits + 1));

    return ucb;
}


Node *Node::best_child(bool exploration_mode) {
    // Calculate UCB values for all the children and pick the highest
    double ucb;
    double best_value = static_cast<double>(INT_MIN);
    Node *best = nullptr;

    for (Node* child : this->children) {
        ucb = child->UCB(exploration_mode);
        if (ucb > best_value) {
            best_value = ucb;
            best = child;
        }
    }
    assert(best != nullptr);

    // REVIEW: Randomly determine ties between best children?
    // Note: We'll rarely run into a scenario where two children have
    // the same UCB score due to floating point imprecision
    return best;
}

Node *Node::insert_child(move_t move, const board_t *board) {
    // Node *child = new Node(board, move, this);
    void *memory = arena.allocate(sizeof(Node));
    Node *child = memory ? new (memory) Node(board, move, this) : nullptr;

    // Mark move as tried
    for (auto it = this->untried_moves.begin(); it != this->untried_moves.end(); ++it) {
        if (*it == move) {
            untried_moves.erase(it);
            break;
        }
    }

    // Store the child within the node
    this->children.push_back(child);
    return child;
}

void Node::update(double reward) {

    // See 184 Lecture slides on AlphaZero
    /*
    double n = this->visits;
    this->avg = (n/(n+1)) * this->avg + (1.0 / (n+1)) * reward;
    */
    this->visits++;
    this->total_reward += reward;
}


/*
To avoid local optima, we can use some parametrized policy
to pick the most 'interesting' areas of the tree to expand
// See: https://xyzml.medium.com/learn-ai-game-playing-algorithm-part-ii-monte-carlo-tree-search-2113896d6072

(Could it be e.g. a Thompson distribution or sth?)

For now, we use a random policy lol
*/

Action prior_prob(movelist_t& actions) {
    return random_policy(actions);
}


// TODO: Review this for correctness
Node *insert_node_with_tree_policy(Node *root, State *s) {
    assert(root != nullptr);

    Node *node = root;
    while (!node->is_terminal()) {
        LOG("At node " << to_fen(s) << " @ " << node);
        if (node->is_fully_expanded()) {
            LOG("Node fully expanded!");
            node = node->best_child();
            LOG("Best child is " << move_to_str(node->a) << " @ " << node);
            /* Make sure the state follows the path along the tree as well */
            if (make_move(s, node->a) == false) {
                LOG("Node " << s << " stores invalid child\n");
                *((char*) 0) = 0;
            } 
        } else {
            LOG("Inserting new child");
            Node *child = select_and_insert(node, s, &random_policy);
            return child;
        }
    }
    return node;
}

/**
 * @brief Given the MCTS root and current board state, find a node 
 * within the tree to expand
 * @todo We might want to exploit more here instead of fully expanding each 
 * node
 * 
 * @param root Root of the game tree
 * @param s Board state at the root
 * @return Node* The node selected for expansion
 */
Node *select(Node *root, State *s) {
    assert(root != nullptr);
    assert(s != nullptr);

    Node *node = root;
    while (!node->is_terminal()) {
        if (node->is_fully_expanded()) {
            node = node->best_child(true);
            /* Make sure the state follows the path along the tree as well */
            make_move(s, node->a);
        } else {
            return node;
        }
    }
    return node;
}

/**
 * @brief Given a node and corresponding state, find and play
 * a legal action according to policy. Mutates the board state @param s
 * @param s Board state corresponding to @param node
 * @param policy A policy function 
 * @param moves Allowed moves
 * @return Action played on success, NULLMV otherwise.
 */
Action play_legal(State *s, Action (*policy)(movelist_t&, State*), movelist_t& moves) {

    // Pick a move according to policy 
    // (REVIEW: We might want the policy to take in the state too?)
    Action a = policy(moves, s);
    while (!make_move(s, a)) {
        // Remove the action from the list, as it is illegal
        moves.erase(moves.find(a));
        if (moves.size() <= 0) {
            // Ran out of moves -> can't play any legal action from state s
            return NULLMV;
        }
        a = policy(moves, s);
    }
    return a;
}

/**
 * @brief Attempts to expand node. If successful, returns the new child,
 * and otherwise the input node.
 * @param node Pointer to a node to be expanded
 * @param s Current board state corresponding to @param node
 * @param info Search information including # of nodes created in tree
 * @return Node* New child if successful, @param node otherwise
 */
Node *expand(Node *node, State *s, searchinfo_t *info) {
    assert(node != nullptr);
    assert(s != nullptr);

    // REVIEW: Redundant is_fully_expanded check?
    if (node->is_terminal() || node->is_fully_expanded())
        return node;

    // Check if can expand:
    if (!arena.has_space(sizeof(Node))) {
        LOG("Arena ran out of space!\n");
        return node;
    }

    // Attempt to expand the node (note: might mutate s)
    Action a = play_legal(s, &random_policy, node->untried_moves);
    if (a != NULLMV) {
        ++info->nodes;
        info->seldepth = std::max(info->seldepth, s->ply);
        return node->insert_child(a, s);
    }

    return node;
}

/**
 * @brief Perform a light rollout simulation (playout).
 *  Like rollout(), but doesn't insert any new nodes into the tree
 * @param node Node to start the playout from
 * @param s Board state corresponding to @param node
 * @param int The root player's color (WHITE or BLACK)
 * @return double The reward, r \in [0, 1] for the side to move in state s
 */
double simulate(State *s) {
    assert(s != nullptr);

    // We'll return the reward for the player to move in state s
    int color = s->turn;

    // Perform rollout according to chosen policy (we use a random one for now)
    Action a;
    movelist_t moves;
    int budget = ROLLOUT_BUDGET;
    do {
        generate_moves(s, &moves);
        a = play_legal(s, &random_policy, moves);
    } while (a != NULLMV && budget --> 0);

    // 1) If terminal, check who won the rollout
    if (a == NULLMV) {
        // If after rollout root player is in check and node is terminal (no moves),
        // we've been mated        
        if (is_in_check(s, color)) {
            return -1; 
        } else if (is_in_check(s, color ^ 1)) { // if opponent got mated
            return 1;
        } else {
            return 0; // stalemate (i.e. draw)
        }
    }
    /* 2) Otherwise, use the evaluation function as a heuristic Note 1: This
    evaluation is from the POV of the side-to-move at the *leaf node* we reached
    during rollout. We take care to flip it appropriately to correspond to the
    evaluation from the POV of the root state s player. Note 2: We convert this
    centipawn score into a winning probability estimate with sigmoid */
    int static_eval_score = evaluate(s, &eval);
    static_eval_score *= 2*(s->turn == color)-1;
    return 2 * winning_prob(static_eval_score) - 1;
}


/**
 * @brief Writes search information to stdout
 * @param root Root node of the MCTS search tree
 * @param searchinfo_t* Search information including e.g. # of nodes in the tree
 */
void print_MCTS_info(Node *root, searchinfo_t *info) {
    // We only want to update periodically
    if (info->nodes % 10000 != 0) 
        return;

    // Calculate the score assuming bestmove is played
    Node* best_child = root->best_child(false);
    double ucb = best_child->UCB(false);

    // Print the info line (we make sure to scale the cp score back)
    std::cout << "info depth " << info->seldepth \
              << " score cp " << centipawn_from_prob((ucb + 1) / 2.0) \
              << " nodes " << info->nodes \
              << " pv " << move_to_str(best_child->a) << std::endl;
}

/*  
    REVIEW:
    Optimization idea: Instead of rebuilding the entire tree everyime
    MCTS_Search() is called, we keep the old tree around (globally?). Depending
    on what moves actually get played, we delete all irrelevant subtrees and
    keep the relevant one.
*/

/**
 * @brief Main MCTS search function
 * @param board Board state to start the search from
 * @param info search info including time to move, depth, etc.
 */
void MCTS_Search(board_t* board, searchinfo_t *info) {

    assert(check(board));
    assert(info->state == ENGINE_SEARCHING);
    LOG("Initial checks done");

    /* Search setup */
    info->clear();
    board->ply = 0;
    const board_t root_board = *board; // Root board
    int root_color = board->turn; // Side to move at the root node (White or Black)
    int color = root_color;

    // Set up the MCTS Tree
    // Node* root = new Node(board, NULLMV, nullptr);
    // TODO: Cleanup
    arena.reset();
    void *memory = arena.allocate(sizeof(Node));
    Node *root = memory ? new (memory) Node(board, NULLMV, nullptr) : nullptr;
    LOG("Root is at " << root);

    /* Search */
    Node* node;
    double reward;
    while (!search_stopped(info)) {
        // 1) Selection
        node = select(root, board);

        // 2) Expansion (TODO: Skip this step when OOM)
        node = expand(node, board, info);

        // 3) Simulation
        color = board->turn;
        reward = simulate(board);

        // 4) Backpropagation
        backprop(reward, node, root_color, color);

        // 5) Update client with current search information
        print_MCTS_info(root, info);

        // 6) Restore board state after traversing up to the root
        *board = root_board;
    }

    // Figure out the best move at root of the tree (current game state)
    // TODO: We should report the entire principal variation of moves by
    // convention
                                        // ignore the exploration term for UCB
    move_t best_move = root->best_child(false)->a;

    std::cout << "bestmove " << move_to_str(best_move) << '\n';

    #ifdef DEBUG
    std::cout << "info string UCB scores at the root: ";
    for (Node* child : root->children) {
        std::cout << move_to_str(child->a) << ':' << child->UCB(false) << ' ';
    }
    std::cout << std::endl;

    std::cout << "info string w/ exploration term on: ";
    for (Node* child : root->children) {
        std::cout << move_to_str(child->a) << ':' << child->UCB(true) << ' ';
    }
    std::cout << std::endl;

    std::cout << "info string visits at root: ";
    for (Node* child : root->children) {
        std::cout << child->visits << ' ';
    }
    std::cout << std::endl;

    std::cout << "info string accumulated reward at root: ";
    for (Node* child : root->children) {
        std::cout << child->total_reward << ' ';
    }
    std::cout << std::endl;
    #endif

    /* Cleanup */
    root->~Node(); // should recursively destruct the entire tree (REVIEW)
    arena.reset();
    info->state = ENGINE_STOPPED;
    assert(check(board));
    LOG("Cleanup checks done");
}
