#include "mcts.h"

#include <vector>
#include <cmath>
#include <climits>

#include "eval.h"
#include "threads.h"
#include "sgd.h"
#include "board.h"
#include "arena.h"

// Global evaluator
extern eval_t eval;
extern double winning_prob(double score);

// Constants (TODO: Tune with self-play?)
constexpr double UCB_CONST = 0.7; 
constexpr int ROLLOUT_BUDGET = 10;
constexpr size_t DEFAULT_ARENA_MB = 256; /* Default size of the arena in MB */

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

[[__always_inline__]] static inline Action random_policy(movelist_t& actions) {
    return actions[rand_uint64() % actions.size()];
}


/**
 * @brief Given a node, select a *legal* action according to policy,
 * perform the action, and insert a child node with the resulting state
 * 
 * @return Node* or nullptr if no legal action could be taken
 */
Node* select_and_insert(Node* node, State* s, Action (*policy)(movelist_t&)) {

    assert(node != nullptr);
    assert(s != nullptr);
    assert(!node->is_terminal());

    // We sample actions until we find a valid one
    LOG("Board before making move:");
    print(s);
    Action a = policy(node->untried_moves);
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
        a = policy(node->untried_moves);
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

double rollout(Node *node, State *s) {

    assert(node != nullptr);
    assert(s != nullptr);

    // We limit the number of rollouts (tree height)
    int budget = ROLLOUT_BUDGET;
    while (!node->is_terminal() && budget --> 0) {
        // Note: select_and_insert can return a nullptr
        Node *child = select_and_insert(node, s, &rollout_policy);
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
 */
void backprop(double reward, Node *node) {
    assert(node != nullptr);

    Node *curr = node;
    while (curr != nullptr) {
        curr->update(reward);
        curr = curr->parent;
    }
}

double Node::UCB(bool exploration_mode) {
    // Avoid div-by-zero
    double ucb = static_cast<double>(total_reward) / (visits + 1);
    if (exploration_mode)
        ucb += UCB_CONST * std::sqrt(std::log(parent->visits) / (visits + 1));

    return ucb;
}


Node *Node::best_child(bool exploration_mode) {
    // Calculate UCB values for all the children and pick the highest
    double ucb;
    double best_value = static_cast<double>(INT_MIN);

    std::vector<Node*> best_children;

    for (Node* child : this->children) {

        ucb = child->UCB(exploration_mode);

        if (ucb > best_value) {
            best_value = ucb;
            best_children.clear();
            best_children.push_back(child);
        } else if (ucb == best_value) {
            best_children.push_back(child);
        }
        
    }

    assert(best_children.size() > 0);

    // Randomly determine ties between best children (REVIEW: This could be improved)
    size_t random_pick = rand_uint64() % best_children.size();
    return best_children[random_pick];
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
    this->total_reward += reward;
    ++this->visits;
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
            Node *child = select_and_insert(node, s, &prior_prob);
            return child;
        }
    }
    return node;
}

/**
 * @brief Given the MCTS root and current board state, find a node 
 * within the tree to expand
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
            node = node->best_child();
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
Action play_legal(State *s, Action (*policy)(movelist_t&), movelist_t& moves) {

    // Pick a move according to policy 
    // (REVIEW: We might want the policy to take in the state too?)
    Action a = policy(moves);
    while (!make_move(s, a)) {
        // Remove the action from the list, as it is illegal
        moves.erase(moves.find(a));
        if (moves.size() <= 0) {
            // Ran out of moves -> can't play any legal action from state s
            return NULLMV;
        }
        a = policy(moves);
    }
    return a;
}

/**
 * @brief Attempts to expand node. If successful, returns the new child,
 * and otherwise the input node.
 * @param node Pointer to a node to be expanded
 * @param s Current board state corresponding to @param node
 * @return Node* New child if successful, @param node otherwise
 */
Node *expand(Node *node, State *s) {
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
    Action a = play_legal(s, &prior_prob, node->untried_moves);
    if (a != NULLMV) 
        return node->insert_child(a, s);

    return node;
}

/**
 * @brief Perform a simulation (playout).
 *  Like rollout(), but doesn't insert any new nodes into the tree
 * @param node Node to start the playout from
 * @param s Board state corresponding to @param node
 * @return double The reward, r \in [0, 1] (REVIEW: For which side?)
 */
double simulate(State *s) {
    assert(s != nullptr);

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
        // If side to turn (us?) is in check and node is terminal (no moves),
        // we've been mated (we get a centipawn score of negative infinity,
        // equivalent to a zero probability of winning)
        // REVIEW: Should we return -oo or 0?
        if (is_in_check(s, s->turn)) {
            return 0;
        
        // REVIEW: Will this condition *ever* hold?
        } else if (is_in_check(s, s->turn ^ 1)) { // if opponent got mated
            return 1;
        } else {
            return 0.5; // stalemate (i.e. draw)
        }
    }

    // 2) Otherwise, use the static evaluation function as a heuristic
    // Note: We convert this centipawn score into a winning probability 
    // estimate with sigmoid
    int static_eval_score = evaluate(s, &eval);
    return winning_prob(static_eval_score);
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

    // Set up the MCTS Tree
    const board_t root_board = *board;
    // Node* root = new Node(board, NULLMV, nullptr);
    // TODO: Cleanup
    arena.reset();
    void *memory = arena.allocate(sizeof(Node));
    Node *root = memory ? new (memory) Node(board, NULLMV, nullptr) : nullptr;
    LOG("Root is at " << root);

    // Perform the search
    Node* node;
    double reward;
    while (!search_stopped(info)) {
        // 1) Selection
        node = select(root, board);
        LOG("Selected '" << to_fen(board) << "' for expansion at " << node);

        // 2) Expansion (TODO: Skip this step when OOM)
        node = expand(node, board);

        // 3) Simulation
        reward = simulate(board);

        // 4) Backpropagation
        backprop(reward, node);

        // 5) Restore board state after traversing up to the root
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
    root->~Node(); // should recursively delete the entire tree (REVIEW)
    arena.reset();
    info->state = ENGINE_STOPPED;
    assert(check(board));
    LOG("Cleanup checks done");
}
