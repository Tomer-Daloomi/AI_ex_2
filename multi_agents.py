import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        # "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        successor_game_state = current_game_state.generate_successor(action=action)
        score = smoothness_heuristic(successor_game_state) + \
                monotonic_heuristic(successor_game_state)
        num_of_empty_tiles = len(successor_game_state.get_empty_tiles())
        empty_tiles_penalty = (successor_game_state._num_of_rows *
                               successor_game_state._num_of_columns) - num_of_empty_tiles
        return score


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """

        # we'll use this list to collect the minmax returned value for the best score
        best_moves = []

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        successor_states = [game_state.generate_successor(action=action) for action in legal_moves]

        # we first expand all the possible game state successors for our agent, and only then use
        # the minimax algorithm over each of them, so that eventually, when the minimax algorithm
        #  will return the maximal value over depth 'n' of each branch (each successor),
        # we could use it to determine which successor is the best one.
        for successor in successor_states:
            depth_counter = 1
            best_moves.append(self.minimax_val(successor, depth_counter))

        best_choice = max(best_moves)
        best_indices = [index for index in range(len(best_moves)) if best_moves[index] == best_choice]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index]

    def minimax_val(self, game_state, depth_counter):
        """
        executes the minimax recursive algorithm over a given game state, and returns the value
        of the highest valued leaf of the tree within the given depth bounds

        :param game_state:
        :param depth_counter:
        :return:
        """

        if depth_counter == self.depth * 2 - 1 or game_state.get_legal_actions(0) == [] or \
           game_state.get_legal_actions(1) == []:
            return score_evaluation_function(game_state)

        agent = depth_counter % 2  # determines whether the agent is us or the opponent

        legal_moves = game_state.get_legal_actions(agent)  # returns the legal moves for the
        # relevant agent

        # creates a list of all the successors for the relevant agent
        successor_states = [game_state.generate_successor(agent_index=agent, action=action)
                            for action in legal_moves]

        # if the agent is us - we shall return the maximal value over all possible future
        # game scores. This will help us choose the best direction towards which we should
        # advance from the current state of the game.
        if agent == 0:
            return max([self.minimax_val(game_state, depth_counter + 1) for game_state in
                        successor_states])
        # if the agent is the opponent - the contrary reason is applied
        else:
            return min([self.minimax_val(game_state, depth_counter + 1) for game_state in
                        successor_states])


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        # we'll use this list to collect the minmax returned value for the best score
        best_moves = []
        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()
        # Choose one of the best actions
        successor_states = [game_state.generate_successor(action=action) for action in legal_moves]

        for successor in successor_states:
            depth_counter = self.depth*2 - 1
            move = self.alpha_beta(successor, depth_counter, alpha=(- np.infty),
                                   beta=np.infty)
            # these two following lines take care of the returned value after pruning was executed
            if move is not None:
                best_moves.append(move)

        best_choice = max(best_moves)
        best_indices = [index for index in range(len(best_moves)) if
                        best_moves[index] == best_choice]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index]

    def alpha_beta(self, game_state, depth_counter, alpha, beta):
        """
        executes the alpha beta pruning recursive algorithm over a given game state, and returns
        the value of the highest valued leaf of the tree within the given depth bounds

        :param game_state:
        :param depth_counter:
        :param alpha
        :param beta
        :return:
        """

        if depth_counter == 0 or game_state.get_legal_actions(0) == [] or \
                        game_state.get_legal_actions(1) == []:
            return score_evaluation_function(game_state)

        agent = depth_counter % 2

        legal_moves = game_state.get_legal_actions(agent)

        successor_states = [game_state.generate_successor(agent_index=agent, action=action) for
                            action in legal_moves]

        if agent == 0:
            v = - np.infty
            for successor in successor_states:
                v = max(v, self.alpha_beta(successor, depth_counter - 1, alpha, beta))
                alpha = max(v, alpha)
                if beta <= alpha:
                    break
            return v

        if agent == 1:
            v = np.infty
            for successor in successor_states:
                v = min(v, self.alpha_beta(successor, depth_counter - 1, alpha, beta))
                beta = min(v, alpha)
                if beta <= alpha:
                    break
            return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # we'll use this list to collect the minmax returned value for the best score
        best_moves = []

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        successor_states = [game_state.generate_successor(action=action) for action in legal_moves]

        # we first expand all the possible game state successors for our agent, and only then use
        # the minimax algorithm over each of them, so that eventually, when the minimax algorithm
        #  will return the maximal value over depth 'n' of each branch (each successor),
        # we could use it to determine which successor is the best one.
        for successor in successor_states:
            depth_counter = 1
            best_moves.append(self.expectimax_val(successor, depth_counter))

        best_choice = max(best_moves)
        best_indices = [index for index in range(len(best_moves)) if best_moves[index] == best_choice]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index]

    def expectimax_val(self, game_state, depth_counter):
        """
        executes the expectimax recursive algorithm over a given game state, and returns the value
        of the highest valued leaf of the tree within the given depth bounds

        :param game_state:
        :param depth_counter:
        :return:
        """

        if depth_counter == self.depth * 2 - 1 or game_state.get_legal_actions(0) == [] or \
                game_state.get_legal_actions(1) == []:
            return score_evaluation_function(game_state)

        agent = depth_counter % 2  # determines whether the agent is us or the opponent

        legal_moves = game_state.get_legal_actions(agent)  # returns the legal moves for the
        # relevant agent

        # creates a list of all the successors for the relevant agent
        successor_states = [game_state.generate_successor(agent_index=agent, action=action)
                            for action in legal_moves]

        # if the agent is us - we shall return the maximal value over all possible future
        # game scores. This will help us choose the best direction towards which we should
        # advance from the current state of the game.
        if agent == 0:
            return max([self.expectimax_val(game_state, depth_counter + 1) for game_state in
                        successor_states])
        # if the agent is the opponent - return the average of current child nodes
        else:
            return sum([self.expectimax_val(game_state, depth_counter + 1) for game_state in
                        successor_states])//len(successor_states)


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: We decided to use a hybridization of the "monotonic heuristic" and the
    "smoothness_heuristic" (see function documentation of both for farther understanding),
    due to the fact that while the smoothness heuristic consistently returns scores that are
    higher than 7,000 , the monotonic heuristic tends to return a better "highest tile" and
    often also very high scores (relative to the depth 2 restriction of the alpha beta pruning)
    """

    return monotonic_heuristic(current_game_state) + smoothness_heuristic(current_game_state)


def monotonic_heuristic(game_state):
    """
    the heuristic examines the monotony of the rows and columns of the rows and columns of the
    matrix tht represents the game board, and returns a score - based on the amount of
    rows/columns that were monotonically ascending/descending

    :param game_state:
    :return:number rows/columns that were monotonically ascending/descending
    """

    board = game_state.board
    score = 0
    rows = np.diff(board, axis=1) >= 0
    columns = np.diff(board, axis=0) >= 0

    for i in range(game_state._num_of_rows - 1):
        if False not in rows[i]:
            score +=1
        if False not in columns[i]:
            score += 1

    return score


def smoothness_heuristic(game_state):
    """
    the heuristic examines the number of identical adjacent tiles on the game board.

    :param game_state:
    :return: number of identical adjacent tiles
    """

    board = game_state.board
    rows = np.sum(np.diff(board, axis=1) == 0)
    columns = np.sum(np.diff(board, axis=0) == 0)

    return rows + columns


# Abbreviation
better = better_evaluation_function
