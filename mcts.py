import math
import optimized_functions as gf
import numpy as np
import torch
import chess
from resnet import ResNet
global max_depth 
max_depth = set()

class Node:
    def __init__(self, args:'dict', state:'chess.Board', move_counter:'int', depth:'int'=0, parent:'Node | None'=None, action:'int|None'=None, prior:'float|None'=None, policy:'np.ndarray|None'=None):
        self.args = args
        self.state = state
        self.parent = parent
        self.prior = prior
        self.policy = policy
        self.action = action
        self.depth = depth
        self.move_counter = move_counter
        

        self.Q = 0.
        self.N = 0
        self.children:list[Node] = []

    def is_terminal(self) -> bool:
        return gf.game_result(self.state, self.move_counter, 1000)[1]

    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def ucb(self, child: 'Node') -> float:
        c_base = self.args["c_base"]
        c_init = self.args["c_init"]
        t = self.args["t"]
        return ((child.Q + 1e-6) / (child.N + 1e-12)) +(math.log((1 + self.N + c_base) / c_base) + c_init) * (child.prior** (1/t)) * (math.sqrt(self.N) / (1 + child.N))


    def best_child(self) -> 'Node':
        child_ucbs = [self.ucb(child) for child in self.children]
        return self.children[np.argmax(child_ucbs)]
    
    @torch.no_grad()
    def value_policy_batch(self, states: 'list[chess.Board]', move_counter: 'int ', model: 'ResNet') -> tuple[np.ndarray, np.ndarray]:
        
        model.eval()
        inputs = torch.stack([gf.prepare_input(state, move_counter) for state in states]).to(self.args['device'])
        values, policies = model(inputs)
        values = values.cpu().numpy()
        policies = torch.softmax(policies, dim=1).cpu().numpy()
        
        
        valid_policies = gf.parallel_valid_policy(policies, states) 
        
        return values, valid_policies

    def expand(self, model: 'ResNet') -> None:
        top_actions = self.args['top_actions']
        new_states = []
        actions = []
        probs = []
        top_actions_probs = sorted(enumerate(self.policy), key=lambda x: x[1], reverse=True)[:top_actions]
        for action, prob in top_actions_probs:
            if prob > 0:
                new_state = self.state.copy()
                move = gf.alphazero_to_move(action)
                new_state.push_san(move)
                new_state.apply_mirror()
                new_states.append(new_state)
                actions.append(action)
                probs.append(prob)
                
                
        _,policies = self.value_policy_batch(new_states, self.move_counter+1, model)
        values = [gf.board_value(new_states[i]) for i in range(len(new_states))]
        max_depth.add(self.depth+1)
        for i, state in enumerate(new_states):
            if state.is_checkmate():
                values[i] = -1000
            child = Node(self.args, new_states[i], self.move_counter+1, self.depth+1, self, actions[i], probs[i], policies[i])
            self.children.append(child)
            child.backpropagation(-values[i])
     

    def backpropagation(self, rollout_value: float) -> None:
        self.N += 1

        self.Q += rollout_value
        node = self.parent
        while node is not None:
            rollout_value = -rollout_value
            node.N += 1
            node.Q += rollout_value
            node = node.parent

class MCTS:
    def __init__(self, args: 'dict', model: 'ResNet') -> None:
        self.args = args
        self.model = model.to(args['device'])




    @torch.no_grad()
    def value_policy(self, state: 'chess.Board', move_counter: 'int') -> tuple[float, np.ndarray]:
       
        self.model.eval()
        _, policy = self.model(gf.prepare_input(state, move_counter).unsqueeze(0).to(self.args['device']))
        value =gf.board_value(state)
        
        #value = value.cpu().item()
        policy = torch.softmax(policy.squeeze(0), dim=0).cpu().numpy()
       
        valid_policy = gf.valid_policy(policy, state)
        return value, valid_policy




    def _simulate(self, root: 'Node') -> None:
        current_node = root
        while not current_node.is_leaf():
            current_node = current_node.best_child()

        if current_node.is_terminal():
            value = -gf.game_result(current_node.state, current_node.move_counter, 1000)[0]*1000
            current_node.backpropagation(value)
        else:
            current_node.expand(self.model)



    def search(self, state: 'chess.Board', move_counter: 'int') -> np.ndarray:
        max_depth.clear()
        root_state = state.copy()
        value, policy = self.value_policy(root_state, move_counter)
     
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet(
            [self.args['dirichlet_alpha']] * self.args['action_space'])
        policy = gf.valid_policy(policy, root_state)
        root = Node(self.args, root_state, move_counter)
        root.policy = policy

        for _ in range(self.args["num_simulation"]):
            self._simulate(root)
        print("max_depth", max(max_depth ))
        mcts_action_probs = np.zeros(self.args['action_space'])
        for child in root.children:
            mcts_action_probs[child.action] = child.N
        return mcts_action_probs / np.sum(mcts_action_probs)
    
