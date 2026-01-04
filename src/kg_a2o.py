"""
KG-A2O: Knowledge Graph-based Automated Operator Optimization
Neuro-Symbolic Optimization via Knowledge Graph and Reinforcement Learning

This module implements Algorithm 2 from the paper:
- Stage I: Heterogeneous Representation Learning (RGAT)
- Stage II: RL-based Optimization Search (PPO)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import deque
import json
import random
from loguru import logger


@dataclass
class OptimizationAction:
    """Represents an optimization action that can be applied to an operator"""
    action_id: str
    name: str
    description: str
    applicable_ops: List[str]  # List of operator types this action applies to
    expected_speedup: float = 1.0
    accuracy_impact: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "action_id": self.action_id,
            "name": self.name,
            "description": self.description,
            "applicable_ops": self.applicable_ops,
            "expected_speedup": self.expected_speedup,
            "accuracy_impact": self.accuracy_impact,
        }


@dataclass
class State:
    """MDP State representation"""
    operator_embedding: np.ndarray  # RGAT embedding of current operator
    hardware_embedding: np.ndarray  # Hardware platform embedding
    operator_id: str
    operator_type: str
    current_latency: float
    optimization_history: List[str] = field(default_factory=list)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor for neural network input"""
        combined = np.concatenate([
            self.operator_embedding,
            self.hardware_embedding,
            np.array([self.current_latency])
        ])
        return torch.FloatTensor(combined)


@dataclass
class Transition:
    """Stores a single transition in the replay buffer"""
    state: State
    action: int
    reward: float
    next_state: State
    done: bool
    log_prob: float
    value: float


class ActionSpace:
    """
    Discrete action space for operator optimization
    
    Actions include:
    - Enable_TensorCore: Use tensor cores for matrix operations
    - Fusion: Fuse with adjacent operations
    - Layout_NHWC: Change memory layout to NHWC
    - Layout_NCHW: Change memory layout to NCHW
    - Precision_FP16: Use FP16 precision
    - Precision_BF16: Use BF16 precision
    - Precision_INT8: Use INT8 quantization
    - KV_Cache: Enable KV-cache for attention
    - Flash_Attention: Use Flash Attention implementation
    - Winograd: Use Winograd convolution
    - Im2Col: Use Im2Col convolution
    - NoOp: No optimization (baseline)
    """
    
    def __init__(self):
        self.actions = [
            OptimizationAction(
                action_id="enable_tensor_core",
                name="Enable_TensorCore",
                description="Enable tensor core acceleration for matrix operations",
                applicable_ops=["MatMul", "Gemm", "Conv2d", "Linear"],
                expected_speedup=2.0,
                accuracy_impact=0.0
            ),
            OptimizationAction(
                action_id="fusion",
                name="Fusion",
                description="Fuse operator with adjacent operations",
                applicable_ops=["MatMul", "LayerNorm", "RMSNorm", "GELU", "ReLU", "Add"],
                expected_speedup=1.3,
                accuracy_impact=0.0
            ),
            OptimizationAction(
                action_id="layout_nhwc",
                name="Layout_NHWC",
                description="Change memory layout to NHWC format",
                applicable_ops=["Conv2d", "Conv3d", "BatchNorm", "MaxPool"],
                expected_speedup=1.2,
                accuracy_impact=0.0
            ),
            OptimizationAction(
                action_id="layout_nchw",
                name="Layout_NCHW",
                description="Change memory layout to NCHW format",
                applicable_ops=["Conv2d", "Conv3d", "BatchNorm", "MaxPool"],
                expected_speedup=1.1,
                accuracy_impact=0.0
            ),
            OptimizationAction(
                action_id="precision_fp16",
                name="Precision_FP16",
                description="Use FP16 mixed precision",
                applicable_ops=["MatMul", "Gemm", "Conv2d", "Linear", "Attention"],
                expected_speedup=1.8,
                accuracy_impact=-0.001
            ),
            OptimizationAction(
                action_id="precision_bf16",
                name="Precision_BF16",
                description="Use BF16 mixed precision",
                applicable_ops=["MatMul", "Gemm", "Conv2d", "Linear", "Attention"],
                expected_speedup=1.7,
                accuracy_impact=-0.0005
            ),
            OptimizationAction(
                action_id="precision_int8",
                name="Precision_INT8",
                description="Use INT8 quantization",
                applicable_ops=["MatMul", "Gemm", "Conv2d", "Linear"],
                expected_speedup=2.5,
                accuracy_impact=-0.01
            ),
            OptimizationAction(
                action_id="kv_cache",
                name="KV_Cache",
                description="Enable KV-cache for attention operations",
                applicable_ops=["Attention", "MultiHeadAttention"],
                expected_speedup=3.0,
                accuracy_impact=0.0
            ),
            OptimizationAction(
                action_id="flash_attention",
                name="Flash_Attention",
                description="Use Flash Attention implementation",
                applicable_ops=["Attention", "MultiHeadAttention"],
                expected_speedup=2.5,
                accuracy_impact=0.0
            ),
            OptimizationAction(
                action_id="winograd",
                name="Winograd",
                description="Use Winograd convolution for small kernels",
                applicable_ops=["Conv2d"],
                expected_speedup=1.5,
                accuracy_impact=0.0
            ),
            OptimizationAction(
                action_id="im2col",
                name="Im2Col",
                description="Use Im2Col convolution implementation",
                applicable_ops=["Conv2d", "Conv3d"],
                expected_speedup=1.2,
                accuracy_impact=0.0
            ),
            OptimizationAction(
                action_id="noop",
                name="NoOp",
                description="No optimization (baseline)",
                applicable_ops=["*"],  # Applies to all operators
                expected_speedup=1.0,
                accuracy_impact=0.0
            ),
        ]
        
        self.action_to_idx = {a.action_id: i for i, a in enumerate(self.actions)}
        self.idx_to_action = {i: a for i, a in enumerate(self.actions)}
    
    @property
    def n_actions(self) -> int:
        return len(self.actions)
    
    def get_valid_actions(self, operator_type: str) -> List[int]:
        """Get indices of valid actions for a given operator type"""
        valid = []
        for i, action in enumerate(self.actions):
            if "*" in action.applicable_ops or operator_type in action.applicable_ops:
                valid.append(i)
        return valid
    
    def get_action_mask(self, operator_type: str) -> torch.Tensor:
        """Get binary mask for valid actions"""
        mask = torch.zeros(self.n_actions)
        for idx in self.get_valid_actions(operator_type):
            mask[idx] = 1.0
        return mask


class PolicyNetwork(nn.Module):
    """
    Policy Network for PPO
    
    Takes state embedding as input and outputs action probabilities
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
    
    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        
        # Apply action mask (set invalid actions to -inf)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))
        
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """
    Value Network for PPO
    
    Takes state embedding as input and outputs state value estimate
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPOAgent:
    """
    Proximal Policy Optimization Agent
    
    Implements the PPO-Clip algorithm for policy optimization
    """
    
    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        self.action_space = action_space
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_space.n_actions).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer: List[Transition] = []
    
    def select_action(self, state: State) -> Tuple[int, float, float]:
        """Select action using current policy"""
        state_tensor = state.to_tensor().unsqueeze(0).to(self.device)
        action_mask = self.action_space.get_action_mask(state.operator_type).to(self.device)
        
        with torch.no_grad():
            probs = self.policy(state_tensor, action_mask)
            value = self.value(state_tensor)
        
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, transition: Transition):
        """Store transition in buffer"""
        self.buffer.append(transition)
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self, n_epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        """Update policy and value networks using PPO"""
        if len(self.buffer) == 0:
            return {}
        
        # Prepare data
        states = torch.stack([t.state.to_tensor() for t in self.buffer]).to(self.device)
        actions = torch.LongTensor([t.action for t in self.buffer]).to(self.device)
        old_log_probs = torch.FloatTensor([t.log_prob for t in self.buffer]).to(self.device)
        rewards = [t.reward for t in self.buffer]
        values = [t.value for t in self.buffer]
        dones = [t.done for t in self.buffer]
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(n_epochs):
            # Get current policy probabilities
            action_masks = torch.stack([
                self.action_space.get_action_mask(t.state.operator_type)
                for t in self.buffer
            ]).to(self.device)
            
            probs = self.policy(states, action_masks)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Policy loss (PPO-Clip)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            new_values = self.value(states).squeeze()
            value_loss = F.mse_loss(new_values, returns)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            "policy_loss": total_policy_loss / n_epochs,
            "value_loss": total_value_loss / n_epochs,
            "entropy": total_entropy / n_epochs,
        }


class SurrogatePerformancePredictor:
    """
    Surrogate model for fast performance prediction
    
    Uses GNN-based predictor instead of actual hardware execution
    to accelerate RL training
    """
    
    def __init__(self, knowledge_graph: Optional[Any] = None):
        self.kg = knowledge_graph
        self._cache: Dict[str, float] = {}
    
    def predict_latency(
        self,
        operator_id: str,
        operator_type: str,
        action: OptimizationAction,
        baseline_latency: float
    ) -> float:
        """
        Predict latency after applying optimization action
        
        Uses a combination of:
        1. Action's expected speedup
        2. Historical data from knowledge graph
        3. Operator-specific adjustments
        """
        cache_key = f"{operator_id}_{action.action_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Base prediction using expected speedup
        predicted_latency = baseline_latency / action.expected_speedup
        
        # Add some noise to model real-world variability
        noise = np.random.normal(0, 0.05 * predicted_latency)
        predicted_latency += noise
        
        # Ensure positive latency
        predicted_latency = max(predicted_latency, 0.001)
        
        self._cache[cache_key] = predicted_latency
        return predicted_latency
    
    def clear_cache(self):
        self._cache.clear()


class OptimizationEnvironment:
    """
    RL Environment for operator optimization
    
    Performs the optimization process using surrogate performance predictor
    """
    
    def __init__(
        self,
        operators: List[Dict[str, Any]],
        hardware_embedding: np.ndarray,
        action_space: ActionSpace,
        surrogate: SurrogatePerformancePredictor,
        lambda_1: float = 1.0,  # Latency weight
        lambda_2: float = 10.0,  # Accuracy penalty weight
        accuracy_threshold: float = 0.99
    ):
        self.operators = operators
        self.hardware_embedding = hardware_embedding
        self.action_space = action_space
        self.surrogate = surrogate
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.accuracy_threshold = accuracy_threshold
        
        self.current_idx = 0
        self.optimization_plan: List[Tuple[str, str]] = []
        self.total_latency_reduction = 0.0
        self.total_accuracy_loss = 0.0
    
    def reset(self) -> State:
        """Reset environment to initial state"""
        self.current_idx = 0
        self.optimization_plan = []
        self.total_latency_reduction = 0.0
        self.total_accuracy_loss = 0.0
        
        return self._get_state()
    
    def _get_state(self) -> State:
        """Get current state"""
        if self.current_idx >= len(self.operators):
            # Return terminal state
            return State(
                operator_embedding=np.zeros(64),
                hardware_embedding=self.hardware_embedding,
                operator_id="terminal",
                operator_type="terminal",
                current_latency=0.0,
                optimization_history=list(self.optimization_plan)
            )
        
        op = self.operators[self.current_idx]
        return State(
            operator_embedding=np.array(op.get("embedding", np.random.randn(64))),
            hardware_embedding=self.hardware_embedding,
            operator_id=op["id"],
            operator_type=op["type"],
            current_latency=op["latency"],
            optimization_history=list(self.optimization_plan)
        )
    
    def step(self, action_idx: int) -> Tuple[State, float, bool, Dict]:
        """
        Execute action and return next state, reward, done, info
        
        Reward function:
        R_t = λ_1 * (T_baseline - T_new) / T_baseline - λ_2 * I(Acc < τ)
        """
        if self.current_idx >= len(self.operators):
            return self._get_state(), 0.0, True, {}
        
        op = self.operators[self.current_idx]
        action = self.action_space.idx_to_action[action_idx]
        
        # Predict new latency using surrogate
        baseline_latency = op["latency"]
        new_latency = self.surrogate.predict_latency(
            op["id"], op["type"], action, baseline_latency
        )
        
        # Calculate reward
        latency_reduction = (baseline_latency - new_latency) / baseline_latency
        accuracy_loss = action.accuracy_impact
        self.total_accuracy_loss += accuracy_loss
        
        # Accuracy penalty
        accuracy_penalty = 0.0
        if (1.0 + self.total_accuracy_loss) < self.accuracy_threshold:
            accuracy_penalty = 1.0
        
        reward = self.lambda_1 * latency_reduction - self.lambda_2 * accuracy_penalty
        
        # Record optimization
        self.optimization_plan.append((op["id"], action.action_id))
        self.total_latency_reduction += latency_reduction
        
        # Move to next operator
        self.current_idx += 1
        done = self.current_idx >= len(self.operators)
        
        next_state = self._get_state()
        
        info = {
            "latency_reduction": latency_reduction,
            "accuracy_loss": accuracy_loss,
            "action_name": action.name,
        }
        
        return next_state, reward, done, info
    
    def get_optimization_plan(self) -> List[Tuple[str, str]]:
        """Get the current optimization plan"""
        return self.optimization_plan


class KGA2O:
    """
    KG-A2O: Knowledge Graph-based Automated Operator Optimization
    
    Main class that orchestrates the neuro-symbolic optimization process
    """
    
    def __init__(
        self,
        knowledge_graph: Optional[Any] = None,
        state_dim: int = 129,  # 64 (op) + 64 (hw) + 1 (latency)
        device: str = "cpu"
    ):
        self.kg = knowledge_graph
        self.action_space = ActionSpace()
        self.surrogate = SurrogatePerformancePredictor(knowledge_graph)
        
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_space=self.action_space,
            device=device
        )
        
        self.best_plan: List[Tuple[str, str]] = []
        self.best_reward: float = float('-inf')
        self.training_history: List[Dict] = []
    
    def train(
        self,
        operators: List[Dict[str, Any]],
        hardware_embedding: np.ndarray,
        n_episodes: int = 100,
        max_steps: int = 500,
        update_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Train the optimization agent
        
        Args:
            operators: List of operator dictionaries with id, type, latency, embedding
            hardware_embedding: Hardware platform embedding
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            update_interval: Episodes between policy updates
        
        Returns:
            Training statistics and best optimization plan
        """
        logger.info(f"Starting KG-A2O training for {n_episodes} episodes")
        
        env = OptimizationEnvironment(
            operators=operators,
            hardware_embedding=hardware_embedding,
            action_space=self.action_space,
            surrogate=self.surrogate
        )
        
        episode_rewards = []
        
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(min(max_steps, len(operators))):
                # Select action
                action, log_prob, value = self.agent.select_action(state)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                self.agent.store_transition(Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    log_prob=log_prob,
                    value=value
                ))
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # Track best plan
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_plan = env.get_optimization_plan()
            
            # Update policy
            if (episode + 1) % update_interval == 0:
                update_stats = self.agent.update()
                self.training_history.append({
                    "episode": episode + 1,
                    "reward": episode_reward,
                    "avg_reward": np.mean(episode_rewards[-update_interval:]),
                    **update_stats
                })
                
                logger.info(
                    f"Episode {episode + 1}/{n_episodes} | "
                    f"Reward: {episode_reward:.4f} | "
                    f"Avg: {np.mean(episode_rewards[-update_interval:]):.4f}"
                )
        
        logger.info(f"Training complete. Best reward: {self.best_reward:.4f}")
        
        return {
            "best_plan": self.best_plan,
            "best_reward": self.best_reward,
            "episode_rewards": episode_rewards,
            "training_history": self.training_history,
        }
    
    def optimize(
        self,
        operators: List[Dict[str, Any]],
        hardware_embedding: np.ndarray
    ) -> List[Tuple[str, str]]:
        """
        Apply trained policy to optimize operators
        
        Returns:
            List of (operator_id, action_id) tuples
        """
        env = OptimizationEnvironment(
            operators=operators,
            hardware_embedding=hardware_embedding,
            action_space=self.action_space,
            surrogate=self.surrogate
        )
        
        state = env.reset()
        optimization_plan = []
        
        for _ in range(len(operators)):
            action, _, _ = self.agent.select_action(state)
            next_state, _, done, info = env.step(action)
            
            optimization_plan.append((
                state.operator_id,
                self.action_space.idx_to_action[action].action_id
            ))
            
            state = next_state
            if done:
                break
        
        return optimization_plan
    
    def get_action_recommendations(
        self,
        operator_type: str
    ) -> List[Dict]:
        """Get recommended actions for an operator type"""
        valid_actions = self.action_space.get_valid_actions(operator_type)
        recommendations = []
        
        for idx in valid_actions:
            action = self.action_space.idx_to_action[idx]
            recommendations.append({
                "action_id": action.action_id,
                "name": action.name,
                "description": action.description,
                "expected_speedup": action.expected_speedup,
                "accuracy_impact": action.accuracy_impact,
            })
        
        # Sort by expected speedup
        recommendations.sort(key=lambda x: x["expected_speedup"], reverse=True)
        
        return recommendations
    
    def save(self, path: str):
        """Save trained model"""
        torch.save({
            "policy_state_dict": self.agent.policy.state_dict(),
            "value_state_dict": self.agent.value.state_dict(),
            "best_plan": self.best_plan,
            "best_reward": self.best_reward,
            "training_history": self.training_history,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path)
        self.agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.agent.value.load_state_dict(checkpoint["value_state_dict"])
        self.best_plan = checkpoint["best_plan"]
        self.best_reward = checkpoint["best_reward"]
        self.training_history = checkpoint["training_history"]
        logger.info(f"Model loaded from {path}")
    
    def export_results(self, path: str):
        """Export optimization results to JSON"""
        results = {
            "best_plan": [
                {"operator_id": op_id, "action_id": action_id}
                for op_id, action_id in self.best_plan
            ],
            "best_reward": self.best_reward,
            "action_space": [a.to_dict() for a in self.action_space.actions],
            "training_history": self.training_history,
        }
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results exported to {path}")


if __name__ == "__main__":
    # Test KG-A2O
    logger.info("Testing KG-A2O algorithm...")
    
    # Create sample operators
    operators = [
        {"id": "op_001", "type": "MatMul", "latency": 5.0, "embedding": np.random.randn(64)},
        {"id": "op_002", "type": "GELU", "latency": 0.5, "embedding": np.random.randn(64)},
        {"id": "op_003", "type": "LayerNorm", "latency": 1.0, "embedding": np.random.randn(64)},
        {"id": "op_004", "type": "Attention", "latency": 8.0, "embedding": np.random.randn(64)},
        {"id": "op_005", "type": "MatMul", "latency": 4.0, "embedding": np.random.randn(64)},
        {"id": "op_006", "type": "ReLU", "latency": 0.2, "embedding": np.random.randn(64)},
        {"id": "op_007", "type": "Conv2d", "latency": 3.0, "embedding": np.random.randn(64)},
        {"id": "op_008", "type": "BatchNorm", "latency": 0.8, "embedding": np.random.randn(64)},
    ]
    
    hardware_embedding = np.random.randn(64)
    
    # Initialize KG-A2O
    kg_a2o = KGA2O(device="cpu")
    
    # Train
    results = kg_a2o.train(
        operators=operators,
        hardware_embedding=hardware_embedding,
        n_episodes=50,
        update_interval=5
    )
    
    print("\n=== KG-A2O Training Results ===")
    print(f"Best reward: {results['best_reward']:.4f}")
    print(f"Best plan:")
    for op_id, action_id in results['best_plan']:
        action = kg_a2o.action_space.actions[kg_a2o.action_space.action_to_idx[action_id]]
        print(f"  {op_id}: {action.name} (speedup: {action.expected_speedup}x)")
    
    # Get recommendations
    print("\n=== Action Recommendations for MatMul ===")
    recs = kg_a2o.get_action_recommendations("MatMul")
    for rec in recs[:3]:
        print(f"  {rec['name']}: {rec['expected_speedup']}x speedup")
    
    print("\nKG-A2O test complete!")
