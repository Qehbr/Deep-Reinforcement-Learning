import os
import time
import json
import numpy as np
import optuna
import torch

from hyper_params import HyperParamsRanges


class OptunaSearch:
    """
    Class to perform hyperparameter optimization using Optuna.
    """

    def __init__(
            self,
            train_function,
            env_name,
            max_input_dim,
            max_output_dim,
            episodes,
            hyper_params_ranges: HyperParamsRanges,
            log_dir="runs/optuna_search",
    ):
        self.train_function = train_function
        self.env_name = env_name
        self.max_input_dim = max_input_dim
        self.max_output_dim = max_output_dim
        self.episodes = episodes
        self.hyper_params_ranges = hyper_params_ranges
        self.log_dir = log_dir

    def objective(self,
                  trial,
                  source_policy_network=None,
                  source_value_network=None,
                  fixed_hidden_theta=None,
                  fixed_hidden_w=None):
        """
        Objective function for Optuna to optimize.

        Args:
            trial (optuna.Trial): The Optuna trial object used to sample hyperparameters.
            source_policy_network (nn.Module): The source policy network to use for fine-tuning.
            source_value_network (nn.Module): The source value network to use for fine-tuning.
            fixed_hidden_theta (list): Fixed hidden sizes for the policy network.
            fixed_hidden_w (list): Fixed hidden sizes for the value network.

        Returns:
            float: The scalar objective to maximize (here, average reward over the last 100 episodes).
        """

        # 1. Suggest parameters from the provided candidate sets
        hyper_params = self.hyper_params_ranges.suggest_hyper_params(trial, fixed_hidden_theta, fixed_hidden_w)
        print(f"\n[OPTUNA Trial {trial.number}] Env={self.env_name}")
        hyper_params.print()

        # 2. Train actor-critic with these parameters
        train_params = {
            "env_name": self.env_name,
            "input_dim": self.max_input_dim,
            "output_dim": self.max_output_dim,
            "episodes": self.episodes,
            "hyper_params": hyper_params,
            "log_dir": f"{self.log_dir}/{self.env_name}_{hyper_params.log_dir()}"
        }

        if source_policy_network is not None and source_value_network is not None:
            train_params["source_policy_network"] = source_policy_network
            train_params["source_value_network"] = source_value_network

        policy_network, value_network, rewards, train_time = self.train_function(**train_params)

        # 3. Calculate the metric to minimize
        avg_reward = np.mean(rewards[-100:])

        # 4. Optionally store ancillary info (train_time, avg_reward(100)) for inspection
        trial.set_user_attr("train_time", train_time)
        trial.set_user_attr("avg_reward", np.mean(rewards[-100:]))

        model_dir = f"models/{self.env_name}/trial_{trial.number}"
        os.makedirs(model_dir, exist_ok=True)
        torch.save(policy_network.state_dict(), f"{model_dir}/policy.pth")
        torch.save(value_network.state_dict(), f"{model_dir}/value.pth")

        # 5. Return the metric to be minimized
        return avg_reward

    def optuna_search_for_env(self,
                              n_trials=10,
                              study_name=None,
                              source_policy_network=None,
                              source_value_network=None,
                              fixed_hidden_theta=None,
                              fixed_hidden_w=None):
        """
        Uses Optuna to search for the best (gamma, alpha_theta, alpha_w) in order to maximize
        the final average reward on the given environment. Returns:
          - best_policy_network
          - best_value_network
          - best_params
          - best_reward
          - study (the Optuna study object)
        """
        study_name = study_name or f"{self.env_name}_study"
        # Start timing (optional)
        grid_search_start = time.time()

        # 1. Create a study object. We'll maximize the returned value (average reward).
        study = optuna.create_study(direction="maximize", study_name=study_name)

        # 2. Define a partial function or lambda that includes the non-variable arguments
        def objective_wrapper(trial):
            return self.objective(
                trial=trial,
                source_policy_network=source_policy_network,
                source_value_network=source_value_network,
                fixed_hidden_theta=fixed_hidden_theta,
                fixed_hidden_w=fixed_hidden_w
            )

        # 3. Run the optimization for n_trials
        study.optimize(objective_wrapper, n_trials=n_trials)

        # 4. Print some info about the best trial
        print(f"\n[OPTUNA] Best trial: trail {study.best_trial.number}")
        best_trial = study.best_trial
        best_params = best_trial.params
        best_reward = best_trial.value
        print(f"  Value (Reward): {best_reward:.2f}")
        print(f"  Params: {best_params}")

        # 5. If you want the actual best networks, you can re-run the environment with best_params
        #    because we didn't store the networks themselves in the trial. We just stored metrics.
        hyper_params = self.hyper_params_ranges.extract_best_hyper_params(best_params, fixed_hidden_theta,
                                                                          fixed_hidden_w)

        # write the best hyperparameters to a json file
        file_name = f"best_params/{study_name}.json"
        with open(file_name, "w") as f:
            json.dump(best_params, f, indent=4)

        best_policy_network, best_value_network, best_rewards, best_train_time = self.train_function(
            env_name=self.env_name,
            input_dim=self.max_input_dim,
            output_dim=self.max_output_dim,
            episodes=self.episodes,
            hyper_params=hyper_params,
            log_dir=f"{self.log_dir}/{self.env_name}_best_{hyper_params.log_dir()}"
        )

        # 6. Print total search time
        total_search_time = time.time() - grid_search_start
        print(f"\nTotal Optuna search time for {self.env_name}: {total_search_time:.2f}s")

        return (
            best_policy_network,
            best_value_network,
            best_params,
            best_reward,
            study
        )
