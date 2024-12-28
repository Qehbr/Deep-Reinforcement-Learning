import time

import numpy as np
import optuna


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
            hidden_sizes_theta,
            hidden_sizes_w,
            gamma_values,
            alpha_theta_values,
            alpha_w_values,
            episodes,
            log_dir="runs/optuna_search",
            source_policy_network = None,
            source_value_network = None,
    ):
        self.train_function = train_function
        self.env_name = env_name
        self.max_input_dim = max_input_dim
        self.max_output_dim = max_output_dim
        self.hidden_sizes_theta = hidden_sizes_theta
        self.hidden_sizes_w = hidden_sizes_w
        self.gamma_values = gamma_values
        self.alpha_theta_values = alpha_theta_values
        self.alpha_w_values = alpha_w_values
        self.episodes = episodes
        self.log_dir = log_dir
        self.source_policy_network = source_policy_network
        self.source_policy_network = source_policy_network

    def objective(self, trial):
        """
        Objective function for Optuna to optimize.

        Args:
            trial (optuna.Trial): The Optuna trial object used to sample hyperparameters.

        Returns:
            float: The scalar objective to maximize (here, average reward over the last 100 episodes).
        """

        # 1. Suggest parameters from the provided candidate sets
        gamma = trial.suggest_categorical("gamma", self.gamma_values)
        alpha_theta = trial.suggest_categorical("alpha_theta", self.alpha_theta_values)
        alpha_w = trial.suggest_categorical("alpha_w", self.alpha_w_values)

        print(f"\n[OPTUNA Trial] Env={self.env_name} | gamma={gamma}, alpha_theta={alpha_theta}, alpha_w={alpha_w}")

        # 2. Train actor-critic with these parameters
        train_params = {
            "env_name": self.env_name,
            "input_dim": self.max_input_dim,
            "output_dim": self.max_output_dim,
            "hidden_sizes_theta": self.hidden_sizes_theta,
            "hidden_sizes_w": self.hidden_sizes_w,
            "alpha_theta": alpha_theta,
            "alpha_w": alpha_w,
            "episodes": self.episodes,
            "gamma": gamma,
            "log_dir": f"{self.log_dir}/{self.env_name}_g{gamma}_at{alpha_theta}_aw{alpha_w}"
        }

        if self.source_policy_network is not None and self.source_policy_network is not None:
            train_params["source_policy_network"] = self.source_policy_network
            train_params["source_value_network"] = self.source_policy_network

        policy_network, value_network, rewards, train_time = self.train_function(**train_params)

        # 3. Compute the metric — e.g. average of the last 100 rewards
        avg_reward = float(np.mean(rewards[-100:]))

        # 4. Optionally store ancillary info (episodes, train_time) for inspection
        trial.set_user_attr("episodes_trained", len(rewards))
        trial.set_user_attr("train_time", train_time)

        # 5. Return the metric to be maximized
        return avg_reward

    def optuna_search_for_env(self, n_trials=10):
        """
        Uses Optuna to search for the best (gamma, alpha_theta, alpha_w) in order to maximize
        the final average reward on the given environment. Returns:
          - best_policy_network
          - best_value_network
          - best_params
          - best_reward
          - study (the Optuna study object)
        """

        # Start timing (optional)
        grid_search_start = time.time()

        # 1. Create a study object. We'll maximize the returned value (average reward).
        study = optuna.create_study(direction="maximize")

        # 2. Define a partial function or lambda that includes the non-variable arguments
        def objective_wrapper(trial):
            return self.objective(
                trial=trial,
            )

        # 3. Run the optimization for n_trials
        study.optimize(objective_wrapper, n_trials=n_trials)

        # 4. Print some info about the best trial
        print(f"\n[OPTUNA] Best trial:")
        best_trial = study.best_trial
        best_params = best_trial.params
        best_reward = best_trial.value
        print(f"  Value (Reward): {best_reward:.2f}")
        print(f"  Params: {best_params}")

        # 5. If you want the actual best networks, you can re-run the environment with best_params
        #    because we didn't store the networks themselves in the trial. We just stored metrics.

        gamma_opt = best_params["gamma"]
        alpha_theta_opt = best_params["alpha_theta"]
        alpha_w_opt = best_params["alpha_w"]

        best_policy_network, best_value_network, best_rewards, best_train_time = self.train_function(
            env_name=self.env_name,
            input_dim=self.max_input_dim,
            output_dim=self.max_output_dim,
            hidden_sizes_theta=self.hidden_sizes_theta,
            hidden_sizes_w=self.hidden_sizes_w,
            alpha_theta=alpha_theta_opt,
            alpha_w=alpha_w_opt,
            episodes=self.episodes,
            gamma=gamma_opt,
            log_dir=f"{self.log_dir}/{self.env_name}_best_g{gamma_opt}_at{alpha_theta_opt}_aw{alpha_w_opt}"
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
