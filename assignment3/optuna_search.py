import time
import json
import numpy as np
import optuna


class StudyFloatParamRange:
    def __init__(self, low, high, step):
        self.low = low
        self.high = high
        self.step = step


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
            hidden_sizes_theta_values,
            hidden_sizes_w_values,
            dropout_layers,
            gamma_values: StudyFloatParamRange,
            alpha_theta_values: StudyFloatParamRange,
            alpha_w_values: StudyFloatParamRange,
            dropout_p_values: StudyFloatParamRange,
            episodes,
            log_dir="runs/optuna_search",
    ):
        self.train_function = train_function
        self.env_name = env_name
        self.max_input_dim = max_input_dim
        self.max_output_dim = max_output_dim
        self.hidden_sizes_theta_values = hidden_sizes_theta_values
        self.hidden_sizes_w_values = hidden_sizes_w_values
        self.dropout_layers = dropout_layers
        self.gamma_values = gamma_values
        self.alpha_theta_values = alpha_theta_values
        self.alpha_w_values = alpha_w_values
        self.dropout_p_values = dropout_p_values
        self.episodes = episodes
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
        hidden_sizes_theta_str = trial.suggest_categorical("hidden_sizes_theta", self.hidden_sizes_theta_values)
        hidden_sizes_theta = fixed_hidden_theta or eval(hidden_sizes_theta_str)
        hidden_sizes_w_str = trial.suggest_categorical("hidden_sizes_w", self.hidden_sizes_w_values)
        hidden_sizes_w = fixed_hidden_w or eval(hidden_sizes_w_str)
        gamma = trial.suggest_float("gamma",
                                    low=self.gamma_values.low,
                                    high=self.gamma_values.high,
                                    step=self.gamma_values.step)
        alpha_theta = trial.suggest_float("alpha_theta",
                                          low=self.alpha_theta_values.low,
                                          high=self.alpha_theta_values.high,
                                          step=self.alpha_theta_values.step)
        alpha_w = trial.suggest_float("alpha_w",
                                      low=self.alpha_w_values.low,
                                      high=self.alpha_w_values.high,
                                      step=self.alpha_w_values.step)
        dropout_p = trial.suggest_float("dropout_p",
                                        low=self.dropout_p_values.low,
                                        high=self.dropout_p_values.high,
                                        step=self.dropout_p_values.step)

        print(f"""\n[OPTUNA Trial {trial.number}] Env={self.env_name}:
        hidden_sizes_theta={hidden_sizes_theta}, hidden_sizes_w={hidden_sizes_w},
         gamma={gamma}, dropout_p={dropout_p},
         alpha_theta={alpha_theta}, alpha_w={alpha_w}""")

        # 2. Train actor-critic with these parameters
        train_params = {
            "env_name": self.env_name,
            "input_dim": self.max_input_dim,
            "output_dim": self.max_output_dim,
            "hidden_sizes_theta": hidden_sizes_theta,
            "hidden_sizes_w": hidden_sizes_w,
            "dropout_layers": self.dropout_layers,
            "alpha_theta": alpha_theta,
            "alpha_w": alpha_w,
            "episodes": self.episodes,
            "gamma": gamma,
            "dropout_p": dropout_p,
            "log_dir": f"{self.log_dir}/{self.env_name}_g{gamma}_at{alpha_theta}_aw{alpha_w}"
        }

        if source_policy_network is not None and source_value_network is not None:
            train_params["source_policy_network"] = source_policy_network
            train_params["source_value_network"] = source_value_network

        policy_network, value_network, rewards, train_time = self.train_function(**train_params)

        # 3. Calculate the metric to minimize
        episodes_trained = len(rewards)

        # 4. Optionally store ancillary info (train_time, avg_reward(100)) for inspection
        trial.set_user_attr("train_time", train_time)
        trial.set_user_attr("avg_reward", np.mean(rewards[-100:]))

        # 5. Return the metric to be minimized
        return episodes_trained

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
        study = optuna.create_study(direction="minimize", study_name=study_name)

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
        if fixed_hidden_theta is not None:
            best_params["hidden_sizes_theta"] = fixed_hidden_theta
        if fixed_hidden_w is not None:
            best_params["hidden_sizes_w"] = fixed_hidden_w

        hidden_sizes_theta = eval(best_params["hidden_sizes_theta"])
        hidden_sizes_w = eval(best_params["hidden_sizes_w"])
        gamma_opt = best_params["gamma"]
        alpha_theta_opt = best_params["alpha_theta"]
        alpha_w_opt = best_params["alpha_w"]
        dropout_p_opt = best_params["dropout_p"]

        # write the best hyperparameters to a json file
        file_name = f"best_params/{study_name}.json"
        with open(file_name, "w") as f:
            json.dump(best_params, f, indent=4)

        best_policy_network, best_value_network, best_rewards, best_train_time = self.train_function(
            env_name=self.env_name,
            input_dim=self.max_input_dim,
            output_dim=self.max_output_dim,
            hidden_sizes_theta=hidden_sizes_theta,
            hidden_sizes_w=hidden_sizes_w,
            dropout_layers=self.dropout_layers,
            alpha_theta=alpha_theta_opt,
            alpha_w=alpha_w_opt,
            episodes=self.episodes,
            gamma=gamma_opt,
            dropout_p=dropout_p_opt,
            log_dir=f"{self.log_dir}/{self.env_name}_best_ht{hidden_sizes_theta}_hw{hidden_sizes_w}_g{gamma_opt}_at{alpha_theta_opt}_aw{alpha_w_opt}"
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
