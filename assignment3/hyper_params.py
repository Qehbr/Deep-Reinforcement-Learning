from typing import List

from optuna import Trial


class StudyFloatParamRange:
    def __init__(self, low, high, step):
        self.low = low
        self.high = high
        self.step = step

    def suggest_float(self, trial: Trial, name):
        return trial.suggest_float(name, low=self.low, high=self.high, step=self.step)


class HyperParams:
    def __init__(self,
                 hidden_sizes_theta,
                 hidden_sizes_w,
                 alpha_theta,
                 alpha_w,
                 gamma,
                 dropout_p,
                 epsilon=None,
                 epsilon_decay=None,
                 min_noise_std=None,
                 max_noise_std=None
                 ):
        self.hidden_sizes_theta = hidden_sizes_theta
        self.hidden_sizes_w = hidden_sizes_w
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.gamma = gamma
        self.dropout_p = dropout_p
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std

    def print(self):
        print_str = f"""hidden_sizes_theta={self.hidden_sizes_theta}  |  hidden_sizes_w={self.hidden_sizes_w}
        gamma={self.gamma:.4f}  |  dropout_p={self.dropout_p:.4f}
        alpha_theta={self.alpha_theta:.4f}  |  alpha_w={self.alpha_w:.4f}"""
        if self.epsilon is not None:
            print_str += f"""
            epsilon={self.epsilon:.4f}  |  epsilon_decay={self.epsilon_decay:.4f}
            min_noise_std={self.min_noise_std:.4f}  |  max_noise_std={self.max_noise_std:.4f}"""
        print(print_str)

    @staticmethod
    def hidden_sizes_str(hidden_sizes):
        return '_'.join(map(str, hidden_sizes))

    def log_dir(self):
        log_dir = f"hst{self.hidden_sizes_str(self.hidden_sizes_theta)}_hsw{self.hidden_sizes_str(self.hidden_sizes_w)}_at{self.alpha_theta:.4f}_aw{self.alpha_w:.4f}_g{self.gamma:.4f}_dp{self.dropout_p:.4f}"
        if self.epsilon is not None:
            log_dir += f"_e{self.epsilon:.4f}_ed{self.epsilon_decay:.4f}_min{self.min_noise_std:.4f}_max{self.max_noise_std:.4f}"
        return log_dir


class HyperParamsRanges:
    def __init__(self,
                 hidden_sizes_theta_values: List[str],
                 hidden_sizes_w_values: List[str],
                 alpha_theta_values: StudyFloatParamRange,
                 alpha_w_values: StudyFloatParamRange,
                 gamma_values: StudyFloatParamRange,
                 dropout_p_values: StudyFloatParamRange,
                 epsilon_values: StudyFloatParamRange = None,
                 epsilon_decay_values: StudyFloatParamRange = None,
                 min_noise_std_values: StudyFloatParamRange = None,
                 max_noise_std_values: StudyFloatParamRange = None
                 ):
        self.hidden_sizes_theta_values = hidden_sizes_theta_values
        self.hidden_sizes_w_values = hidden_sizes_w_values
        self.alpha_theta_values = alpha_theta_values
        self.alpha_w_values = alpha_w_values
        self.gamma_values = gamma_values
        self.dropout_p_values = dropout_p_values
        self.epsilon_values = epsilon_values
        self.epsilon_decay_values = epsilon_decay_values
        self.min_noise_std_values = min_noise_std_values
        self.max_noise_std_values = max_noise_std_values

    def suggest_hyper_params(self, trial: Trial, fixed_hidden_theta=None, fixed_hidden_w=None):
        hidden_sizes_theta_str = trial.suggest_categorical("hidden_sizes_theta", self.hidden_sizes_theta_values)
        hidden_sizes_theta = fixed_hidden_theta or eval(hidden_sizes_theta_str)
        hidden_sizes_w_str = trial.suggest_categorical("hidden_sizes_w", self.hidden_sizes_w_values)
        hidden_sizes_w = fixed_hidden_w or eval(hidden_sizes_w_str)
        params = {
            "hidden_sizes_theta": hidden_sizes_theta,
            "hidden_sizes_w": hidden_sizes_w,
            "alpha_theta": self.alpha_theta_values.suggest_float(trial, "alpha_theta"),
            "alpha_w": self.alpha_w_values.suggest_float(trial, "alpha_w"),
            "gamma": self.gamma_values.suggest_float(trial, "gamma"),
            "dropout_p": self.dropout_p_values.suggest_float(trial, "dropout_p")
        }
        if self.epsilon_values is not None:
            params["epsilon"] = self.epsilon_values.suggest_float(trial, "epsilon")
            params["epsilon_decay"] = self.epsilon_decay_values.suggest_float(trial, "epsilon_decay")
            params["min_noise_std"] = self.min_noise_std_values.suggest_float(trial, "min_noise_std")
            params["max_noise_std"] = self.max_noise_std_values.suggest_float(trial, "max_noise_std")
        return HyperParams(**params)

    @staticmethod
    def extract_best_hyper_params(best_params, fixed_hidden_theta=None, fixed_hidden_w=None):
        hidden_sizes_theta = fixed_hidden_theta if fixed_hidden_theta is not None else best_params["hidden_sizes_theta"]
        hidden_sizes_theta = eval(hidden_sizes_theta)
        hidden_sizes_w = fixed_hidden_w if fixed_hidden_w is not None else best_params["hidden_sizes_w"]
        hidden_sizes_w = eval(hidden_sizes_w)
        gamma_opt = best_params["gamma"]
        alpha_theta_opt = best_params["alpha_theta"]
        alpha_w_opt = best_params["alpha_w"]
        dropout_p_opt = best_params["dropout_p"]
        return HyperParams(hidden_sizes_theta, hidden_sizes_w, alpha_theta_opt, alpha_w_opt, gamma_opt, dropout_p_opt)

    def copy(self, **kwargs):
        data = self.__dict__.copy()
        data.update(kwargs)
        return HyperParamsRanges(**data)
