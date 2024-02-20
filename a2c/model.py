import sys
sys.path.append('/home/leo/DRL_Personalized_AT/')
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
from gym import spaces
import math
from math import exp, log
import random
import copy
import scipy.integrate
import os
import contextlib
import logging
from a2c.abm_gpu import perform_checks
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.animation import PillowWriter, FuncAnimation

# hyperparameters
learning_rate = 1e-4

# Constants
GAMMA = 0.9999
max_steps = 8000
num_episodes = 100000


class ActorCritic(nn.Module):
    def __init__(self, architecture_kws={}, device='cpu', batch=False, batch_size=10):
        super(ActorCritic, self).__init__()

        defaults_architecture = {
            'n_values_size': 1,  # Number of prior tumor sizes provided as inputs
            'n_values_delta': 0,  # Number of prior tumor size changes provided as inputs
            'n_inputs': 1,  # Total number of inputs. Provided separately in case we want to play with additional variables as input (e.g. initial growth rate)
            'architecture': [128, 64, 32, 16, 10],  # Size of hidden layers
            'n_doseOptions': 2  # Agent can choose 2 different intensities of chemotherapy
        }
        for key in defaults_architecture.keys():
            architecture_kws[key] = architecture_kws.get(key, defaults_architecture[key])
        # Reward funtion

        self.device = device
        self.num_actions = architecture_kws['n_doseOptions']
        num_inputs = architecture_kws['n_inputs']
        self.architecture = architecture_kws['architecture']
        self.firstLSTM = nn.LSTM(input_size=num_inputs, hidden_size=self.architecture[0], batch_first=False)
        self.hidden_layers = nn.ModuleList()
        # self.hidden_layers.append(nn.Linear(1, self.architecture[0]))
        in_features = self.architecture[0]
        for out_features in self.architecture[1:]:
            self.hidden_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
        self.value = nn.Linear(in_features, 1)
        self.policy = nn.Linear(in_features, self.num_actions)
        self.batch = batch
        self.batch_size = batch_size
        # Here, we still need weights initialisation of these two output layers

    def forward(self, state):
        if not self.batch:
            state = Variable(torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)).to(self.device)
        else:
            state = Variable(torch.from_numpy(state).float().unsqueeze(0).unsqueeze(2)).to(self.device)
        _, (h_n, _) = self.firstLSTM(state)
        out = h_n[-1]
        for layer in self.hidden_layers:
            out = F.relu(layer(out))
        value = self.value(out)
        if not self.batch:
            policy_dist = F.softmax(self.policy(out), dim=0)
        else:
            policy_dist = F.softmax(self.policy(out), dim=1)

        return value, policy_dist


def a2c(env, num_batch=10, batch_operation=False, architecture_kws={}, model_path='/home/yunli/DRL_Personalized_AT/a2c/torch_training/', learning_rate=1e-4, max_steps=8000, num_episodes=100000, GAMMA=0.9999, save_interval=10000, device='cpu'):

    actor_critic = ActorCritic(architecture_kws=architecture_kws, device=device, batch=batch_operation, batch_size=num_batch).to(device)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    if not batch_operation:
        entropy_term = 0
    else:
        entropy_term = np.zeros(num_batch)

    for episode in range(num_episodes):
        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        treatment_history = ''
        for steps in range(max_steps):
            value, policy_dist = actor_critic.forward(state)
            if str(device) == 'cuda':
                value = value.cpu()
                policy_dist = policy_dist.cpu()
            if not batch_operation:
                value = value.detach().numpy()[0]
            else:
                value = value.detach().numpy()
            dist = policy_dist.detach().numpy()

            if not batch_operation:
                action = np.random.choice(2, p=np.squeeze(dist))
            else:
                action = policy_dist.multinomial(1).squeeze(1).numpy()
            if not batch_operation:
                if action == 0:
                    treatment_history += 'T'
                else:
                    treatment_history += 'H'
                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = -np.sum(np.mean(dist) * np.log(dist))
                new_state, reward, done, _ = env.step(action)
                if not env.success:
                    break
            else:
                log_prob = torch.zeros(num_batch)
                entropy = np.zeros(num_batch)
                new_state = np.zeros(num_batch)
                reward = np.zeros(num_batch)
                done = np.zeros(num_batch)
                for ind, act in enumerate(action):
                    log_prob[ind] = torch.log(policy_dist[ind].squeeze(0)[act])
                    entropy[ind] = -np.sum(np.mean(dist[ind]) * np.log(dist[ind]))
                new_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += np.mean(entropy)
            state = new_state

            if not batch_operation:
                if done or steps == max_steps-1:
                    Qval, _ = actor_critic.forward(new_state)
                    if str(device) == 'cuda':
                        Qval = Qval.cpu()
                    Qval = Qval.detach().numpy()[0]
                    log_msg = "- Episode %i - %i x %i = %i. Total score: %f. Treatment history is %s" % (episode, steps + 1, env.dt, (steps + 1) * env.dt, np.sum(rewards), treatment_history)
                    logging.info(log_msg)
                    break
            else:
                if done.all() or steps == max_steps-1:
                    Qval, _ = actor_critic.forward(new_state)
                    if str(device) == 'cuda':
                        Qval = Qval.cpu()
                    Qval = Qval.detach().numpy()[0]
                    sum_reward = [np.sum(ite) for ite in rewards]
                    log_msg = "- Episode %i - Maximum days achieved within this batch: %i x %i = %i. Batch averaged score: %f" % (episode, steps + 1, env.dt, (steps + 1) * env.dt, np.mean(sum_reward))
                    logging.info(log_msg)
                    break

        if not env.success:
            print('Episode {} is discarded since no solution founded, treatment history is {}'.format(episode, treatment_history))
            continue
        # compute Q values
        if batch_operation:
            values = np.array(values).squeeze(2)
        Qvals = np.zeros_like(values)
        if not batch_operation:
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + GAMMA * Qval
                Qvals[t] = Qval
        else:
            for ind in range(num_batch):
                for t in reversed(range(steps + 1)):
                    Qval = rewards[t][ind] + GAMMA * Qval
                    Qvals[t, ind] = Qval

        # update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        # print(advantage.shape)
        if not batch_operation:
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
        else:
            actor_loss = (-log_probs * advantage).mean(dim=0)
            critic_loss = 0.5 * advantage.pow(2).mean(dim=0)
        ac_loss = actor_loss + critic_loss + 0.001 * torch.tensor(entropy_term)
        if batch_operation:
            ac_loss = ac_loss.mean()

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

        if episode != 0:
            if episode % save_interval == 0:
                torch.save(actor_critic.state_dict(), model_path + 'a2c_model_' + str(episode) + '.pth')

    # Save the model
    torch.save(actor_critic.state_dict(), model_path + 'a2c_model_final.pth')

    # # Plot results
    # smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    # smoothed_rewards = [elem for elem in smoothed_rewards]
    # plt.plot(all_rewards)
    # plt.plot(smoothed_rewards)
    # plt.plot()
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.show()

    # plt.plot(all_lengths)
    # plt.plot(average_lengths)
    # plt.xlabel('Episode')
    # plt.ylabel('Episode length')
    # plt.show()


class SimulationEnv(gym.Env):
    def __init__(self, model, treatment_period, reward_kws={}):
        super(SimulationEnv, self).__init__()
        self.n0 = model.n0
        self.tumor_size = self.n0
        self.time = 0
        self.dt = treatment_period
        self.result_df = None

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.array([0.0]), high=np.array([1.0]))
        defaults_reward = {
            'base': 0.1,  # Base reward given
            'hol': 0.05,  # Addition for a treatment holiday
            'punish': -0.1,  # Punishment for tumor progression within simulation time frame
            'drug_use': 0  # Reward for "correct" drug use
        }
        for key in defaults_reward.keys():
            reward_kws[key] = reward_kws.get(key, defaults_reward[key])
        self.reward_kws = reward_kws
        self.model = model
        self.plot = []

    def step(self, action):
        t_start = self.time
        t_end = self.time + self.dt
        if action == 0:
            result_df = self.model.simulate([t_start, t_end, self.model.dose_max])
        elif action == 1:
            result_df = self.model.simulate([t_start, t_end, 0])
        self.success = self.model.success
        if self.result_df is None:
            self.result_df = result_df
        else:
            self.result_df = pd.concat([self.result_df, result_df])
        if isinstance(self.model, ABMModel):
            self.plot += self.model.plate_state_log

        reward, done = self.calculate_reward(self.time, action)
        info = {}
        self.time += self.dt
        self.tumor_size = np.array(result_df.loc[0, 'TumourSize'])

        return self.tumor_size, reward, done, info

    def calculate_reward(self, time, action):
        tumor_size = self.tumor_size
        done = False
        reward = self.reward_kws['base']
        if tumor_size > 1.2 * self.n0:
            done = True
            reward = self.reward_kws['punish']
            # print("Punished for tumor size " + str(tumor_size))
        elif time > 20 * 365:
            done = True
            reward += 5
        if not done:
            if tumor_size > self.n0 and action == 1:  # Punishment for near failure
                reward += self.reward_kws['punish']
            if action == 1:  # Holiday Reward
                reward += self.reward_kws['hol']
        return reward, done

    def reset(self):
        self.tumor_size = self.n0
        self.result_df = None
        self.time = 0
        self.model.result_state_vec = copy.deepcopy(self.model.initial_dist)
        return self.tumor_size


class VectorisedEnv(gym.Env):
    """Vectorized wrapper for the SimulationEnv"""
    def __init__(self, env_class, n_envs, model, treatment_period, reward_kws={}):
        self.envs = [env_class(model, treatment_period, reward_kws) for _ in range(n_envs)]
        self.n_envs = n_envs

        # Assuming all environments have the same observation and action space
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.dt = self.envs[0].dt

    def reset(self):
        """Reset all environments and return an array of observations."""
        observations = [env.reset() for env in self.envs]
        return np.array(observations)

    def step(self, actions):
        """Step through all environments with the given list of actions."""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        observations, rewards, dones, infos = zip(*results)
        self.success = all([env.success for env in self.envs])
        return np.array(observations), np.array(rewards), np.array(dones), np.array(infos)


class Simulation():
    def __init__(self):
        pass

    def simulate(self):
        pass

    # Define the model mapping cell counts to observed fluorescent area
    def _run_cell_count_to_tumour_size_model(self, popModelSolDf):
        # Note: default scaleFactor value assumes a cell radius of 10uM. Volume is given in mm^3 -> r^3 = (10^-2 mm)^3 = 10^-6
        theta = self.scale_factor
        return theta * (np.sum(popModelSolDf[self.state_vars].values, axis=1))


class ABMModel(Simulation):
    def __init__(self, **kwargs):
        self.dt = float(kwargs.get('dt', 1))  # Time resolution to return the model prediction on
        self.side_len = int(kwargs.get('l', 100))
        self.r_s = kwargs.get('rS', 0.027)
        self.c_r = kwargs.get('cR', 0.25)
        self.d_cell = kwargs.get('dCell', 0.25)
        self.d_drug = kwargs.get('dDrug', 0.75)
        self.scale_factor = kwargs.get('scaleFactor', 1)
        self.initial_cell_den = kwargs.get('initialCellDen', 0.5)
        self.initial_resist_den = kwargs.get('initialResistDen', 0.05)
        self.initial_dist = kwargs.get('initialDist', None)
        self.state_vars = ['P1']
        self.dose_max = kwargs.get('DMax', 1)
        self.success = True
        if type(self.initial_dist) is float:
            if math.isnan(self.initial_dist):
                self.initial_dist = None

        if self.initial_dist is None:
            pos_cell = np.random.choice(range(0, self.side_len ** 2), round(self.initial_cell_den * self.side_len ** 2), replace=False)
            pos_resist_cell = random.sample(list(pos_cell), round(len(pos_cell) * self.initial_resist_den))
            curr_state_vec = [0] * self.side_len ** 2
            for pos in pos_cell:
                curr_state_vec[pos] = 1
            for pos in pos_resist_cell:
                curr_state_vec[pos] = 2
            curr_state_vec = np.array(curr_state_vec).reshape(self.side_len, self.side_len)
            self.result_state_vec = copy.deepcopy(curr_state_vec)
        else:
            self.result_state_vec = copy.deepcopy(np.array(self.initial_dist))
        self.n0 = np.array((np.sum(curr_state_vec == 1) + np.sum(curr_state_vec == 2)) / self.side_len ** 2)
        self.initial_dist = copy.deepcopy(self.result_state_vec)

    def simulate(self, treatment_schedule_list):
        curr_state_vec = copy.deepcopy(self.result_state_vec)
        # Seems to be a bug here, no matter which intervalID is, due to exclusion of end points of np.arange, need to add one extra self.dt
        # tVec = np.arange(interval[0], interval[1], self.dt)
        # if intervalId == (len(treatmentScheduleList) - 1):
        t_start = treatment_schedule_list[0]
        t_end = treatment_schedule_list[1]
        dose = treatment_schedule_list[2]

        # threshold_s = [self.r_s, self.r_s * (1 + self.d_cell), 1]
        # threshold_r = [(1 - self.c_r) * self.r_s, (1 - self.c_r) * self.r_s + self.r_s * self.d_cell, 1]

        self.plate_state_log = []
        for t in range(t_start, t_end):
            self.plate_state_log.append(copy.deepcopy(curr_state_vec))
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device("cpu")
            grid = torch.tensor(curr_state_vec, device=device)
            padded_grid = F.pad(grid, (1, 1, 1, 1), 'constant', -1)
            patches = padded_grid.unfold(0, 3, 1).unfold(1, 3, 1)
            patches = patches.contiguous().view(-1, 3*3)

            columns_to_keep = [1, 3, 4, 5, 7]

            # Select the desired columns
            patches = patches[:, columns_to_keep]
            res = perform_checks(patches, self, dose)
            for key in res:
                curr_state_vec[key] = res[key]

            # for i in range(self.side_len):
            #     for j in range(self.side_len):
            #         if curr_state_vec[i, j] == 0:
            #             pass
            #         elif curr_state_vec[i, j] == 100:
            #             pass
            #         elif curr_state_vec[i, j] == 10:
            #             pass
            #         elif curr_state_vec[i, j] == 1:
            #             rand = np.random.rand()
            #             if rand < threshold_s[0]:
            #                 curr_neighb = []
            #                 if i == 0:
            #                     curr_neighb.append(None)
            #                 else:
            #                     curr_neighb.append([(i - 1, j), curr_state_vec[i - 1, j]])
            #                 if i == self.side_len - 1:
            #                     curr_neighb.append(None)
            #                 else:
            #                     curr_neighb.append([(i + 1, j), curr_state_vec[i + 1, j]])
            #                 if j == 0:
            #                     curr_neighb.append(None)
            #                 else:
            #                     curr_neighb.append([(i, j - 1), curr_state_vec[i, j - 1]])
            #                 if j == self.side_len - 1:
            #                     curr_neighb.append(None)
            #                 else:
            #                     curr_neighb.append([(i, j + 1), curr_state_vec[i, j + 1]])
            #                 curr_neighb = [item for item in curr_neighb if item is not None]
            #                 curr_neighb = [item for item in curr_neighb if item[1] == 0]
            #                 if curr_neighb:
            #                     rand_kill = np.random.rand()
            #                     if rand_kill < dose * self.d_drug:
            #                         curr_state_vec[i, j] == 0
            #                     else:
            #                         pos_proliferate = random.sample(curr_neighb, 1)[0][0]
            #                         curr_state_vec[pos_proliferate] = 10
            #             elif rand < threshold_s[1]:
            #                 curr_state_vec[i, j] = 0
            #             else:
            #                 pass
            #         elif curr_state_vec[i, j] == 2:
            #             rand = np.random.rand()
            #             if rand < threshold_r[0]:
            #                 curr_neighb = []
            #                 if i == 0:
            #                     curr_neighb.append(None)
            #                 else:
            #                     curr_neighb.append([(i - 1, j), curr_state_vec[i - 1, j]])
            #                 if i == self.side_len - 1:
            #                     curr_neighb.append(None)
            #                 else:
            #                     curr_neighb.append([(i + 1, j), curr_state_vec[i + 1, j]])
            #                 if j == 0:
            #                     curr_neighb.append(None)
            #                 else:
            #                     curr_neighb.append([(i, j - 1), curr_state_vec[i, j - 1]])
            #                 if j == self.side_len - 1:
            #                     curr_neighb.append(None)
            #                 else:
            #                     curr_neighb.append([(i, j + 1), curr_state_vec[i, j + 1]])
            #                 curr_neighb = [item for item in curr_neighb if item is not None]
            #                 curr_neighb = [item for item in curr_neighb if item[1] == 0]
            #                 if curr_neighb:
            #                     pos_proliferate = random.sample(list(curr_neighb), 1)[0][0]
            #                     curr_state_vec[pos_proliferate] = 100
            #             elif rand < threshold_r[1]:
            #                 curr_state_vec[i, j] = 0
            #             else:
            #                 pass
            # curr_state_vec = curr_state_vec.reshape(1, -1)[0]
            # curr_state_vec = np.array([1 if item == 10 else item for item in curr_state_vec])
            # curr_state_vec = np.array([2 if item == 100 else item for item in curr_state_vec])
            # curr_state_vec = curr_state_vec.reshape(self.side_len, self.side_len)
        num_s = [np.sum(self.result_state_vec == 1) / self.side_len ** 2, np.sum(curr_state_vec == 1) / self.side_len ** 2]
        num_r = [np.sum(self.result_state_vec == 2) / self.side_len ** 2, np.sum(curr_state_vec == 2) / self.side_len ** 2]
        self.result_state_vec = copy.deepcopy(curr_state_vec)
        results_df = pd.DataFrame({"Time": [t_start, t_end], "DrugConcentration": [dose, dose],
                                       **dict(zip(self.state_vars, [num_s, num_r]))})
        # Compute the fluorescent area that we'll see
        results_df['TumourSize'] = pd.Series(self._run_cell_count_to_tumour_size_model(results_df),
                                                  index=results_df.index)
        return results_df


class ODEModel(Simulation):
    def __init__(self, **kwargs):
        self.scale_factor = kwargs.get('scaleFactor', 1)
        self.dt = kwargs.get('dt', 1e-3)  # Time resolution to return the model prediction on
        self.absErr = kwargs.get('absErr', 1.0e-8)  # Absolute error allowed for ODE solver
        self.relErr = kwargs.get('relErr', 1.0e-6)  # Relative error allowed for ODE solver
        self.solverMethod = kwargs.get('method', 'DOP853')  # ODE solver used
        self.suppress_output = kwargs.get('suppressOutputB',
                                          False)  # If true, suppress output of ODE solver (including warning messages)
        self.success = False  # Indicate successful solution of the ODE system
        self.state_vars = ['P1']
        self.max_step = kwargs.get('max_step', 1)
        self.dose_max = kwargs.get('DMax', 1)

    def simulate(self, treatment_schedule_list):
        curr_state_vec = copy.deepcopy(self.result_state_vec)
        # Solve
        encountered_problem = False
        if self.suppress_output:
            with stdout_redirected():
                y0 = copy.copy(curr_state_vec)
                y0 = np.append(y0, treatment_schedule_list[2])
                sol_obj = scipy.integrate.solve_ivp(self.model_eqns, y0=y0,
                                                    t_span=(treatment_schedule_list[0], treatment_schedule_list[1]),
                                                    t_eval=(treatment_schedule_list[0], treatment_schedule_list[1]),
                                                    method=self.solverMethod,
                                                    atol=self.absErr, rtol=self.relErr,
                                                    max_step=self.max_step)
        else:
            y0 = copy.copy(curr_state_vec)
            y0 = np.append(y0, treatment_schedule_list[2])
            sol_obj = scipy.integrate.solve_ivp(self.model_eqns, y0=y0,
                                                t_span=(treatment_schedule_list[0], treatment_schedule_list[1]),
                                                t_eval=(treatment_schedule_list[0], treatment_schedule_list[1]),
                                                method=self.solverMethod,
                                                atol=self.absErr, rtol=self.relErr,
                                                max_step=self.max_step)
        # Check that the solver converged
        if not sol_obj.success or np.any(sol_obj.y < 0):
            self.err_message = sol_obj.message
            encountered_problem = True
            if not self.suppress_output: print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            if not sol_obj.success:
                if not self.suppress_output: print(self.err_message)
            else:
                if not self.suppress_output: print(
                    "Negative values encountered in the solution. Make the time step smaller or consider using a stiff solver.")
                if not self.suppress_output: print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            self.sol_obj = sol_obj
        # Save results
        if not encountered_problem:
            result_df = pd.DataFrame({"Time": [treatment_schedule_list[0], treatment_schedule_list[1]], "DrugConcentration": sol_obj.y[-1, :],
                                **dict(zip(self.state_vars,sol_obj.y))})
            self.result_state_vec = copy.deepcopy(sol_obj.y[:-1, -1])
        else:
            result_df = pd.DataFrame({"Time": [treatment_schedule_list[0], treatment_schedule_list[1]], "DrugConcentration": np.zeros_like([treatment_schedule_list[0], treatment_schedule_list[1]]),
                                     **dict(zip(self.state_vars,np.zeros_like([treatment_schedule_list[0], treatment_schedule_list[1]])))})
            self.result_state_vec = np.zeros_like([self.state_vars])
        # If the solver diverges in the first interval, it can't return any solution. Catch this here, and in this case
        # replace the solution with all zeros.
        # Compute the fluorescent area that we'll see
        result_df['TumourSize'] = pd.Series(self._run_cell_count_to_tumour_size_model(result_df),
                                            index=result_df.index)
        self.success = True if not encountered_problem else False
        return result_df


class LotkaVolterraModel(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "LotkaVolterraModel"
        defaults_params = {'rS': 0.027,
                          'rR': 0.0207927990105891,
                          'K': 1.0,
                          'dD': 1.5,
                          'dS': 0.0078682167556054,
                          'dR': 0.0078682167556054,
                          'S0': 0.4256703505169107,
                          'R0': 4.256746072629835e-06,
                          'DMax': 1.0}
        for key in defaults_params.keys():
            kwargs[key] = kwargs.get(key, defaults_params[key])
        self.param_dic = kwargs
        self.state_vars = ['S', 'R']
        self.initial_dist = [kwargs['S0'], kwargs['R0']]
        self.result_state_vec = copy.deepcopy(self.initial_dist)
        self.n0 = np.array(sum(self.initial_dist))

    # The governing equations
    def model_eqns(self, t, uVec):
        S, R, D = uVec
        dudtVec = np.zeros_like(uVec)
        dudtVec[0] = self.param_dic['rS'] * (1 - (S+R)/self.param_dic['K']) * (1 - self.param_dic['dD']*D) * S - self.param_dic['dS']*S
        dudtVec[1] = self.param_dic['rR'] * (1 - (S+R)/self.param_dic['K']) * R - self.param_dic['dR']*R
        dudtVec[2] = 0
        return (dudtVec)


class ExponentialModel(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ExponentialModel"
        self.paramDic = {**self.paramDic,
                        'rS': 0.01365,
                        'rR': 0.00825,
                        'Ks': 1,
                        'Kr': 0.25,
                        'dDs': 2.3205,
                        'dDr': 1.3205,
                        'S0': 0.74,
                        'R0':0.01,
                        'DMax':1,
                        'alpha':1,
                        'gamma':0.27385}
        self.stateVars = ['S', 'R']

    # The governing equations
    def ModelEqns(self, t, uVec):
        S, R, D = uVec
        dudtVec = np.zeros_like(uVec)
        try:
            dudtVec[0] = self.paramDic['rS'] * S * (1 - ((S+(R / (1+exp(self.paramDic['gamma']*t))))/self.paramDic['Ks'])**self.paramDic['alpha'] - self.paramDic['dDs']*D)
            dudtVec[1] = self.paramDic['rR'] * R * (1 - ((R+(S / (1+exp(self.paramDic['gamma']*t))))/self.paramDic['Kr'])**self.paramDic['alpha'] - self.paramDic['dDr']*D)
        except OverflowError:
            dudtVec[0] = self.paramDic['rS'] * S * (1 - ((S+(0))/self.paramDic['Ks'])**self.paramDic['alpha'] - self.paramDic['dDs']*D)
            dudtVec[1] = self.paramDic['rR'] * R * (1 - ((R+(0))/self.paramDic['Kr'])**self.paramDic['alpha'] - self.paramDic['dDr']*D)
        dudtVec[2] = 0
        return (dudtVec)  


class StemCellModel(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "StemCellModel"
        self.paramDic = {**self.paramDic,
                        'rR': log(2),
                        'beta': 1e-6,
                        'dR': 0.07,
                        'rho': 0.0001,
                        'phi': 0.01,
                        'S0': 1000,
                        'R0': 10,
                        'P0': 29,
                        'DMax': 1}
        self.stateVars = ['S', 'R', 'P']

    # The governing equations
    def ModelEqns(self, t, uVec):
        S, R, P, D = uVec
        dudtVec = np.zeros_like(uVec)
        dudtVec[0] = (1 - (R / (S + R)) * self.paramDic['beta']) * self.paramDic['rR'] * R - self.paramDic['dR'] * D * S  # Differentiated cells
        dudtVec[1] = (R / (S + R)) * self.paramDic['beta'] * self.paramDic['rR'] * R  # Stem-like (drug resistant) cells
        dudtVec[2] = self.paramDic['rho'] * S - self.paramDic['phi'] * P
        dudtVec[3] = 0
        return (dudtVec)


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def run_training(model, num_batch=10, batch_operation=False, treatment_period=30, device='cpu', save_interval=10000, reward_kws={}, architecture_kws={}, path='/home/yunli/DRL_Personalized_AT/a2c/torch_training/', model_name='a2c_default', learning_rate=1e-4, max_step=8000, num_episodes=100000, GAMMA=0.9999):
    model_folder_path = path + model_name
    if os.path.exists(model_folder_path):
        model_folder_path += '_a'
    while os.path.exists(model_folder_path):
        model_folder_path += 'a'
    model_folder_path += '/'
    os.mkdir(model_folder_path)
    logging.basicConfig(filename=model_folder_path + 'training_log.log', level=logging.INFO, format='%(asctime)s %(message)s')
    if not batch_operation:
        env = SimulationEnv(model=model, treatment_period=treatment_period, reward_kws=reward_kws)
    else:
        env = VectorisedEnv(SimulationEnv, n_envs=num_batch, model=model, treatment_period=treatment_period, reward_kws=reward_kws)
    a2c(env=env, architecture_kws=architecture_kws, model_path=model_folder_path, learning_rate=learning_rate, max_steps=max_step, num_episodes=num_episodes, GAMMA=GAMMA, save_interval=save_interval, device=device, num_batch=num_batch, batch_operation=batch_operation)


def run_prediction(model, treatment_period=30, reward_kws={}, architecture_kws={}, model_store_path='/home/yunli/DRL_Personalized_AT/a2c/torch_training/a2c_default/a2c_model_final.pth'):
    loaded_actor_critic = ActorCritic(architecture_kws)
    loaded_actor_critic.load_state_dict(torch.load(model_store_path, map_location=torch.device('cpu')))
    loaded_actor_critic.eval()
    env = SimulationEnv(model, treatment_period, reward_kws)
    output_df = pd.DataFrame(columns=['Time', 'TumourSize', 'Support_Hol', 'Support_Treat', 'Action'])
    with torch.no_grad():  # This will temporarily set all the requires_grad flag to false
        state = env.reset()
        done = False
        while not done:
            _, dist = loaded_actor_critic(state)
            action = np.random.choice(2, p=np.squeeze(dist).detach().numpy())
            state, _, done, _ = env.step(action)
            if action == 0:
                Action = 'T'
            else:
                Action = 'H'
            new_data = {'Time': env.time, 'TumourSize': env.tumor_size, 'Support_Hol': dist[1].detach().numpy(), 'Support_Treat': dist[0].detach().numpy(), 'Action': Action}
            output_df.loc[len(output_df)] = new_data
    model_path_list = model_store_path.split('/')
    model_path = '/'.join(model_path_list[:-1]) + '/'
    output_df.to_csv(model_path + 'prediction.csv')
    if isinstance(model, ABMModel):
        num_frames = len(env.plot)

        def update(frame_number):
            data = env.plot[frame_number]
            ax.clear()
            
            # Define a custom colormap
            cmap = mcolors.ListedColormap(['white', 'green', 'red'])
            bounds = [-0.5, 0.5, 1.5, 2.5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            # Create the plot
            ax.imshow(data, cmap=cmap, norm=norm)
            ax.axis('off')
            
            # You might need to adjust the legend for each frame or create it once outside the update function
            red_patch = mpatches.Patch(color='red', label='Sensitive')
            green_patch = mpatches.Patch(color='green', label='Resistant')
            ax.legend(handles=[red_patch, green_patch], loc='upper right')

        # Create the initial figure and axes
        fig, ax = plt.subplots(figsize=(10, 10))

        # Create the animation
        ani = FuncAnimation(fig, update, frames=num_frames)

        # Save the animation as a GIF
        ani.save(model_path + 'transfer_learning.gif', writer=PillowWriter(fps=treatment_period))
