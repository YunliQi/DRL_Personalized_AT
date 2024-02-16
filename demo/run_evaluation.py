import sys
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from itertools import product
from tqdm import tqdm
sys.path.append("../utils")
from myUtils import mkdir
from LotkaVolterraModel import LotkaVolterraModel, ExponentialModel, StemCellModel
from drlModel import A3C_Network


def run_evaluation(model_path, patients_to_evaluate, architecture_kws={}, n_replicates=100, updating_interval=7, results_path="./", results_file_name="results.csv",
                   ODE_model=LotkaVolterraModel, tqdm_output=sys.stderr, verbose=0, seed=42):
    '''
    Evaluate the model by simulating a patient drug administration policy chosen by the DRL.
    :param model_path: path to the model directory
    :param architecture_kws: dictionary with variables defining the network architecture
    :param results_path: path to directory to save the results
    :param model_loaded_name: name of the model to load
    :param n_replicates: number of replicates to simulate
    :param paramDic: dictionary of parameters for the model
    :param verbose: verbosity level
    :return:
    '''
    # Setup env and parse input
    mkdir(results_path);
    np.random.seed(seed)
    tf.set_random_seed(seed)
    if isinstance(patients_to_evaluate, pd.DataFrame): 
        patientsDf = patients_to_evaluate
    elif isinstance(patients_to_evaluate, dict): 
        patientsDf = pd.DataFrame.from_dict(patients_to_evaluate)

    if isinstance(ODE_model, str):
        ODE_model = {'LotkaVolterraModel': LotkaVolterraModel,
                     'ExponentialModel': ExponentialModel,
                     'StemCellModel': StemCellModel}[ODE_model]
    
    # Set up A3C network, which consists of one global network, and num_workers copies
    # which are used during training to increase learning performance.
    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        # Generate global network
        master_network = A3C_Network('global', None, architecture_kws=architecture_kws)
        saver = tf.train.Saver(max_to_keep=None)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)

        # Create a data log of the input vectors and the decision results
        timestep_summmaryList = []

        # Evaluate it
        if isinstance(tqdm_output, str): tqdm_output = open(tqdm_output, 'w')
        for patientId, replicateId in tqdm(list(product(patientsDf.PatientId.unique(),np.arange(n_replicates))), file=tqdm_output):
            # Initialize the ODE model instance & parameterize it
            model = ODE_model(method='RK45', dt=updating_interval)

            # Load the parameter set for this virtual patient from the database with the evaluation parameter sets.
            currParamsDic = patientsDf.loc[patientsDf.PatientId==patientId].iloc[0].to_dict()
            model.SetParams(**currParamsDic)
            n0 = currParamsDic['n0']

            # Book keeping variables
            done = False
            num_treatments = 0
            patient_action_history = ""
            currPatient_reward = 0
            currPatient_nCycles = 0
            t_start = 0
            t_end = 0
            a = 0

            # This is the treatment loop
            while not done:
                # Set the new end time
                t_start = t_end

                t_end += updating_interval  # Regular analysis
                # t_end += updating_interval + currParamsDic['sigma']  # For benchmarking tau and sigma sens analysis
                # t_end += updating_interval + int(np.random.exponential(scale=currParamsDic['sigma']))
                num_treatments += 1  # Increment the treatment counter

                # Get the current state to provide as input to the DRL model
                # 1. Tumor sizes
                # Get the state as the initial tumor size and the tumor sizes each day for the past week.
                # In case this is the start of the treatment cycle, the input vector is just the
                # initial tumor size and the 0's.
                results = model.resultsDf
                observations_sizes = []
                n_values_size = master_network.architecture_kws['n_values_size']
                for i in range(n_values_size):
                    try:
                        # IF not, then the observation is the last n_values_size time steps
                        observations_sizes.append(results['TumourSize'].iloc[-(i + 1)])
                    except TypeError:
                        # resultsDF doesn't exist b/c we haven't simulated anything
                        observations_sizes.append(n0)
                    except IndexError:
                        # we have not yet reached n_values_size time steps
                        observations_sizes.append(0)

                observations_sizes = np.array(observations_sizes)/n0

                # In addition, the network uses the per-day changes for each day over the past week,
                # which we compute here.
                observations_deltas = []
                n_values_delta = master_network.architecture_kws['n_values_delta']
                for i in range(n_values_delta):
                    observations_deltas.append(observations_sizes[i + 1] - observations_sizes[i])
                    
                network_input = np.concatenate([observations_sizes, observations_deltas]) 
                network_input = np.expand_dims(network_input, axis=2) # Add a fake dimension to make it compatible with the LSTM layer which needs 3 dims
                assert not np.any(np.isnan(network_input))

                # Take an action using probabilities from policy network output.
                a_dist, v = sess.run([master_network.policy, master_network.value],
                                        feed_dict={master_network.input: [network_input]})
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a_dist == a)

                # Treat the patient and observe the response
                if a == 0:
                    # Treat with 0% strength (holiday)
                    model.Simulate([[t_start, t_end, 0]], scaleTumourVolume=False)
                    patient_action_history += "H"
                else:
                    model.Simulate([[t_start, t_end, model.paramDic['DMax']]], scaleTumourVolume=False)
                    patient_action_history += "T"

                # Calculate the reward
                results = model.resultsDf
                reward, done = master_network.calculate_reward(state=results, n0=n0, time=t_end, action=a)
                
                # Record this iteration
                currPatient_reward += reward
                currPatient_nCycles += 1
                currTumourState = results.loc[results.Time == t_start]
                timestep_summmaryList.append({'ReplicateId': replicateId, 'Time': t_start,
                                              'TumourSize': currTumourState['TumourSize'].values[0],
                                              'S': currTumourState['S'].values[0],
                                              'R': currTumourState['R'].values[0],
                                              'DrugConcentration': currTumourState['DrugConcentration'].values[0],
                                              'Support_Hol': a_dist[0][0], 'Support_Treat': a_dist[0][1],
                                              'Action': patient_action_history[-1],
                                              'PatientId': patientId})
                if verbose > 0: print("Replicate ID: " + str(replicateId) +
                                      " - Treating interval %i to %i" % (t_start, t_end) +
                                      " - Choice is: " + str(a) +
                                      " - Current size is: " + str(round(results['TumourSize'].iloc[-1], 3)))

                if done:  # Record final timestep where progression has occured
                    finTumourState = results.iloc[-1].to_frame().transpose()
                    timestep_summmaryList.append({'ReplicateId': replicateId, 'Time': t_end,
                                                 'TumourSize': finTumourState['TumourSize'].values[0],
                                                 'S': finTumourState['S'].values[0],
                                                 'R': finTumourState['R'].values[0],
                                                 'DrugConcentration': finTumourState['DrugConcentration'].values[0],
                                                 'Support_Hol': a_dist[0][0], 'Support_Treat': a_dist[0][1],
                                                 'Action': patient_action_history[-1],
                                                 'PatientId': patientId})

        # Save the results
        longitudinalTrajectoriesDf = pd.DataFrame(timestep_summmaryList)
        longitudinalTrajectoriesDf.to_csv(
            os.path.join(results_path, results_file_name))