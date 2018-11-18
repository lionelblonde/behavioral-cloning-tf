import time
import copy
import os.path as osp
from collections import deque

import numpy as np

from utils import tf_util as U
import logger
from utils.misc_util import prettify_time
from mpi_adam import MpiAdamOptimizer
from utils.summary_util import CustomSummary


def learn(comm,
          env,
          bc_agent_wrapper,
          experiment_name,
          ckpt_dir,
          summary_dir,
          expert_dataset,
          lr,
          batch_size,
          max_iters):

    rank = comm.Get_rank()

    # Create the BC agent
    pol = bc_agent_wrapper('pol')

    # Create mpi adam optimizer for the policy
    pol_optimizer = MpiAdamOptimizer(comm, clip_norm=pol.hps.clip_norm,
                                     learning_rate=lr, name='pol_adam')
    _optimize_pol = pol_optimizer.minimize(pol.loss, var_list=pol.trainable_vars)

    # Retrieve already-existing placeholders
    e_obs = U.get_placeholder_cached(name='e_obs')
    e_acs = U.get_placeholder_cached(name='e_acs')

    # Create Theano-like ops
    optimize_pol = U.function([e_obs, e_acs], _optimize_pol)

    # Initialize variables
    U.initialize()
    # Sync params of all processes with the params of the root process
    pol_optimizer.sync_from_root(pol.trainable_vars)

    if rank == 0:
        # Create summary writer
        writer = U.file_writer(summary_dir)
        # Create the summary
        _names = ['train_loss', 'val_loss']
        _summary = CustomSummary(scalar_keys=_names, family="bc")

    # Define the origin of time
    tstart = time.time()

    # Define rolling buffers for loss collection
    maxlen = 100
    pol_train_loss_buffer = deque(maxlen=maxlen)
    pol_val_loss_buffer = deque(maxlen=maxlen)

    for iters_so_far in range(max_iters):

        # Verify that the processes are still in sync
        if iters_so_far > 0 and iters_so_far % 10 == 0:
            pol_optimizer.check_synced(pol.trainable_vars)

        # Save the model
        if rank == 0 and iters_so_far % int(1e4) == 0 and ckpt_dir is not None:
            model_path = osp.join(ckpt_dir, experiment_name)
            U.save_state(model_path, iters_so_far=iters_so_far)
            logger.info("saving model")
            logger.info("  @: {}".format(model_path))

        # Make non-zero-rank workers wait for rank zero
        comm.Barrier()

        # Go through mini-batches of the demonstration dataset, training fraction
        obs, acs = expert_dataset.get_next_pair_batch(batch_size, 'train')
        # Update running mean and std on states
        if hasattr(pol, "obs_rms"):
            pol.obs_rms.update(obs, comm)
        # Perform a gradient step to update the policy parameters
        optimize_pol(obs, acs)
        # Compute training loss
        pol_train_loss = pol.compute_pol_loss(obs, acs)
        pol_train_loss_buffer.append(pol_train_loss)
        # Go through mini-batches of the demonstration dataset, validation fraction
        obs, acs = expert_dataset.get_next_pair_batch(-1, 'val')
        # Compute validation loss
        pol_val_loss = pol.compute_pol_loss(obs, acs)
        pol_val_loss_buffer.append(pol_val_loss)

        if iters_so_far % 100 == 0:
            # Log training and validation losses
            logger.info(('iter #{} '
                         '| train loss: {} '
                         '| val loss: {} '
                         '| elapsed: {}').format(iters_so_far,
                                                 pol_train_loss,
                                                 pol_val_loss,
                                                 prettify_time(time.time() - tstart)))

        # Prepare losses to be dumped in summaries
        all_summaries = [np.mean(pol_train_loss_buffer),
                         np.mean(pol_val_loss_buffer)]  # must be visible by all workers
        if rank == 0:
            assert len(_names) == len(all_summaries), "mismatch in list lengths"
            _summary.add_all_summaries(writer, all_summaries, iters_so_far)


def traj_ep_generator(env, pol, render):
    """Generator that spits out a trajectory collected during a single episode
    `append` operation is also significantly faster on lists than numpy arrays,
    they will be converted to numpy arrays once complete and ready to be yielded.
    """
    ob = env.reset()
    cur_ep_len = 0
    cur_ep_env_ret = 0
    obs = []
    acs = []
    env_rews = []

    while True:
        ac = pol.predict(ob)
        obs.append(ob)
        acs.append(ac)
        if render:
            env.render()
        new_ob, env_rew, done, _ = env.step(ac)
        env_rews.append(env_rew)
        cur_ep_len += 1
        cur_ep_env_ret += env_rew
        ob = copy.copy(new_ob)
        if done:
            obs = np.array(obs)
            acs = np.array(acs)
            env_rews = np.array(env_rews)
            yield {"obs": obs,
                   "acs": acs,
                   "env_rews": env_rews,
                   "ep_len": cur_ep_len,
                   "ep_env_ret": cur_ep_env_ret}
            cur_ep_len = 0
            cur_ep_env_ret = 0
            obs = []
            acs = []
            env_rews = []
            ob = env.reset()


def evaluate(env,
             bc_agent_wrapper,
             num_trajs,
             render,
             exact_model_path=None,
             model_ckpt_dir=None):
    """Evaluate a trained SAM agent"""

    # Only one of the two arguments can be provided
    assert sum([exact_model_path is None, model_ckpt_dir is None]) == 1

    # Rebuild the computational graph
    pol = bc_agent_wrapper('pol')
    # Create episode generator
    traj_gen = traj_ep_generator(env, pol, render)
    # Initialize and load the previously learned weights into the freshly re-built graph
    U.initialize()
    if exact_model_path is not None:
        U.load_model(exact_model_path)
        logger.info("model loaded from exact path:\n  {}".format(exact_model_path))
    else:  # `exact_model_path` is None -> `model_ckpt_dir` is not None
        U.load_latest_checkpoint(model_ckpt_dir)
        logger.info("model loaded from ckpt dir:\n  {}".format(model_ckpt_dir))
    # Initialize the history data structures
    ep_lens = []
    ep_env_rets = []
    # Collect trajectories
    for i in range(num_trajs):
        logger.info("evaluating [{}/{}]".format(i + 1, num_trajs))
        traj = traj_gen.__next__()
        ep_len, ep_env_ret = traj['ep_len'], traj['ep_env_ret']
        # Aggregate to the history data structures
        ep_lens.append(ep_len)
        ep_env_rets.append(ep_env_ret)
    # Log some statistics of the collected trajectories
    ep_len_mean = np.mean(ep_lens)
    ep_env_ret_mean = np.mean(ep_env_rets)
    logger.record_tabular("ep_len_mean", ep_len_mean)
    logger.record_tabular("ep_env_ret_mean", ep_env_ret_mean)
    logger.dump_tabular()
