import os.path as osp

import gym.spaces  # noqa

from utils import tf_util as U
from argparser import bc_argparser
from experiment_initializer import ExperimentInitializer
from env_makers import make_env
from utils.misc_util import set_global_seeds
from bc_agent import BCAgent
import behavioral_cloning
from demo_dataset import DemoDataset


def clone(args):
    """Train a behavioral cloning policy"""

    # Create a single-threaded session
    U.single_threaded_session().__enter__()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    # Initialize and configure experiment
    experiment = ExperimentInitializer(args, comm=comm)
    experiment.configure_logging()
    # Create experiment name
    experiment_name = experiment.get_long_name()
    # Seedify
    rank = comm.Get_rank()
    worker_seed = args.seed + 1000000 * rank
    set_global_seeds(worker_seed)
    # Create environment
    name = "{}.worker_{}".format(args.task, rank)
    env = make_env(args.env_id, worker_seed, name, args.horizon)

    def bc_agent_wrapper(name):
        return BCAgent(name=name, env=env, hps=args)

    # Create the expert demonstrations dataset from expert trajectories
    dataset = DemoDataset(expert_arxiv=args.expert_path, size=args.num_demos,
                          train_fraction=0.7, randomize=True)

    comm.Barrier()

    # Train via behavioral cloning
    behavioral_cloning.learn(comm=comm,
                             env=env,
                             bc_agent_wrapper=bc_agent_wrapper,
                             experiment_name=experiment_name,
                             ckpt_dir=osp.join(args.checkpoint_dir, experiment_name),
                             summary_dir=osp.join(args.summary_dir, experiment_name),
                             expert_dataset=dataset,
                             lr=args.lr,
                             batch_size=args.batch_size,
                             max_iters=args.num_iters)

    # Close environment
    env.close()


def evaluate_bc_policy(args):
    """Evaluate a policy trained via behavioral cloning"""
    # Create a single-threaded session
    U.single_threaded_session().__enter__()
    # Initialize and configure experiment
    experiment = ExperimentInitializer(args)
    experiment.configure_logging()
    # Seedify
    set_global_seeds(args.seed)
    # Create environment
    env = make_env(args.env_id, args.seed, args.task, args.horizon)

    def bc_agent_wrapper(name):
        return BCAgent(name=name, env=env, hps=args)

    # Train via behavioral cloning
    behavioral_cloning.evaluate(env=env,
                                bc_agent_wrapper=bc_agent_wrapper,
                                num_trajs=args.num_trajs,
                                render=args.render,
                                exact_model_path=args.exact_model_path,
                                model_ckpt_dir=args.model_ckpt_dir)

    # Close environment
    env.close()


if __name__ == '__main__':
    _args = bc_argparser().parse_args()
    if _args.task == 'clone':
        clone(_args)
    elif _args.task == 'evaluate_bc_policy':
        evaluate_bc_policy(_args)
    else:
        raise NotImplementedError
