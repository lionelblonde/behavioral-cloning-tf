from utils.misc_util import boolean_flag


def argparse(description):
    """Create an empty argparse.ArgumentParser"""
    import argparse
    return argparse.ArgumentParser(description=description,
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def bc_argparser(description="Behavioral Cloning Experiment"):
    """Create an argparse.ArgumentParser for behavioral cloning-related tasks"""
    parser = argparse(description)
    parser.add_argument('--note', help='w/e', type=str, default=None)
    parser.add_argument('--env_id', help='environment identifier', default='Hopper-v2')
    parser.add_argument('--horizon', help='maximum number of timesteps in an episode',
                        type=int, default=None)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--checkpoint_dir', help='directory to save the models',
                        default=None)
    parser.add_argument('--log_dir', help='directory to save the log files',
                        default='data/logs')
    parser.add_argument('--summary_dir', help='directory to save the summaries',
                        default='data/summaries')
    parser.add_argument('--task', help='task to carry out', type=str,
                        choices=['clone',
                                 'evaluate_bc_policy'],
                        default='clone')
    parser.add_argument('--expert_path', help='.npz archive containing the demos',
                        type=str, default=None)
    parser.add_argument('--num_demos', help='number of expert demo trajs for imitation',
                        type=int, default=None)
    parser.add_argument('--save_frequency', help='save model every xx iterations',
                        type=int, default=100)
    parser.add_argument('--num_iters', help='cummulative number of iterations since launch',
                        type=int, default=int(1e6))
    parser.add_argument('--batch_size', help='minibatch size', type=int, default=64)
    parser.add_argument('--lr', help='adam learning rate', type=float, default=3e-4)
    parser.add_argument('--clip_norm', type=float, default=None)
    boolean_flag(parser, 'render', help='whether to render the interaction traces', default=False)
    parser.add_argument('--num_trajs', help='number of trajectories to evaluate/gather',
                        type=int, default=10)
    parser.add_argument('--exact_model_path', help='exact path of the model',
                        type=str, default=None)
    parser.add_argument('--model_ckpt_dir', help='checkpoint directory containing the models',
                        type=str, default=None)
    parser.add_argument('--demos_dir', type=str, help='directory to save the demonstrations',
                        default='data/expert_demonstrations')
    boolean_flag(parser, 'rmsify_obs', default=True)
    parser.add_argument('--hid_widths', nargs='+', type=int, default=[64, 64])
    parser.add_argument('--hid_nonlin', type=str, default='leaky_relu',
                        choices=['relu', 'leaky_relu', 'prelu', 'elu', 'selu', 'tanh'])
    parser.add_argument('--hid_w_init', type=str, default='he_normal',
                        choices=['he_normal', 'he_uniform', 'xavier_normal', 'xavier_uniform'])
    return parser
