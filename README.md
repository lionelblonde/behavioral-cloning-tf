# TensorFlow Implementation of Behavioral Cloning

We consider continuous control environments
built with the [MuJoCo](http://mujoco.org/) physics engine
and wrapped with [OpenAI Gym](https://github.com/openai/gym).

*Code makes use of helper functions inspired from the ones used in 
[Openai Baselines](https://github.com/openai/baselines),
and uses Deepmind's [Sonnet](https://github.com/deepmind/sonnet)
TensorFlow-based neural network library.*

Expert demonstrations are available [here](https://drive.google.com/drive/folders/1ihVMUk9Ewm7cHv4tpFgnDkXkxNXjDYeS?usp=sharing).

## Example usage

To imitate via Behavioral Cloning with 2 parallel MPI workers,
in the `InvertedPendulum-v2` environment, using 16 expert demonstrations
from the trajectory archive `InvertedPendulum-v2_s0_mode_d32.npz` placed in `DEMOS/`
at the project root,
execute the following command:

```shell
./mujoco_clone.sh 2 InvertedPendulum-v2 DEMOS/InvertedPendulum-v2_s0_mode_d32.npz 16
```

To evaluate a policy trained via behavioral cloning
in the `InvertedPendulum-v2` environment,
whose tensorflow checkpoints were saved in the directory
`data/imitation_checkpoints/<ckpt_name>`,
execute the following command:

```shell
./mujoco_evaluate.sh InvertedPendulum-v2 data/imitation_checkpoints/<ckpt_name>
```
