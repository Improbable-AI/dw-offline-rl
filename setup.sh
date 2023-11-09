export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/d3rlpy:$PYTHONPATH
export PYTHONPATH=$(pwd)/TD3BC:$PYTHONPATH
export PYTHONPATH=$(pwd)/IQL:$PYTHONPATH
export PYTHONPATH=$(pwd)/CQL:$PYTHONPATH
export WANDB_DISABLE_SERVICE=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=osmesa

export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1