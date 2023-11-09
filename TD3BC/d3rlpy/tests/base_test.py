import os
from unittest.mock import Mock

import numpy as np

from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import MDPDataset, Transition, TransitionMiniBatch
from d3rlpy.logger import D3RLPyLogger


def base_tester(model, impl, observation_shape, action_size=2):
    # dummy impl object
    model._impl = impl

    # check save  model
    impl.save_model = Mock()
    model.save_model("model.pt")
    impl.save_model.assert_called_with("model.pt")

    # check load model
    impl.load_model = Mock()
    model.load_model("mock.pt")
    impl.load_model.assert_called_with("mock.pt")

    # check get_params
    params = model.get_params(deep=False)
    clone = model.__class__(**params)
    for key, val in clone.get_params(deep=False).items():
        assert params[key] is val

    # check deep flag
    deep_params = model.get_params(deep=True)
    assert deep_params["impl"] is not impl

    # check set_params
    clone = model.__class__()
    for key, val in params.items():
        if np.isscalar(val) and not isinstance(val, str):
            params[key] = val + np.random.random()
    # set_params returns itself
    assert clone.set_params(**params) is clone
    for key, val in clone.get_params(deep=False).items():
        assert params[key] is val

    # check fit and fitter
    update_backup = model.update
    model.update = Mock(return_value={"loss": np.random.random()})
    n_episodes = 4
    episode_length = 25
    n_batch = 32
    n_steps = 10
    n_steps_per_epoch = 5
    n_epochs = n_steps // n_steps_per_epoch
    data_size = n_episodes * episode_length
    model._batch_size = n_batch
    shape = (data_size,) + observation_shape
    if len(observation_shape) == 3:
        observations = np.random.randint(256, size=shape, dtype=np.uint8)
    else:
        observations = np.random.random(shape).astype("f4")
    if model.get_action_type() == ActionSpace.CONTINUOUS:
        actions = np.random.random((data_size, action_size))
    else:
        actions = np.random.randint(action_size, size=data_size)
    rewards = np.random.random(data_size)
    terminals = np.zeros(data_size)
    for i in range(n_episodes):
        terminals[(i + 1) * episode_length - 1] = 1.0
    dataset = MDPDataset(observations, actions, rewards, terminals)

    # check fit
    results = model.fit(
        dataset.episodes,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        logdir="test_data",
        verbose=False,
        show_progress=False,
    )

    assert isinstance(results, list)
    assert len(results) == n_epochs

    # check if the correct number of iterations are performed
    assert len(model.update.call_args_list) == n_steps

    # check arguments at each iteration
    for i, call in enumerate(model.update.call_args_list):
        epoch = i // n_steps_per_epoch
        assert isinstance(call[0][0], TransitionMiniBatch)
        assert len(call[0][0]) == n_batch

    # check fitter
    fitter = model.fitter(
        dataset.episodes,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        logdir="test_data",
        verbose=False,
        show_progress=False,
    )

    for epoch, metrics in fitter:
        assert isinstance(epoch, int)
        assert isinstance(metrics, dict)

    assert epoch == n_epochs

    # save params.json
    logger = D3RLPyLogger("test", root_dir="test_data", verbose=False)
    # save parameters to test_data/test/params.json
    model.save_params(logger)
    # load params.json
    json_path = os.path.join(logger.logdir, "params.json")
    new_model = model.__class__.from_json(json_path)
    assert new_model.impl is not None
    assert new_model.impl.observation_shape == observation_shape
    assert new_model.impl.action_size == action_size
    assert type(model.scaler) == type(new_model.scaler)

    # check __setattr__ override
    prev_batch_size = model.impl.batch_size
    model.batch_size = prev_batch_size + 1
    assert model.impl.batch_size == model.batch_size

    # check builds
    model._impl = None
    model.build_with_dataset(dataset)
    assert model.impl.observation_shape == dataset.get_observation_shape()
    assert model.impl.action_size == dataset.get_action_size()

    # set backed up methods
    model._impl = None
    model.update = update_backup

    return dataset


def base_update_tester(model, observation_shape, action_size, discrete=False):
    # make mini-batch
    transitions = []
    prev_transition = None
    for i in range(model.batch_size):
        if len(observation_shape) == 3:
            observation = np.random.randint(
                256, size=observation_shape, dtype=np.uint8
            )
            next_observation = np.random.randint(
                256, size=observation_shape, dtype=np.uint8
            )
        else:
            observation = np.random.random(observation_shape).astype("f4")
            next_observation = np.random.random(observation_shape).astype("f4")
        reward = np.random.random()
        terminal = np.random.randint(2)
        if discrete:
            action = np.random.randint(action_size)
        else:
            action = np.random.random(action_size).astype("f4")

        transition = Transition(
            observation_shape=observation_shape,
            action_size=action_size,
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            prev_transition=prev_transition,
        )

        # set transition to the next pointer
        if prev_transition:
            prev_transition.next_transition = transition

        prev_transition = transition

        transitions.append(transition)

    batch = TransitionMiniBatch(transitions)

    # build models
    model.create_impl(observation_shape, action_size)

    # check if update runs without errors
    grad_step = model.grad_step
    loss = model.update(batch)
    assert model.grad_step == grad_step + 1

    model.set_grad_step(0)
    assert model.grad_step == 0

    assert len(loss.items()) > 0

    return transitions
