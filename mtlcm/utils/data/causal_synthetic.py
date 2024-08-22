from typing import Union
import numpy as np
import torch


def gen_data(
    num_tasks,
    num_support_points=16,
    num_query_points=128,
    num_features=10,
    sigma_range_support=None,
    sigma_range_query=None,
    num_causal=None,
    deterministic_causal=False,
    target_noise="sigmas",
    flip_spurious=False,
    fix_causal_ids=False,
    sigma_support=None,
    sigma_query=None,
    spurious_noise=1,
    causal_noise=None,
    sample_gammas=True,
    sample_weights=True,
    standardize_features=False,
):
    """
    Generate the data for one batch for the synthetic experiment on causal variables.

    Args:
        fix_causal_ids (bool): Flag for debugging. Fixes the Causal IDS to be identical everywhere.
        fix_causal_ids_front (bool): Flag for debugging. Fixes the Causal IDS to be the first num_causal vars (True)
        or last num_causal vars(False). This flags is used when fix_causal_ids=True.
        num_causal (int): If specified, fixes the number of causal variables present per task.
        num_tasks (B): Number of tasks per batch.
        num_support_points (S): Number of points in support set (per task)
        num_query_points (Q): Number of points in query set (per task)
        num_features (F): Synthetic feature dimension to be generated.
        sigma_range_support: Range for the uniform distribution used to sample the "environment" variables. These variables are
        used to establish the noise scale parameters for each environment.
        causal_noise: If specified, overrides the noise scale generated per environment for the causal variables.
        sample_gammas: If True, samples the gamma coefficients for each task from a uniform distribution. Otherwise, gamma is fixed to 1.

    Returns:
        x_s: Support set features (B x S x F)
        y_s: Support set targets (B x S x 1)
        x_q: Query set features (B x Q x F)
        y_q: Query set targets (B x Q x 1)
        causal_indicator: Binary indicator variables for the causal variables in each task (B x F)

    """

    (
        causal_indicator,
        gamma_coeffs,
        sample_weights,
        sigma_range_query,
        sigma_range_support,
    ) = _gen_invariant_params(
        fix_causal_ids,
        num_causal,
        num_features,
        num_tasks,
        sample_gammas,
        sample_weights,
        sigma_range_query,
        sigma_range_support,
    )

    # support set
    x_s, y_s, flip_spurious_s, _ = _gen_data_split(
        num_tasks,
        causal_indicator,
        deterministic_causal,
        num_features,
        num_support_points,
        sigma_range=sigma_range_support,
        flip_spurious=(
            False if isinstance(flip_spurious, bool) else flip_spurious
        ),  # Only flip on query set if bool
        fixed_sigma=sigma_support,
        target_noise=target_noise,
        spurious_noise=spurious_noise,
        causal_noise=causal_noise,
        gamma_coeffs=gamma_coeffs,
        sample_weights=sample_weights,
        standardize_features=standardize_features,
    )

    # query set
    x_q, y_q, flip_spurious_q, _ = _gen_data_split(
        num_tasks,
        causal_indicator,
        deterministic_causal,
        num_features,
        num_query_points,
        sigma_range=sigma_range_query,
        fixed_sigma=sigma_query,
        flip_spurious=(
            False
            if (not flip_spurious and flip_spurious is not None)
            else (1 - flip_spurious_s)
        ),
        target_noise=target_noise,
        spurious_noise=spurious_noise,
        causal_noise=causal_noise,
        gamma_coeffs=gamma_coeffs,
        sample_weights=sample_weights,
        standardize_features=standardize_features,
    )

    if flip_spurious is None:
        assert (flip_spurious_q + (flip_spurious_s * 1) == 1).all()
    if flip_spurious:
        assert (flip_spurious_q + flip_spurious_s) == 1

    x_s, y_s, x_q, y_q = (
        torch.from_numpy(x_s).float(),
        torch.from_numpy(y_s).float(),
        torch.from_numpy(x_q).float(),
        torch.from_numpy(y_q).float(),
    )

    if sample_gammas:
        # Reshape gamma to include zeros for the causal positions
        spurious_idx = np.where(causal_indicator == 0)
        gammas = np.zeros_like(causal_indicator)
        gammas[spurious_idx[0], spurious_idx[1]] = gamma_coeffs.flatten()
        gammas = torch.from_numpy(gammas).float()

        return x_s, y_s, x_q, y_q, causal_indicator, gammas, sample_weights

    return (
        x_s,
        y_s,
        x_q,
        y_q,
        causal_indicator,
        torch.ones_like(torch.from_numpy(causal_indicator)).float(),
        sample_weights,
    )


def _gen_invariant_params(
    fix_causal_ids,
    num_causal,
    num_features,
    num_tasks,
    sample_gammas,
    sample_weights,
    sigma_range_query=None,
    sigma_range_support=None,
):
    if sigma_range_support is None:
        sigma_range_support = [0, 1]
    if sigma_range_query is None:
        sigma_range_query = [0, 1]
    if num_causal is None:
        causal_indicator = np.random.randint(
            low=0, high=2, size=[num_tasks, num_features]
        )
    else:
        causal_indicator = np.zeros((num_tasks, num_features))
        for i in range(num_tasks):
            if fix_causal_ids:
                causal_indicator[
                    i,
                    [
                        j
                        for j in range(
                            num_features - 1, num_features - num_causal - 1, -1
                        )
                    ],
                ] = 1
            else:
                # Sample a fixed number of causal indicators for each task
                indices = np.random.choice(
                    a=num_features, size=num_causal, replace=False
                )
                causal_indicator[i, indices] = 1
    if sample_gammas:
        gamma_coeffs = np.random.uniform(
            low=-1, high=1, size=(num_tasks, num_features - num_causal)
        )
    else:
        gamma_coeffs = None
    if sample_weights:
        sample_weights = np.random.randn(num_tasks, 1, num_causal)
    else:
        sample_weights = np.ones((num_tasks, 1, num_causal))
    return (
        causal_indicator,
        gamma_coeffs,
        sample_weights,
        sigma_range_query,
        sigma_range_support,
    )


def _gen_data_split(
    num_tasks,
    causal_indicator,
    deterministic_causal,
    num_features,
    num_data_points,
    sigma_range,
    flip_spurious=False,
    fixed_sigma=None,
    target_noise="sigmas",
    spurious_noise=1,
    causal_noise=None,
    gamma_coeffs=None,
    sample_weights: Union[bool, np.ndarray] = False,
    standardize_features=False,
):
    """
    Generates the data for a set of tasks. Each task is generated by sampling a set of causal variables from a uniform distribution, the target is generated as the sum of the causal variables,
    and the spurious variables are generated by a weighting of the targets. The noise on the causal, spurious, and target variables are specified by the user.

    """

    if gamma_coeffs is not None:
        assert gamma_coeffs.shape[0] == num_tasks
        assert (
            gamma_coeffs.shape[1] == num_features - np.sum(causal_indicator, axis=1)[0]
        )

    # Check that all tasks have same number of causal variables
    assert np.all(
        np.sum(causal_indicator, axis=1) == np.sum(causal_indicator, axis=1)[0]
    )

    # create an empty feature matrix and an empty target vector
    x = np.zeros(shape=[num_tasks, num_data_points, num_features])
    y = np.zeros(shape=[num_tasks, num_data_points, 1])
    sigma_vals = np.zeros(shape=[num_tasks, 1])
    if flip_spurious is None:
        flip_spurious = 1 * (np.random.uniform(size=num_tasks) > 0.5)
    # generate x and y for each task
    for batch in range(num_tasks):
        # collect indices for causal/spurious features

        causal_idx = []
        spurious_idx = []
        for i in range(num_features):
            if causal_indicator[batch, i] == 1:
                causal_idx.append(i)
            else:
                spurious_idx.append(i)

        # sample sigma (the environment variable E1)
        if fixed_sigma is None:
            task_sigma = np.random.uniform(low=sigma_range[0], high=sigma_range[1])
        else:
            task_sigma = fixed_sigma
        sigma_vals[batch] = task_sigma

        # generate causal features by x_i = N(0, sigma^2)
        if causal_noise is None:
            x_causal = np.random.normal(
                loc=0.0, scale=task_sigma, size=[num_data_points, len(causal_idx)]
            )
        else:
            x_causal = np.random.normal(
                loc=0.0, scale=causal_noise, size=[num_data_points, len(causal_idx)]
            )

        for i, idx in enumerate(causal_idx):
            if deterministic_causal:
                x[batch, :, idx] = 1
            else:
                x[batch, :, idx] = x_causal[:, i].copy()

        # generate target by y = \sum_{causal x_i} x_i + N(0, sigma^2)
        if target_noise == "deterministic":
            y_noise = 0
        elif target_noise == "standard_normal":
            y_noise = np.random.normal(loc=0.0, scale=1, size=[num_data_points, 1])
        elif target_noise == "sigmas":
            y_noise = np.random.normal(
                loc=0.0, scale=task_sigma, size=[num_data_points, 1]
            )
        elif isinstance(target_noise, float):
            y_noise = np.random.normal(
                loc=0.0, scale=target_noise, size=[num_data_points, 1]
            )
        else:
            raise ValueError("Target noise argument nor recognized")

        if isinstance(sample_weights, bool) and sample_weights:
            causal_weights = np.random.randn(1, len(causal_idx))
            causal_sum = np.sum(
                np.transpose(x[batch, :, causal_idx]) * causal_weights,
                axis=-1,
                keepdims=True,
            )
        elif isinstance(sample_weights, (list, np.ndarray)):
            causal_weights = sample_weights[batch]
            causal_sum = np.sum(
                np.transpose(x[batch, :, causal_idx]) * causal_weights,
                axis=-1,
                keepdims=True,
            )
        else:
            causal_sum = np.transpose(x[batch, :, causal_idx]).sum(
                axis=-1, keepdims=True
            )
        y[batch, :, :] = causal_sum + y_noise

        # generate  spurious features by x_i = y + N(0, 1)
        if isinstance(flip_spurious, (bool, int)):
            flip = -1 if flip_spurious else 1
        else:  # array
            flip = -1 if flip_spurious[batch] else 1

        x_spurious = flip * y[batch, :, :].copy()

        if gamma_coeffs is not None:
            x_spurious = x_spurious * gamma_coeffs[batch]

        x_spurious = x_spurious + np.random.normal(
            loc=0, scale=spurious_noise, size=[num_data_points, len(spurious_idx)]
        )

        for i, idx in enumerate(spurious_idx):
            x[batch, :, idx] = x_spurious[:, i].copy()

    if standardize_features:
        for task in range(num_tasks):
            x[task] = (x[task] - x[task].mean(axis=0)) / x[task].std(axis=0)

    return x, y, flip_spurious, sigma_vals
