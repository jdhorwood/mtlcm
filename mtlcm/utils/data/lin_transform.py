import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from mtlcm.utils.data.causal_synthetic import gen_data
import matplotlib.pyplot as plt


def cal_mcc(z1, z2, pearson=True):
    assert z1.shape == z2.shape
    d = z1.shape[1]
    if pearson:
        cc = np.corrcoef(z1, z2, rowvar=False)[:d, d:]
    else:
        cc = spearmanr(z1, z2)[0][:d, d:]
    abs_cc = np.abs(cc)
    assignments = linear_sum_assignment(-1.0 * abs_cc)
    score = abs_cc[assignments].mean()
    return score, assignments


def gen_observed_data_from_x(x_s, x_q, transformation, noise_std=1e-3):
    o_s = x_s @ transformation.T
    o_s = o_s + torch.randn_like(o_s) * noise_std
    o_q = x_q @ transformation.T
    o_q = o_q + torch.randn_like(o_q) * noise_std

    return o_s, o_q, transformation


def generate_linear_transform(
    dimension, device="cpu", orthogonal=False, identity=False
):
    # generate a random orthogonal matrix as the ground truth transformation if it's not given
    tmp_mat = torch.randn(dimension, dimension).to(device)

    if identity:
        transformation = torch.eye(dimension).to(device)
    elif orthogonal:
        tmp_mat = torch.linalg.svd(tmp_mat)
        transformation = tmp_mat[0] @ tmp_mat[2]
    else:
        transformation = tmp_mat

    return transformation


def generate_training_data(
    num_tasks,
    observation_noise,
    spurious_noise=0.1,
    transformation=None,
    observation_dim=10,
    num_causal=2,
    num_support_points=20,
    sample_gammas=False,
    standardize_features=False,
    device="cpu",
):
    if transformation is None:
        transformation = generate_linear_transform(dimension=observation_dim)

    x_support, y_support, x_query, y_query, causal_indicators, gamma_coeffs, _ = (
        gen_data(
            num_tasks=num_tasks,
            num_support_points=num_support_points,
            num_query_points=128,
            num_features=observation_dim,
            sigma_range_support=[2, 3],
            num_causal=num_causal,
            target_noise="sigmas",
            flip_spurious=False,
            # device=device,
            spurious_noise=spurious_noise,
            causal_noise=1,
            sample_gammas=sample_gammas,
            standardize_features=standardize_features,
        )
    )

    o_support, o_query, transformation = gen_observed_data_from_x(
        x_s=x_support,
        x_q=x_query,
        transformation=transformation,
        noise_std=observation_noise,
    )

    return (
        o_support,
        y_support,
        o_query,
        y_query,
        x_support,
        x_query,
        causal_indicators,
        gamma_coeffs,
    )


def generate_synthetic_transform_data(
    observation_dim,
    sigma_s,
    sigma_obs,
    num_tasks,
    transformation=None,
    identity=False,
    orthogonal=False,
    num_causal=2,
    num_support_points=20,
    sample_gammas=False,
    device=None,
    standardize_features=False,
):
    """
    Entrypoint for generating synthetic data for linear transformation experiments.

    Args:
        observation_dim: Dimension of observed features.
        sigma_s: Standard deviation of spurious noise.
        sigma_obs: Standard deviation of observation noise.
        num_tasks: Number of tasks to generate.
        transformation: Linear transformation matrix.
        identity: Whether to use identity matrix as transformation.
        orthogonal: Whether to use orthogonal matrix as transformation.
        num_causal: Number of causal features.
        num_support_points: Number of support points per task.

    Returns:
        causal_index: Causal index.
        latents: Latent variables.
        o_supportx: Support set of observed features.
        o_supporty: Support set of target.
        transformation: Linear transformation matrix.

    """
    if transformation is None:
        transformation = generate_linear_transform(
            dimension=observation_dim, identity=identity, orthogonal=orthogonal
        )

    # Construct batches
    (
        o_supportx,
        o_supporty,
        _,
        _,
        latents,
        _,
        causal_index,
        gamma_coeffs,
    ) = generate_training_data(
        num_tasks=num_tasks,
        observation_dim=observation_dim,
        transformation=transformation,
        observation_noise=sigma_obs,
        spurious_noise=sigma_s,
        num_causal=num_causal,
        num_support_points=num_support_points,
        sample_gammas=sample_gammas,
        standardize_features=standardize_features,
        device=device,
    )

    # Put everything onto device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    o_supportx = o_supportx.to(device)
    o_supporty = o_supporty.to(device)
    latents = latents.to(device)
    causal_index = torch.as_tensor(causal_index).to(device)
    gamma_coeffs = gamma_coeffs.to(device)
    transformation = transformation.to(device)

    return gamma_coeffs, causal_index, latents, o_supportx, o_supporty, transformation


def cal_weak_strong_mcc(z1, z2, cca_dim=None):
    if cca_dim is None:
        cca_dim = z1.shape[-1]

    cca = CCA(n_components=cca_dim)
    cca.fit(z1, z2)
    res_out = cca.transform(z1, z2)
    z1_cca = res_out[0]
    z2_cca = res_out[1]

    if z1.shape == z2.shape:
        strong = cal_mcc(z1, z2, pearson=False)[0]
    else:
        strong = None

    weak = cal_mcc(z1_cca, z2_cca, pearson=False)[0]
    # weak = cal_mcc(cca.predict(z1), z2, pearson=False)[0]
    return weak, strong
