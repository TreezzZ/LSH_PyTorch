import torch

from utils.evaluate import mean_average_precision


def train(
        query_data,
        query_targets,
        retrieval_data,
        retrieval_targets,
        code_length,
        device,
        topk,
):
    """
    Training model

    Args
        query_data(torch.Tensor): Query data.
        query_targets(torch.Tensor): One-hot query targets.
        retrieval_data(torch.Tensor): Retrieval data.
        retrieval_targets(torch.Tensor): One-hot retrieval targets.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.
        topk(int): Calculate top k data map.

    Returns
        mAP(float): Mean Average Precision.
    """
    query_data, retrieval_data, query_targets, retrieval_targets = query_data.to(device), retrieval_data.to(device), query_targets.to(device), retrieval_targets.to(device)

    # Generate random projection matrix
    W = torch.randn(query_data.shape[1], code_length).to(device)

    # Generate query and retrieval code
    query_code = (query_data @ W).sign()
    retrieval_code = (retrieval_data @ W).sign()

    # Compute map
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        query_targets,
        retrieval_targets,
        device,
        topk,
    )

    return mAP
