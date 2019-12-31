import argparse

import torch
from loguru import logger

import lsh
from data.dataloader import load_data


def run():
    # Load configuration
    args = load_config()
    logger.add('logs/{}.log'.format(args.dataset), rotation='500 MB', level='INFO')
    logger.info(args)

    # Load dataset
    _, _, query_data, query_targets, retrieval_data, retrieval_targets = load_data(args.dataset, args.root)

    # Training
    for code_length in args.code_length:
        mAP = lsh.train(
            query_data,
            query_targets,
            retrieval_data,
            retrieval_targets,
            code_length,
            args.device,
            args.topk,
        )
        logger.info('[code length:{}][map:{:.4f}]'.format(code_length, mAP))


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='LSH_PyTorch')
    parser.add_argument('--dataset', type=str,
                        help='Dataset name.')
    parser.add_argument('--root', type=str,
                        help='Path of dataset')
    parser.add_argument('--code-length', default='8,16,24,32,48,64,96,128', type=str,
                        help='Binary hash code length.(default: 8,16,24,32,48,64,96,128)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate top k data map.(default: all)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))

    return args


if __name__ == "__main__":
    run()

