import numpy as np, sys, os, random, pdb, json, uuid, time, argparse
from pprint import pprint
import logging, logging.config
from collections import defaultdict as ddict
from ordered_set import OrderedSet

# PyTorch related imports
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader
from torch.nn import Parameter
from torch_scatter import scatter_add


np.set_printoptions(precision=4)

def set_gpu(gpus):
	"""
	Sets the GPU to be used for the run

	Parameters
	----------
	gpus:           List of GPUs to be used for the run

	Returns
	-------

	"""
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def get_logger(name):
    """
    创建一个只将日志输出到控制台的 logger 对象。

    Parameters
    ----------
    name: str
        Logger 的名称。

    Returns
    -------
    logging.Logger
        一个将日志输出到标准输出 (stdout) 的 logger 对象。

    """
    # 获取指定名称的 logger 实例
    logger = logging.getLogger(name)

    # 设置 logger 的最低日志级别，避免被全局配置覆盖
    # INFO 级别意味着 INFO, WARNING, ERROR, CRITICAL 级别的日志都会被处理
    logger.setLevel(logging.INFO)

    # 检查 logger 是否已经有关联的 handlers，防止重复添加
    if not logger.handlers:
        # 创建一个流处理器 (StreamHandler)，用于将日志输出到标准输出
        console_handler = logging.StreamHandler(sys.stdout)

        # 定义日志输出格式
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')

        # 为处理器设置格式
        console_handler.setFormatter(formatter)

        # 将处理器添加到 logger
        logger.addHandler(console_handler)

    return logger

def get_combined_results(left_results, right_results):
	results = {}
	count   = float(left_results['count'])

	results['left_mr']	= round(left_results ['mr'] /count, 5)
	results['left_mrr']	= round(left_results ['mrr']/count, 5)
	results['right_mr']	= round(right_results['mr'] /count, 5)
	results['right_mrr']	= round(right_results['mrr']/count, 5)
	results['mr']		= round((left_results['mr']  + right_results['mr']) /(2*count), 5)
	results['mrr']		= round((left_results['mrr'] + right_results['mrr'])/(2*count), 5)

	for k in range(10):
		results['left_hits@{}'.format(k+1)]	= round(left_results ['hits@{}'.format(k+1)]/count, 5)
		results['right_hits@{}'.format(k+1)]	= round(right_results['hits@{}'.format(k+1)]/count, 5)
		results['hits@{}'.format(k+1)]		= round((left_results['hits@{}'.format(k+1)] + right_results['hits@{}'.format(k+1)])/(2*count), 5)
	return results

def get_param(shape):
	param = Parameter(torch.Tensor(*shape));
	xavier_normal_(param.data)
	return param

def com_mult(a, b):
	r1, i1 = a[..., 0], a[..., 1]
	r2, i2 = b[..., 0], b[..., 1]
	return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)

def conj(a):
	a[..., 1] = -a[..., 1]
	return a

def cconv(a, b):
	return torch.fft.irfftn(torch.fft.rfftn(a, (-1)) * torch.fft.rfftn(b, (-1)), (-1))

def ccorr(a, b):
	# return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))
	return torch.fft.irfftn(torch.conj(torch.fft.rfftn(a, (-1))) * torch.fft.rfftn(b, (-1)), (-1))
