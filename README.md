# AMI

Implementations of the paper [Adversarial Mutual Information for Text Generation](https://proceedings.icml.cc/paper/2020/file/85ea6fd7a2ca3960d0cf5201933ac998-Paper.pdf). The code is based on the [OpenNMT](https://github.com/OpenNMT/OpenNMT-py).

## Required

All dependencies can be installed by:

```bash
pip install -r requirements.opt.txt
```

## Data

We support dialogue generation and neural machine translation datasets, and the datasets we used in the paper are [PersonaChat](https://github.com/facebookresearch/
ParlAI/tree/master/projects/personachat) and [WMT translation dataset](http://www.statmt.org/wmt14/
translation-task.html).

## Preprocessing

See `build_dataset.sh` for preprocessing commands.


## Run

See `train.sh` and `test.sh` for training and testing commands.

If you want to train on GPU, you need to set, as an example: `CUDA_VISIBLE_DEVICES=1,3 -world_size 2 -gpu_ranks 0 1` to use (say) GPU 1 and 3 on this node only. 

For any question or suggestions, feel free to contact panby@zju.edu.cn.


## Reference

**"Adversarial Mutual Information for Text Generation"**
Boyuan Pan*, Yazheng Yang*, Kaizhao Liang, Bhavya Kailkhura, Zhongming Jin, Xian-sheng Hua, Deng Cai, Bo Li. _ICML (2020)_ 

```
@article{Pan2020AdversarialMI,
  title={Adversarial Mutual Information for Text Generation},
  author={Boyuan Pan and Y. Yang and Kaizhao Liang and B. Kailkhura and Zhongming Jin and Xiansheng Hua and Deng Cai and B. Li},
  journal={ArXiv},
  year={2020},
  volume={abs/2007.00067}
}
```
