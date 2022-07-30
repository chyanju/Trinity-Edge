<div align="left">
  <h1>
    <img src="./resources/trinity-edge-white.png" width=50>
  	Trinity-Edge
  </h1>
</div>

Exploring faster deployments of Trinity for the paper "[Trinity: An Extensible Synthesis Framework for Data Science](https://dl.acm.org/doi/10.14778/3352063.3352098)".

This is a customized/optimized version of the original Trinity ([https://github.com/fredfeng/Trinity](https://github.com/fredfeng/Trinity)), with some additional features that support efficiency and research purpose.

## Prerequisites

- Python 3.8.8 (tested)
- [xxhash](https://github.com/ifduyue/python-xxhash) 2.0.2 (tested)
- [sexpdata](https://github.com/jd-boyd/sexpdata) 0.0.3 (tested)

## Quick Commands

```bash
# testing a benchmark with its target skeleton only
python ./test_morpheus_benchmark.py --benchmark 5 --dsl test_min --skeletons test_min

# testing a benchmark with shared skeleton list
python ./test_morpheus_benchmark.py --benchmark 5 --dsl test_min --skeletons ngram3_nojoin
```

## Usage

##### single benchmark

```bash
usage: test_morpheus_benchmark.py [-h] [-b BENCHMARK] [-d {test_min}] [-s {test_min,test_ks0,ngram3_nojoin}] [--record]

optional arguments:
  -h, --help            show this help message and exit
  -b BENCHMARK, --benchmark BENCHMARK
                        morpheus benchmark id, default: 5
  -d {test_min}, --dsl {test_min}
                        DSL definition to use, default: test_min
  -s {test_min,test_ks0,ngram3_nojoin}, --skeletons {test_min,test_ks0,ngram3_nojoin}
                        skeleton list to use, default: test_min
  --record              whether to enable recorder or not, default: False
```

## Design Notes

- For a list of columns in the parameter, sort it **<u>*ASC*</u>**, and apply the semantics (negative: exclude, positive: include) according to the order.

## TODOs / Known Issues

- `spread`/`group` component are much slower than others.
- Test all morpheus benchmarks to make sure they can actually be solved.
- Sketch level CDCL is not yet ready.
- Automatically synthesize the abstract interpreter based on the provided language specs.
- To speed up parameter enumeration and improve efficacy, consider letting newly generated columns stay in front (not at the end).

## Citation

If you find our work and this tool useful in your research, please consider citing:

```
@article{vldb19_trinity,
  author    = {Ruben Martins and
               Jia Chen and
               Yanju Chen and
               Yu Feng and
               Isil Dillig},
  title     = {Trinity: An Extensible Synthesis Framework for Data Science},
  journal   = {{PVLDB}},
  volume    = {12},
  number    = {12},
  pages     = {1914--1917},
  year      = {2019},
  url       = {http://www.vldb.org/pvldb/vol12/p1914-martins.pdf},
  timestamp = {Fri, 30 Aug 2019 13:15:08 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/pvldb/MartinsCCFD19},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```