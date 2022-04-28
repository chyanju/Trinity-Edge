import pandas as pd
import numpy as np
import argparse
import os
import json
import time
import pickle

import trinity.spec as S
from trinity.enumerator import LineSkeletonEnumerator
from trinity.interpreter import MorpheusInterpreter, MorpheusCoarseAbstractInterpreter, MorpheusPartialInterpreter, MorpheusFineAbstractInterpreter
from trinity.decider import Example, MorpheusDecider
from trinity.synthesizer import MorpheusSynthesizer
from trinity.logger import get_logger
from trinity.recorder import get_recorder, set_recorder_dest
from trinity.profiler import get_profiler_pool

logger = get_logger('trinity')
set_recorder_dest(os.path.abspath("./logs"))

def load_example(arg_id):
    tmp_folder = "./benchmarks/morpheus/{}/".format(arg_id)
    # load example.json
    with open("{}/example.json".format(tmp_folder), "r") as f:
        tmp_json = json.load(f)
    # load input0
    tmp_input0 = pd.DataFrame.from_records(tmp_json["input0"]["data"], columns=tmp_json["input0"]["columns"])
    # load input1
    if tmp_json["input1"] is None:
        tmp_input1 = None
    else:
        tmp_input1 = pd.DataFrame.from_records(tmp_json["input1"]["data"], columns=tmp_json["input1"]["columns"])
    # load ouptut
    tmp_output = pd.DataFrame.from_records(tmp_json["output"]["data"], columns=tmp_json["output"]["columns"])
    # construct Example
    tmp_inputs = [tmp_input0] if tmp_input1 is None else [tmp_input0, tmp_input1]
    return Example( input=tmp_inputs, output=tmp_output )

if __name__ == "__main__":
    logger.setLevel('DEBUG')

    args = {
        "dsl": "test_min",
        "benchmark": 5,
        "skeletons": "test_min",
        "record": False,
    }

    # load dsl spec
    spec = None
    if args["dsl"].startswith("test"):
        spec = S.parse_file("./benchmarks/morpheus/{}/{}.dsl".format(args["benchmark"], args["dsl"]))
    else:
        raise Exception("You should not reach here.")

    # load benchmark
    print("# loading benchmark...")
    benchmark_example = load_example(args["benchmark"])

    print("# input0 type: {}".format(benchmark_example.input[0].dtypes.tolist()))
    print("# input0 is: {}".format(benchmark_example.input[0]))
    print("# output type: {}".format(benchmark_example.output.dtypes.tolist()))
    print("# output is: {}".format(benchmark_example.output))


    # wrap priority pools
    # select(separate(gather(@param0, ['-99', '-1']), 2), ['-3'])
    priority_pools = [
        # only 1 skeleton, order list according to depth-first post-order order

        # for: select(separate(gather(@param0, hole0), hole1), hole2)
        [
            # arguments for gather, hole0
            [
                spec.get_enum_production(spec.get_type("ColList"),['-99', '-2']),
                spec.get_enum_production(spec.get_type("ColList"),['-99', '-1'])
            ],
            # arguments for separate, hole1
            [
                spec.get_enum_production(spec.get_type("ColInt"),"1"),
                spec.get_enum_production(spec.get_type("ColInt"),"3"),
                spec.get_enum_production(spec.get_type("ColInt"),"4"),
                spec.get_enum_production(spec.get_type("ColInt"),"2"),
            ],
            # arguments for select, hole2
            [
                spec.get_enum_production(spec.get_type("ColList"),['-3'])
            ]
        ]
    ]


    # load skeleton
    print("# loading skeleton list...")
    skeleton_list = None
    if args["skeletons"].startswith("test"):
        with open("./benchmarks/morpheus/{}/{}_skeletons.json".format(args["benchmark"], args["skeletons"]), "r") as f:
            skeleton_list = json.load(f)
    else:
        with open("./skeletons/{}_skeletons.json".format(args["skeletons"]), "r") as f:
            skeleton_list = json.load(f)

    enumerator = LineSkeletonEnumerator( spec=spec, cands=skeleton_list, pools=priority_pools )
    interpreter = MorpheusInterpreter( spec=spec )
    # coarse: perform skeleton level abstract interpretation, throws SkeletonAssertion
    coarse_abstract_interpreter = MorpheusCoarseAbstractInterpreter()
    # fine: perform finer level (sub-skeleton level) abstract interpretation throws EnumAssertion
    fine_abstract_interpreter = MorpheusFineAbstractInterpreter( spec=spec )
    partial_interpreter = MorpheusPartialInterpreter(
        interpreter=interpreter,
        # partial interpreter currently only works with coarse abstract interpreter
        # abstract_interpreter=fine_abstract_interpreter,
        abstract_interpreter=coarse_abstract_interpreter,
    )

    # print some basic info
    # print("# search space (skeletons): {}".format(len(enumerator._iters)))
    # print("# search space (concrete programs): {}".format(
    #     sum([len(p) for p in enumerator._iters])
    # ))
    # input("press to start")
    recorder = None
    if args["record"]:
        recorder = get_recorder("test.b={}.d={}.s={}".format(
            args["benchmark"], args["dsl"], args["skeletons"]
        ))

    decider = MorpheusDecider( 
        interpreter=interpreter, 
        coarse_abstract_interpreter=coarse_abstract_interpreter,
        # fine_abstract_interpreter=fine_abstract_interpreter,
        partial_interpreter=partial_interpreter,
        examples=[benchmark_example], 
        equal_output=interpreter.equal_tb,
        recorder=recorder,
    )
    synthesizer = MorpheusSynthesizer(
        enumerator=enumerator,
        decider=decider
    )

    program = synthesizer.synthesize()

    ppool = get_profiler_pool()
    for p in ppool.keys():
        print(ppool[p])
        if recorder:
            recorder.record(("Profiler", str(ppool[p])))

    print("# search space (skeletons): {}".format(len(enumerator._iters[:enumerator._iter_ptr+1])))
    print("# search space (concrete programs): {}".format(
        sum([len(p) for p in enumerator._iters[:enumerator._iter_ptr+1]])
    ))

    if program is None:
        logger.info('Exhausted.')
    else:
        logger.info("Accepted: {}.".format(program))
        print("{}".format(str(program.to_jsonexp()).replace("'",'"')))


