import argparse
import glob
import json
import os
import shutil
import warnings
from datetime import datetime, timedelta
from multiprocessing import Pool
from os.path import abspath, basename, dirname, exists, join

# turn off all warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from RCAEval.benchmark.evaluation import Evaluator
from RCAEval.classes.graph import Node

from RCAEval.io.time_series import drop_constant, drop_time, preprocess
from RCAEval.utility import (
    dump_json,
    is_py38,
    is_py312,
    load_json,
    download_online_boutique_dataset,
    prepare_llm_ref_stack_dataset,
    download_sock_shop_1_dataset,
    download_sock_shop_2_dataset,
    download_train_ticket_dataset,
    download_re1_dataset,
    download_re2_dataset,
    download_re3_dataset, 
)


if is_py312():
    from RCAEval.e2e import (
        baro,
        mmbaro,
        causalrca,
        circa,
        cloudranger,
        cmlp_pagerank,
        dummy,
        e_diagnosis,
        easyrca,
        fci_pagerank,
        fci_randomwalk,
        ges_pagerank,
        granger_pagerank,
        granger_randomwalk,
        lingam_pagerank,
        lingam_randomwalk,
        micro_diag,
        microcause,
        microrank,
        mscred,
        nsigma,
        ntlr_pagerank,
        ntlr_randomwalk,
        pc_pagerank,
        pc_randomwalk,
        run,
        tracerca,
        causalai,
        microrca,
        microscope,
        monitorrank,
        pdiagnose,
    )

elif is_py38():
    from RCAEval.e2e import dummy, e_diagnosis, ht, rcd, mmrcd
else:
    print("Please use Python 3.8 or 3.12")
    exit(1)

try:
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from RCAEval.e2e.causalrca import causalrca
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="RCAEval evaluation")
    parser.add_argument("--method", type=str, help="Choose a method.")
    parser.add_argument("--dataset", type=str, help="Choose a dataset.", choices=[
        "online-boutique", "sock-shop-1", "sock-shop-2", "train-ticket",
        "re1-ob", "re1-ss", "re1-tt", "re2-ob", "re2-ss", "re2-tt", "re3-ob", "re3-ss", "re3-tt"
    ])
    parser.add_argument("--length", type=int, default=20, help="Time series length (RQ4)")
    parser.add_argument("--tdelta", type=int, default=0, help="Specify $t_delta$ to simulate delay in anomaly detection")
    parser.add_argument("--test", action="store_true", help="Perform smoke test on certain methods without fully run on all data")
    args = parser.parse_args()

    if args.method not in globals():
        raise ValueError(f"{args.method=} not defined. Please check imported methods.")

    return args

def prepare_data(args):
    # download dataset
    if "llm-ref-stack" in args.dataset:
        prepare_llm_ref_stack_dataset(args.dataset_root, args.dataset)
    else:
        raise Exception(f"{args.dataset} is not defined!")

    dataset = f"data/{args.dataset}"

    # prepare input paths
    data_paths = list(glob.glob(os.path.join(dataset, "**/data.csv"), recursive=True))
    if not data_paths: 
        data_paths = list(glob.glob(os.path.join(dataset, "**/simple_metrics.csv"), recursive=True))
    # new_data_paths = []
    # for p in data_paths: 
    #     if os.path.exists(p.replace("data.csv", "simple_data.csv")):
    #         new_data_paths.append(p.replace("data.csv", "simple_data.csv"))
    #     elif os.path.exists(p.replace("data.csv", "simple_metrics.csv")):
    #         new_data_paths.append(p.replace("data.csv", "simple_metrics.csv"))
    #     else:
    #         new_data_paths.append(p)
    # data_paths = new_data_paths
    if args.test is True:
        data_paths = data_paths[:2]

    # prepare output paths
    from tempfile import TemporaryDirectory
    # output_path = TemporaryDirectory().name
    output_path = "output"
    report_path = join(output_path, args.dataset, args.method, f"report.csv")
    result_path = join(output_path, args.dataset, args.method, "results")
    os.makedirs(result_path, exist_ok=True)
    return data_paths, result_path, report_path


def process(data_path, args, result_path):
    run_args = argparse.Namespace()
    run_args.root_path = os.getcwd()
    run_args.data_path = data_path
    
    # convert length from minutes to seconds
    if args.length is None:
        args.length = 10
    data_length = args.length * 60 // 2

    data_dir = dirname(data_path)

    service, metric = basename(dirname(dirname(data_path))).split("_")
    case = basename(dirname(data_path))

    rp = join(result_path, f"{service}_{metric}_{case}.json")

    # == Load and Preprocess data ==
    data = pd.read_csv(data_path)
    
    # remove lat-50, only selecte lat-90 
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    
    # handle inf
    data = data.replace([np.inf, -np.inf], np.nan)
    # handle na
    data = data.fillna(method="ffill")
    data = data.fillna(0)

    with open(join(data_dir, "inject_time.txt")) as f:
        inject_time = int(f.readlines()[0].strip()) + args.tdelta
    # for metrics, minutes -> seconds // 2
    normal_df = data[data["time"] < inject_time].tail(args.length * 60 // 2)
    anomal_df = data[data["time"] >= inject_time].head(args.length * 60 // 2)

    data = pd.concat([normal_df, anomal_df], ignore_index=True)

    # num column, exclude time
    num_node = len(data.columns) - 1

    # rename latency
    data = data.rename(
        columns={
            c: c.replace("_latency-90", "_latency")
            for c in data.columns
            if c.endswith("_latency-90")
        }
    )
    
    # == Get SLI ===
    sli = None
    if "llm-ref-stack" in data_path:
        data = data.drop(columns=[col for col in data.columns if "_pod_" in col])
        data = data.drop(columns=[col for col in data.columns if "_node_" in col])
        data = data.drop(columns=[col for col in data.columns if "unknown" in col])
        sli = "nginx-proxy"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
        if f"{service}_lat_90" in data:
            sli = f"{service}_lat_90"

    else:
        raise ValueError("SLI not implemented")

    # == PROCESS ==
    func = globals()[args.method]

    try:
        st = datetime.now()
        
        out = func(
            data,
            inject_time,
            dataset=args.dataset,
            anomalies=None,
            dk_select_useful=False,
            sli=sli,
            verbose=False,
            n_iter=num_node,
            args=run_args,
        )
        root_causes = out.get("ranks")
        # print("==============")
        # print(f"{data_path=}")
        # print(root_causes[:5])
        dump_json(filename=rp, data={0: root_causes})
    except Exception as e:
        raise e
        print(f"{args.method=} failed on {data_path=}")
        print(e)
        rp = join(result_path, f"{service}_{metric}_{case}_failed.json")
        with open(rp, "w") as f:
            json.dump({"error": str(e)}, f)

def run_evaluation(data_paths, args, result_path, report_path):
    start_time = datetime.now()

    for data_path in tqdm(sorted(data_paths)):
        process(data_path, args, result_path)

    end_time = datetime.now()
    time_taken = end_time - start_time
    avg_speed = round(time_taken.total_seconds() / len(data_paths), 2)


    # ======== EVALUTION ===========
    rps = glob.glob(join(result_path, "*.json"))
    services = sorted(list(set([basename(x).split("_")[0] for x in rps])))
    faults = sorted(list(set([basename(x).split("_")[1] for x in rps])))

    eval_data = {
        "service-fault": [],
        "top_1_service": [],
        "top_3_service": [],
        "avg@5_service": [],
    }

    s_evaluator_all = Evaluator()
    s_evaluator_cpu = Evaluator()
    s_evaluator_mem = Evaluator()
    s_evaluator_net = Evaluator()
    s_evaluator_gpu = Evaluator()

    for service in services:
        for fault in faults:
            s_evaluator = Evaluator()

            for rp in rps:
                s, m = basename(rp).split("_")[:2]
                if s != service or m != fault:
                    continue  # ignore

                data = load_json(rp)
                if "error" in data:
                    continue  # ignore

                for i, ranks in data.items():
                    s_ranks = [Node(x.split("_")[0].replace("-db", ""), "unknown") for x in ranks]
                    # remove duplication
                    old_s_ranks = s_ranks.copy()
                    s_ranks = (
                        [old_s_ranks[0]]
                        + [
                            old_s_ranks[i]
                            for i in range(1, len(old_s_ranks))
                            if old_s_ranks[i] not in old_s_ranks[:i]
                        ]
                        if old_s_ranks
                        else []
                    )

                    s_evaluator.add_case(ranks=s_ranks, answer=Node(service, "unknown"))

                    if fault == "cpu":
                        s_evaluator_cpu.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                        s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))

                    elif fault == "memory":
                        s_evaluator_mem.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                        s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))

                    elif fault == "network":
                        s_evaluator_net.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                        s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))

                    elif fault == "gpu":
                        s_evaluator_gpu.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                        s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))

            eval_data["service-fault"].append(f"{service}_{fault}")
            eval_data["top_1_service"].append(s_evaluator.accuracy(1))
            eval_data["top_3_service"].append(s_evaluator.accuracy(3))
            eval_data["avg@5_service"].append(s_evaluator.average(5))


    print(f"--- Evaluation results for '{args.method}' ---")
    for name, s_evaluator in [
        ("cpu", s_evaluator_cpu),
        ("memory", s_evaluator_mem),
        ("network", s_evaluator_net),
        ("gpu", s_evaluator_gpu),
    ]:
        eval_data["service-fault"].append(f"overall_{name}")
        eval_data["top_1_service"].append(s_evaluator.accuracy(1))
        eval_data["top_3_service"].append(s_evaluator.accuracy(3))
        eval_data["avg@5_service"].append(s_evaluator.average(5))

        if s_evaluator.average(5) is not None:
            print( f"AC@1-{name.upper()}:".ljust(12), round(s_evaluator.accuracy(1), 2))
            print( f"AC@3-{name.upper()}:".ljust(12), round(s_evaluator.accuracy(3), 2))
            print( f"Avg@5-{name.upper()}:".ljust(12), round(s_evaluator.average(5), 2))

    print("---")
    print("Avg speed:", avg_speed)
    pd.DataFrame(eval_data).to_csv(report_path, index=False)

def main(args):
    data_paths, result_path, report_path = prepare_data(args)
    if not (any(os.path.isfile(os.path.join(result_path, f)) for f in os.listdir(result_path))):
        run_evaluation(data_paths, args, result_path, report_path)

if __name__ == "__main__":
    main(parse_args())

