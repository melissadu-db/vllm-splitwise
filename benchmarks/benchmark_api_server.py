"""
Start a proxy and (potential a lot of, if data parallel is enabled) API servers.
The proxy acts as a load balancer which uses round-robin to distribute the requests to the API servers.
"""
import os, sys
import subprocess
import argparse
import multiprocessing

from benchmark_utils import BACKEND_TO_PORTS

os.environ["HF_TOKEN"] = "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"

MODEL_TO_PARALLEL_PARAMS = {
    "facebook/opt-125m": {
        "vllm": 1,
        "distserve": (1, 1, 1, 1)
    },
    "facebook/opt-1.3b": {
        "vllm": 1,
        "distserve": (1, 1, 1, 1)
    },
    "facebook/opt-6.7b": {
        "vllm": 1,
        "distserve": (1, 1, 1, 1)   # (context_tp, context_pp, decoding_tp, decoding_pp)
    },
    "facebook/opt-13b": {
        "vllm": 1,
        "distserve": (2, 1, 1, 1)   # TODO adjust me
    },
    "facebook/opt-66b": {
        "vllm": 4,
        "distserve": (4, 1, 2, 2)
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "vllm": 1,
        "distserve": (4, 1, 2, 2)
    },
    "facebook/opt-175b": {
        "vllm": 8,
        "distserve": (3, 3, 4, 3)
    },
}

def api_server_starter_routine(
    port: int,
    args: argparse.Namespace
):
    """
    Start the target API server on the target port
    """
    use_dummy_weight = os.environ.get("USE_DUMMY_WEIGHT", "0") in ["1", "true", "True"]
    if args.backend == "vllm":
        tp_world_size = MODEL_TO_PARALLEL_PARAMS[args.model]["vllm"]
        script = f"""python -u -m vllm.entrypoints.api_server \\
    --model meta-llama/Meta-Llama-3-8B-Instruct --host 0.0.0.0 --port {port} --tensor-parallel-size 1 \\
    --engine-use-ray --worker-use-ray --disable-log-requests \\
    --model {args.model} --dtype half \\
    {"--load-format dummy" if use_dummy_weight else ""} \\
    -tp {tp_world_size} \\
    --block-size 16 --seed 0 \\
    --swap-space 16 \\
    --gpu-memory-utilization 0.95 \\
    --max-num-batched-tokens 16384 \\
    --max-num-seqs 1024 \\
        """

    elif args.backend == "distserve":
        context_tp, context_pp, decoding_tp, decoding_pp = MODEL_TO_PARALLEL_PARAMS[args.model]["distserve"]
        script = f"""
conda activate distserve;
python -m distserve.api_server.distserve_api_server \\
    --host 0.0.0.0 \\
    --port {port} \\
    --model {args.model} \\
    --tokenizer {args.model} \\
    {"--use-dummy-weights" if use_dummy_weight else ""} \\
    \\
    --context-tensor-parallel-size {context_tp} \\
    --context-pipeline-parallel-size {context_pp} \\
    --decoding-tensor-parallel-size {decoding_tp} \\
    --decoding-pipeline-parallel-size {decoding_pp} \\
    \\
    --block-size 16 \\
    --max-num-blocks-per-req 128 \\
    --gpu-memory-utilization 0.95 \\
    --swap-space 16 \\
    \\
    --context-sched-policy fcfs \\
    --context-max-batch-size 128 \\
    --context-max-tokens-per-batch 8192 \\
    \\
    --decoding-sched-policy fcfs \\
    --decoding-max-batch-size 1024 \\
    --decoding-max-tokens-per-batch 65536
"""
    
    print(f"Starting server with command {script}")
    subprocess.run(["bash", "-c", script])


def metadata_server_process(port, args: argparse.Namespace):
    """
    Start a small HTTP server, which returns the metadata of the API servers
    as JSON
    """
    import json
    import http.server
    import socketserver
    
    class MetadataServer(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(args.__dict__).encode())
        def log_message(self, format, *args):
            pass
    
    with socketserver.TCPServer(("", port), MetadataServer) as httpd:
        print("The metadata server is serving at port", port)
        httpd.serve_forever()
    
    
def main(args: argparse.Namespace):
    print(args)
    port = BACKEND_TO_PORTS[args.backend]
    process = multiprocessing.Process(
        target=metadata_server_process,
        args=(port+1, args,)
    )
    process.start()
    api_server_starter_routine(
        port,
        args
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help="The serving backend",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model to be served"
    )
    
    args = parser.parse_args()
    main(args)
    