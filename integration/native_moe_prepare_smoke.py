#!/usr/bin/env python3
"""Compare bounded MoE CPU preparation reuse with a tiny local CUDA fixture.

Both controls use the same complete native payload, original weights, workspace
and memory budgets. No download, installation or large-model inference occurs.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import time


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def stable(value):
    if isinstance(value, dict):
        return {key: stable(item) for key,item in value.items() if key not in ('timings','tokens_cached')}
    if isinstance(value, list):
        return [stable(item) for item in value]
    return value


def trace(server):
    rows = []
    for line in server.log_text().splitlines():
        if 'moe_prepare:' in line:
            values = dict((name,int(value)) for name,value in re.findall(r'(\w+)=(\d+)', line))
            require({'builds','reuses','invalidations','key_changes','cpu_metadata','cpu_ids','gpu_workspace','pending'} <= values.keys(),
                    'Incomplete actual MoE preparation counters')
            rows.append(values)
    return rows


def server_for(launcher, args, label, option, expert_major, diagnostic):
    server = launcher.NativeServer(args.executable, args.model, args.output_dir/label, True)
    for flag,value in (('--ctx-size','512'),('--batch-size','128'),('--ubatch-size','128'),
                       ('--cache-ram','0'),('--n-gpu-layers','all'),
                       ('--log-verbosity','5' if diagnostic else '4')):
        server.command[server.command.index(flag)+1] = value
    server.command.extend(('--cpu-moe','--no-host','--moe-ram-limit-mib','8',
                           '--moe-cache-mib','1','--gpu-exclusive','--gpu-vram-limit-mib','1024',
                           '--override-kv','qwen3moe.context_length=int:512'))
    server.env.update(WILDLLAMA_MOE_RUNTIME='1', WILDLLAMA_MOE_DYNAMIC_PREFILL='1',
                      WILDLLAMA_MOE_PREPARE_REUSE=option,
                      WILDLLAMA_MOE_PREPARE_TRACE='1' if diagnostic else '0',
                      WILDLLAMA_MOE_EXPERT_MAJOR=str(expert_major),
                      WILDLLAMA_GRAPH_VARIANTS_MIB='0')
    return server


def body(tokens, probabilities=False):
    return dict(prompt=[40+(i*13)%180 for i in range(tokens)], n_predict=4,
                temperature=0.0, seed=1234, samplers=['temperature'],
                repeat_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0,
                dry_multiplier=0.0, cache_prompt=False, ignore_eos=True,
                return_tokens=True, backend_sampling=True, n_probs=8 if probabilities else 0)


def complete(server, request):
    started = time.perf_counter_ns()
    response = server.request('/completion', request)
    elapsed = (time.perf_counter_ns()-started)/1e6
    require(math.isfinite(elapsed) and elapsed > 0, 'Invalid completion time')
    require(response.get('stop') is True and response.get('tokens_predicted') == request['n_predict'],
            'Incomplete actual provider completion')
    return response,elapsed


def run(args):
    require(__debug__, 'Do not disable validation with Python -O')
    require(args.executable.is_file() and args.model.is_file(), 'Explicit existing payload and fixture required')
    require(0 < args.model.stat().st_size <= 64*1024*1024, 'Only local synthetic fixtures up to 64 MiB')
    require(args.output_dir.drive.upper() != 'C:', 'Outputs must remain outside C:')
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), 'Use a fresh output directory')
    require(0 <= args.cycles <= 12, 'Use at most twelve bounded ABBA cycles')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    launcher = module('moe_prepare_launcher','native_execution_features_smoke.py')
    provenance = module('moe_prepare_provenance','native_payload_provenance.py')
    identity = provenance.manifest(args.executable)
    fixture_hash = sha(args.model)
    report = dict(status='running',model_bytes=args.model.stat().st_size,model_sha256=fixture_hash,
                  harness_sha256=sha(__file__), launcher_sha256=sha(Path(launcher.__file__)),
                  provenance_sha256=sha(Path(provenance.__file__)),payload_identity=identity,
                  same_payload_controls=True, configurations=[],
                  scope='Tiny synthetic MoE provider; no trained-model language-quality or TPS claim')
    raw = []
    try:
        for expert_major in (0,1):
            observed = {}
            for label,option in (('disabled','0'),('enabled','1')):
                server = server_for(launcher,args,f'correctness-{expert_major}-{label}',option,expert_major,True)
                with server:
                    modules = provenance.verify_loaded_payload(server,identity)
                    outputs = []
                    for tokens in (32,64,128,32,64,128):
                        request = body(tokens,True)
                        response,elapsed = complete(server,request)
                        outputs.append(stable(response))
                        raw.append(dict(phase='correctness',expert_major=expert_major,label=label,
                                        request=request,response=response,wall_ms=elapsed))
                    counters = trace(server)
                    require(counters and all(row['pending'] == 0 and row['key_changes'] == 0 for row in counters),
                            'Provider omitted drained counters or changed retained graph keys')
                    require(all(0 < row['gpu_workspace'] <= 16*1024*1024 for row in counters),
                            'Unexpected actual tiny-provider workspace')
                    if option == '1':
                        require(max(row['builds'] for row in counters) > 0, 'Actual provider never used preparation pool')
                        require(max(row['reuses'] for row in counters) > 0, 'Actual provider never reused a graph template')
                        require(max(row['cpu_metadata'] for row in counters) <= 4*1024*1024,
                                'CPU metadata pool exceeded tiny-fixture bound')
                    observed[label] = dict(outputs=outputs,counters=counters,modules=modules,
                                           command=server.command,log_sha256=None)
                observed[label]['log_sha256'] = sha(server.logfile)
            require(observed['disabled']['outputs'] == observed['enabled']['outputs'],
                    'Preparation reuse changed provider tokens, probabilities or response state')
            old_workspace = {row['gpu_workspace'] for row in observed['disabled']['counters']}
            new_workspace = {row['gpu_workspace'] for row in observed['enabled']['counters']}
            require(old_workspace == new_workspace, 'Preparation reuse changed actual provider GPU workspace')
            config = dict(expert_major=expert_major,exact_outputs=True,workspace_unchanged=True,
                          controls=observed,timing=None)
            report['configurations'].append(config)
            if args.cycles:
                with ExitStack() as stack:
                    controls = {label:stack.enter_context(server_for(launcher,args,f'timing-{expert_major}-{label}',
                                  option,expert_major,False)) for label,option in (('disabled','0'),('enabled','1'))}
                    request = body(64)
                    expected = None
                    pins = {}
                    for label,server in controls.items():
                        for _ in range(3):
                            response,elapsed = complete(server,request)
                            if expected is None: expected = stable(response)
                            require(stable(response) == expected,'Warm timing output differs')
                            raw.append(dict(phase='warmup',expert_major=expert_major,label=label,
                                            request=request,response=response,wall_ms=elapsed))
                        pins[label] = provenance.verify_loaded_payload(server,identity)
                    windows = []
                    for cycle in range(args.cycles):
                        for label in ('disabled','enabled','enabled','disabled'):
                            response,elapsed = complete(controls[label],request)
                            require(stable(response) == expected,'Timed provider output differs')
                            windows.append(dict(cycle=cycle,label=label,wall_ms=elapsed,timings=response.get('timings')))
                            raw.append(dict(phase='timing',expert_major=expert_major,label=label,
                                            request=request,response=response,wall_ms=elapsed))
                    timing_servers = {}
                    for label,server in controls.items():
                        after_modules = provenance.verify_loaded_payload(server,identity)
                        require(after_modules['validated_required_modules'] ==
                                pins[label]['validated_required_modules'], 'Loaded modules changed')
                        timing_servers[label] = dict(command=server.command,observed_modules=pins[label],
                            observed_modules_after=after_modules,log_sha256=None)
                    medians = {label:statistics.median(row['wall_ms'] for row in windows if row['label'] == label)
                               for label in controls}
                    config['timing'] = dict(cycles=args.cycles,order='ABBA',windows=windows,median_wall_ms=medians,
                         after_over_before=medians['enabled']/medians['disabled'],model_tokens_per_second_not_measured=True,
                         trace_disabled=True,log_verbosity=4,servers=timing_servers)
                for label,server in controls.items():
                    config['timing']['servers'][label]['log_sha256'] = sha(server.logfile)
        require(provenance.manifest(args.executable) == identity and sha(args.model) == fixture_hash,
                'Payload or original fixture changed during provider checks')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed',failure=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        raw_path = args.output_dir/'raw-requests.json'
        raw_path.write_text(json.dumps(raw,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        report['raw_requests_sha256'] = sha(raw_path)
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',configurations=2,exact_provider_outputs=True,workspace_unchanged=True,
                          model_bytes=args.model.stat().st_size,cycles=args.cycles)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cycles',type=int,default=0)
    parser.add_argument('--execute',action='store_true',help='Start the explicit tiny-fixture checks')
    args = parser.parse_args()
    for name in ('executable','model','output_dir'):
        setattr(args,name,getattr(args,name).resolve())
    if args.execute:
        run(args)
    else:
        print(json.dumps(dict(executed=False,executable=str(args.executable),model=str(args.model),
                              output_dir=str(args.output_dir),cycles=args.cycles)))
