#!/usr/bin/env python3
"""Check bounded adaptive prefill on one payload and a tiny synthetic model.

No download, build, installation or trained-model inference is performed. The
positive path uses an explicit 100 us fixture-only goal. Optional quiet timings
use the ordinary relative goal, not that override, and report both latency and
throughput tradeoffs. Isolated greedy parity uses a validated multi-candidate
printable-ASCII support: allowed logits retain their model ranking, while unsafe
byte/special tokens are excluded for strict SSE token conservation. This is not
an unrestricted-vocabulary test. Concurrent natural outputs are compared and
reported without an equality guarantee. Import and invocation without --execute are inert.
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
import sys
import threading
import time
import urllib.request

if not __debug__:
    raise RuntimeError("Do not use Python -O; all validation checks are required")
sys.dont_write_bytecode = True

FIXTURE_BYTES = 14179936
FIXTURE_SHA256 = "71099f10a7bffa665c1e39b8e382bc6086a0923efb87480b72d8e008fdbeed63"


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def percentile(values, fraction):
    require(values, "No observed inter-token interval")
    ordered = sorted(values)
    offset = (len(ordered)-1)*fraction
    low, high = math.floor(offset), math.ceil(offset)
    return ordered[low]*(high-offset) + ordered[high]*(offset-low) if low != high else ordered[low]


def stable(value):
    if type(value) is dict:
        return {key: stable(item) for key, item in value.items() if key not in ("timings", "tokens_cached")}
    if type(value) is list:
        return [stable(item) for item in value]
    return value


class Stream:
    """One owned request. Timestamps are client observations, not GPU events."""
    def __init__(self, server, body, label, threshold=0, cancel_prompt=False):
        self.server, self.body, self.label = server, body, label
        self.threshold, self.cancel_prompt = threshold, cancel_prompt
        self.threshold_reached, self.completed = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.events, self.token_times_ns, self.tokens, self.content = [], [], [], []
        self.started_ns = self.headers_ns = self.finished_ns = None
        self.first_content_ns = self.cancelled_ns = None
        self.cancelled_progress = self.terminal = self.error = None
        self.thread = threading.Thread(target=self._read, name="prefill-"+label, daemon=True)

    def start(self):
        self.thread.start()

    def _read(self):
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            request = urllib.request.Request(self.server.url+"/completion", json.dumps(self.body).encode("utf-8"),
                                             {"Content-Type":"application/json"})
            self.started_ns = time.perf_counter_ns()
            with opener.open(request, timeout=120) as response:
                self.headers_ns = time.perf_counter_ns()
                require("text/event-stream" in response.headers.get("Content-Type", ""), "Missing SSE framing")
                for raw in response:
                    received = time.perf_counter_ns()
                    line = raw.decode("utf-8").strip()
                    if not line or line.startswith(":"):
                        continue
                    require(line.startswith("data: "), "Unexpected SSE field")
                    encoded = line[6:]
                    if encoded == "[DONE]":
                        break
                    event = json.loads(encoded)
                    require(type(event) is dict and "error" not in event, "Completion stream returned an error")
                    require(event.get("id_slot") == self.body["id_slot"], "Explicit slot assignment was ignored")
                    with self.lock:
                        self.events.append(dict(received_ns=received,event=event))
                        progress = event.get("prompt_progress")
                        if self.cancel_prompt and type(progress) is dict and 0 < progress.get("processed",0) < progress.get("total",0):
                            require(self.first_content_ns is None, "Cancellation happened after B began generating")
                            self.cancelled_progress = progress
                            self.cancelled_ns = received
                            break
                        if event.get("stop") is True:
                            self.terminal = event
                            break
                        if type(progress) is not dict:
                            ids = event.get("tokens", [])
                            require(type(ids) is list, "Missing partial token ID array")
                            valid = [token for token in ids if type(token) is int and token >= 0]
                            self.tokens.extend(valid)
                            self.token_times_ns.extend([received]*len(valid))
                            text = event.get("content", "")
                            require(type(text) is str, "Partial content is not text")
                            self.content.append(text)
                            if text and self.first_content_ns is None:
                                self.first_content_ns = received
                            if self.threshold and len(self.tokens) >= self.threshold:
                                self.threshold_reached.set()
                if self.cancel_prompt:
                    require(self.cancelled_ns is not None, "No partial B prefill was observed before cancellation")
                else:
                    require(self.terminal is not None, "SSE stream omitted its terminal event")
        except BaseException as error:
            self.error = error
        finally:
            self.finished_ns = time.perf_counter_ns()
            self.completed.set()

    def join(self):
        self.thread.join(130)
        require(not self.thread.is_alive(), "Owned SSE reader did not finish")
        if self.error is not None:
            raise RuntimeError(self.label+": "+str(self.error)) from self.error

    def result(self):
        self.join()
        return dict(label=self.label,request=self.body,started_ns=self.started_ns,headers_ns=self.headers_ns,
                    finished_ns=self.finished_ns,first_content_ns=self.first_content_ns,
                    token_times_ns=self.token_times_ns,tokens=self.tokens,content="".join(self.content),
                    events=self.events,terminal=self.terminal,cancelled_ns=self.cancelled_ns,
                    cancelled_progress=self.cancelled_progress)


def wait_idle(server):
    deadline = time.monotonic()+30
    while True:
        rows = server.request("/slots")
        require(type(rows) is list and len(rows) == 2, "Missing actual two-slot state")
        require({row.get("id") for row in rows} == {0,1}, "Unexpected slot identity inventory")
        if not any(row.get("is_processing",False) for row in rows):
            return rows
        require(time.monotonic() < deadline, "Cancellation left an orphan active slot")
        time.sleep(.02)


def semantics(result, body, forced_token=None):
    terminal = result["terminal"]
    require(type(terminal) is dict and terminal.get("stop") is True and terminal.get("stop_type") == "limit" and
            terminal.get("truncated") is False and terminal.get("stopping_word") == "", "Completion stop semantics changed")
    require(terminal.get("tokens_predicted") == body["n_predict"] and
            len(result["tokens"]) == body["n_predict"], "Generated token count changed")
    require(terminal.get("tokens_evaluated") == len(body["prompt"]), "Prompt tokens were omitted or truncated")
    if forced_token is not None:
        require(result["tokens"] == [forced_token]*body["n_predict"] and
                result["content"] == "A"*body["n_predict"], "ASCII diagnostic bias did not produce the intended stream")
    return dict(content=result["content"],tokens=result["tokens"],terminal=stable(terminal))


def marker_rows(server, start=0):
    text = server.log_text(start)
    output = {"active":[],"summaries":[],"memory":[],"raw_lines":dict(active=[],summaries=[],memory=[])}
    for line in text.splitlines():
        for name, key in (("ADAPTIVE_PREFILL_ACTIVE", "active"),("ADAPTIVE_PREFILL_SUMMARY","summaries"),
                          ("ADAPTIVE_PREFILL_MEMORY","memory")):
            if name in line:
                fields = re.findall(r"([a-z_]+)=(\d+)", line.split(name,1)[1])
                require(fields, "Adaptive prefill marker omitted its fields")
                output[key].append({name:int(value) for name,value in fields})
                output["raw_lines"][key].append(line)
    output["active_lines"], output["summary_lines"] = len(output["active"]), len(output["summaries"])
    output["memory_lines"] = len(output["memory"])
    return output


def wait_positive_markers(server, start, require_adaptation=False):
    """Let the asynchronous logger drain after the observed idle boundary."""
    deadline = time.monotonic()+3
    while True:
        rows = marker_rows(server,start)
        if rows["memory"] and (not require_adaptation or (rows["active"] and rows["summaries"])):
            return rows
        require(server.process.poll() is None,"Native server exited while flushing adaptive markers")
        require(time.monotonic() < deadline,"Adaptive correctness/memory markers were not flushed after idle")
        time.sleep(.02)


def compare_memory_peaks(disabled, enabled):
    """Compare actual cumulative context peaks, including ordinary reservation."""
    fields = ("primary_peak_bytes","primary_gpu_peak_bytes","secondary_peak_bytes")
    peaks = {}
    for label,markers in (("disabled",disabled),("enabled",enabled)):
        rows = markers["memory"]
        require(rows and all(row.get("calls",0) > 0 and row.get("scope_restored") == 1 for row in rows),
                "Actual idle memory/restoration counters were not observed")
        require(all(type(row.get(field)) is int and row[field] >= 0 for row in rows for field in fields),
                "Actual context peak fields are missing or invalid")
        peaks[label] = {field:max(row[field] for row in rows) for field in fields}
    require(peaks["disabled"]["primary_peak_bytes"] > 0 and peaks["enabled"]["primary_peak_bytes"] > 0,
            "Actual primary memory was not measured")
    require(all(peaks["enabled"][field] <= peaks["disabled"][field] for field in fields),
            "Adaptive context memory peak exceeds the actual disabled control")
    return dict(disabled=peaks["disabled"],enabled=peaks["enabled"],actual_peaks_not_increased=True,
                ordinary_reservation_included=True,scope="Context-lifetime managed peaks of this fixture; no universal VRAM cap")


def pair_metrics(a, b):
    require(a["started_ns"] < b["started_ns"] < a["finished_ns"], "B did not overlap an active A stream")
    require(a["first_content_ns"] is not None and b["first_content_ns"] is not None, "Forced streams omitted nonempty content")
    times = a["token_times_ns"]
    all_gaps = [(right-left)/1e6 for left,right in zip(times,times[1:])]
    interference = [(right-left)/1e6 for left,right in zip(times,times[1:])
                    if right > b["started_ns"] and left < b["first_content_ns"]]
    require(interference, "No observed A interval overlapped B prefill")
    wall_ms = (max(a["finished_ns"],b["finished_ns"])-a["started_ns"])/1e6
    def gap_stats(values):
        return dict(samples=len(values),mean_ms=statistics.mean(values),max_ms=max(values),p95_ms=percentile(values,.95))
    result = dict(pair_wall_ms=wall_ms,a_intertoken=gap_stats(all_gaps),a_during_b_prefill=gap_stats(interference),
                  a_tokens_observed_during_b_prefill=sum(b["started_ns"] <= t <= b["first_content_ns"] for t in times),
                  a_first_content_ms=(a["first_content_ns"]-a["started_ns"])/1e6,
                  b_first_content_ms=(b["first_content_ns"]-b["started_ns"])/1e6,
                  a_total_ms=(a["finished_ns"]-a["started_ns"])/1e6,
                  b_total_ms=(b["finished_ns"]-b["started_ns"])/1e6,
                  pair_end_to_end_tokens_per_second=(len(a["tokens"])+len(b["tokens"]))*1000/wall_ms,
                  native_tokens_per_second={name: row["terminal"]["timings"]["predicted_per_second"]
                      for name,row in (("a",a),("b",b))})
    require(all(math.isfinite(x) and x > 0 for x in (result["pair_wall_ms"],result["b_first_content_ms"],
            result["pair_end_to_end_tokens_per_second"],*result["native_tokens_per_second"].values())), "Invalid pair timing")
    return result


def paired(server, a_body, b_body, label, cancel=False, forced_token=None):
    start = server.logfile.stat().st_size
    a = Stream(server,a_body,label+"-a",threshold=16)
    b = None
    try:
        a.start()
        deadline = time.monotonic()+60
        while not a.threshold_reached.wait(.005):
            if a.completed.is_set():
                a.join()
                raise AssertionError("A finished before reaching the launch threshold")
            require(time.monotonic() < deadline,"A did not produce 16 launch tokens")
        with a.lock:
            require(len(a.tokens) >= 16 and not a.completed.is_set(), "A completed before B could be launched")
        b = Stream(server,b_body,label+"-b",cancel_prompt=cancel)
        b.start()
        a_result, b_result = a.result(), b.result()
        idle = wait_idle(server)
        # Read marker IO only after the pair's latency window has ended.
        time.sleep(.03)
        markers = marker_rows(server,start)
        semantic_a = semantics(a_result,a_body,forced_token)
        if cancel:
            require(b_result["terminal"] is None and b_result["first_content_ns"] is None,
                    "B cancellation happened after generation")
            require(0 < b_result["cancelled_progress"]["processed"] < b_result["cancelled_progress"]["total"],
                    "B was not partially prefilling at client cancellation")
            metric = None
            semantic_b = None
        else:
            semantic_b = semantics(b_result,b_body,forced_token)
            metric = pair_metrics(a_result,b_result) if forced_token is not None else None
        return dict(label=label,log_start=start,a=a_result,b=b_result,semantics=dict(a=semantic_a,b=semantic_b),
                    metrics=metric,markers=markers,idle_slots=idle,cancelled_b_prefill=cancel)
    finally:
        active = [stream for stream in (a,b) if stream is not None and stream.thread.is_alive()]
        if active:
            # Only this selected test child's failure path is stopped; production
            # endpoints are never contacted by this harness.
            server.close()
            for stream in active:
                stream.thread.join(5)


def configured(launcher, args, label, setting, verbosity, positive=False):
    server = launcher.NativeServer(args.executable,args.model,args.output_dir/label,args.gpu)
    for name,value in (("--ctx-size","8192"),("--batch-size","512"),("--ubatch-size","128"),("--parallel","2"),
                       ("--log-verbosity",str(verbosity))):
        server.command[server.command.index(name)+1] = value
    server.command.extend(("--kv-unified","--slots","--override-kv","llama.context_length=int:8192",
                           "--cpu-mask","14","--cpu-strict","1"))
    server.env["WILDLLAMA_ADAPTIVE_PREFILL"] = setting
    server.env["WILDLLAMA_GRAPH_VARIANTS_MIB"] = "128"
    if positive:
        server.env["WILDLLAMA_PREFILL_LATENCY_US"] = "100"
        server.env["WILDLLAMA_PREFILL_TRACE"] = "1"
    else:
        server.env.pop("WILDLLAMA_PREFILL_LATENCY_US",None)
        server.env.pop("WILDLLAMA_PREFILL_TRACE",None)
    return server


def diagnostic_bodies(server):
    ascii_ids = server.request("/tokenize",{"content":"A","add_special":False})["tokens"]
    exact = [token for token in dict.fromkeys(ascii_ids)
             if server.request("/detokenize",{"tokens":[token]})["content"] == "A"]
    require(len(exact) == 1,"Fixture does not expose an exact ASCII A token")
    prompt = server.request("/tokenize",{"content":"Ola! teste de fluxo.","add_special":True})["tokens"]
    base = dict(temperature=0.0,seed=1234,samplers=["temperature"],repeat_penalty=1.0,frequency_penalty=0.0,
                presence_penalty=0.0,dry_multiplier=0.0,xtc_probability=0.0,backend_sampling=True,
                n_probs=0,cache_prompt=False,stream=True,return_tokens=True,ignore_eos=True,
                logit_bias=[[exact[0],100.0]])
    a = dict(base,prompt=prompt,n_predict=512,id_slot=0,return_progress=False)
    b = dict(base,prompt=[1]+[3+(i*13+11)%256 for i in range(2999)],n_predict=64,id_slot=1,return_progress=False)
    return exact[0],a,b


def printable_ascii_support(server):
    """Validate this pinned byte vocabulary without selecting one forced token.

    Native SSE can suppress intermediate token IDs while incomplete UTF-8 bytes
    accumulate. Use many independently validated printable tokens so this test
    can keep its exact per-token conservation requirement. No model is altered.
    """
    metadata = server.request("/v1/models")
    rows = metadata.get("data")
    require(type(rows) is list and len(rows) == 1 and type(rows[0]) is dict,
            "Fixture model inventory is incomplete")
    vocab = rows[0].get("meta",{}).get("n_vocab")
    require(type(vocab) is int and vocab == 259,"Select the pinned 259-token random-weight fixture")
    pieces,tokenization = {},[]
    for code in range(32,127):
        text = chr(code)
        response = server.request("/tokenize",{"content":text,"add_special":False})
        tokens = response.get("tokens")
        require(type(tokens) is list and tokens and all(type(token) is int and 0 <= token < vocab for token in tokens),
                "Printable fixture tokenization returned invalid IDs")
        tokenization.append(dict(content=text,tokens=tokens))
        for token in dict.fromkeys(tokens):
            if token not in pieces:
                value = server.request("/detokenize",{"tokens":[token]}).get("content")
                require(type(value) is str,"Fixture detokenization returned non-text content")
                pieces[token] = value
    allowed = {token:piece for token,piece in pieces.items()
               if len(piece) == 1 and 32 <= ord(piece) <= 126 and
               any(row["content"] == piece and token in row["tokens"] for row in tokenization)}
    require(len(allowed) >= 32 and len(set(allowed.values())) >= 32,
            "Fixture needs at least 32 independently round-tripped printable ASCII candidates")
    excluded = [token for token in range(vocab) if token not in allowed]
    require(excluded and len(allowed) < vocab,"Unsafe/special fixture tokens were not excluded")
    bias = [[token,-1e9] for token in excluded]
    evidence = dict(model_metadata_response=metadata,vocabulary_tokens=vocab,minimum_allowed_tokens=32,
                    allowed_token_pieces=[dict(id=token,piece=allowed[token]) for token in sorted(allowed)],
                    excluded_token_ids=excluded,excluded_logit_bias=-1e9,allowed_logit_bias_omitted=True,
                    tokenization_evidence=tokenization,
                    scope="Fixture-only multi-candidate printable ASCII support; greedy model ranking among allowed tokens, not unrestricted-vocabulary parity")
    return allowed,bias,evidence


def validate_ascii_support(pair, allowed):
    for name in ("a","b"):
        value = pair[name]
        require(all(token in allowed for token in value["tokens"]),"Greedy stream escaped validated printable support")
        require(value["content"] == "".join(allowed[token] for token in value["tokens"]),
                "Printable stream content and conserved token IDs disagree")


def validate_positive(markers):
    require(markers["active"] and markers["summaries"],"Enabled fixture did not demonstrate actual adaptive prefill")
    for active in markers["active"]:
        require(active.get("generation_rows",0) > 0 and active.get("pending_prompts",0) > 0 and
                active.get("target_us") == 100 and active.get("max_consecutive_deferrals") == 1,
                "Adaptive fixture activation conditions differ")
    for row in markers["summaries"]:
        require(row.get("deferred",0) > 0 and row.get("forced_admissions",0) > 0 and
                row.get("pure_samples",0) >= 2 and row.get("mixed_samples",0) >= 2 and
                row.get("reuse_scopes",0) > 0 and row.get("primary_growth_bytes") == 0 and
                row.get("primary_gpu_growth_bytes") == 0 and
                row.get("secondary_growth_bytes") == 0 and row.get("scope_restored") == 1,
                "Adaptive fixture grew memory after adaptation began or failed restoration")
        for field in ("ordinary_primary_growth_bytes","ordinary_primary_gpu_growth_bytes", "ordinary_primary_before_bytes",
                      "ordinary_primary_after_bytes","ordinary_primary_gpu_before_bytes","ordinary_primary_gpu_after_bytes",
                      "primary_peak_bytes","primary_gpu_peak_bytes","secondary_peak_bytes"):
            require(type(row.get(field)) is int and row[field] >= 0,"Ordinary allocation/actual peak telemetry is incomplete")
        for prefix in ("primary","primary_gpu"):
            before,after = row["ordinary_"+prefix+"_before_bytes"],row["ordinary_"+prefix+"_after_bytes"]
            require(row["ordinary_"+prefix+"_growth_bytes"] >= max(0,after-before) and row[prefix+"_peak_bytes"] >= after,
                    "Ordinary growth maxima/last observed call/actual peak counters disagree")


def concurrent_natural_comparison(before, after):
    result = {}
    for name in ("a","b"):
        left,right = before[name],after[name]
        require(left["request"] == right["request"],"Natural concurrent control requests differ")
        indices = [index for index,(a,b) in enumerate(zip(left["tokens"],right["tokens"])) if a != b]
        text_indices = [index for index,(a,b) in enumerate(zip(left["content"],right["content"])) if a != b]
        result[name] = dict(exact_outputs=before["semantics"][name] == after["semantics"][name],
            token_difference_indices_zero_based=indices,first_different_token_index=None if not indices else indices[0],
            first_different_tokens=None if not indices else [left["tokens"][indices[0]],right["tokens"][indices[0]]],
            first_different_text_character_index=None if not text_indices else text_indices[0],
            token_sha256={label:hashlib.sha256(json.dumps(row["tokens"],separators=(",",":")).encode("utf-8")).hexdigest()
                          for label,row in (("disabled",left),("enabled",right))},
            text_sha256={label:hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
                         for label,row in (("disabled",left),("enabled",right))},
            native_token_counts=[left["terminal"]["tokens_predicted"],right["terminal"]["tokens_predicted"]],
            observed_token_counts=[len(left["tokens"]),len(right["tokens"])],
            terminal_semantics_equal=before["semantics"][name]["terminal"] == after["semantics"][name]["terminal"])
    return dict(exact_all_outputs=all(row["exact_outputs"] for row in result.values()),sequences=result,
                scope="Actual concurrent printable-support outputs; changed batch composition may change greedy choices, so equality is reported rather than promised")


def isolated_natural(args, launcher, provenance, identity, raw):
    """Fresh OFF/ON processes, identical A-then-B histories, before contention."""
    result = dict(servers={},cases=[],fresh_servers=True,request_order=["a","b"])
    observed = {}
    allowed,bias = None,None
    for label,setting in (("disabled","0"),("enabled","1")):
        server = configured(launcher,args,"isolated-"+label,setting,5,positive=True)
        with server:
            if allowed is None:
                _,base_a,base_b = diagnostic_bodies(server)
                allowed,bias,evidence = printable_ascii_support(server)
                result["ascii_support"] = evidence
            bodies = dict(a=dict(base_a,logit_bias=bias),b=dict(base_b,logit_bias=bias))
            values = {}
            start = server.logfile.stat().st_size
            for name in ("a","b"):
                request = Stream(server,bodies[name],"isolated-"+label+"-"+name); request.start()
                values[name] = request.result(); wait_idle(server)
            markers = wait_positive_markers(server,start)
            require(not markers["active"] and not markers["summaries"],"An isolated request unexpectedly adapted")
            pair = dict(label="isolated-"+label,a=values["a"],b=values["b"],metrics=None,
                        semantics={name:semantics(values[name],bodies[name]) for name in ("a","b")},
                        markers=markers,cancelled_b_prefill=False,log_start=start)
            validate_ascii_support(pair,allowed)
            require(values["a"]["finished_ns"] < values["b"]["started_ns"],"Isolated histories overlapped")
            observed[label] = pair["semantics"]; raw.append(pair)
            modules = provenance.verify_loaded_payload(server,identity)
            result["servers"][label] = dict(command=server.command,observed_modules=modules,
                environment_controls={key:value for key,value in server.env.items() if key.startswith("WILDLLAMA_")},
                actual_markers=marker_rows(server))
            result["cases"].append(dict(case="isolated-natural-ascii-"+label,exact_outputs=True,
                                       allowed_candidates=len(allowed),strict_sse_token_conservation=True))
        result["servers"][label]["log_sha256"] = sha(server.logfile)
    require(observed["disabled"] == observed["enabled"],"Fresh isolated natural ASCII outputs changed")
    result["memory_peak_comparison"] = compare_memory_peaks(result["servers"]["disabled"]["actual_markers"],
                                                           result["servers"]["enabled"]["actual_markers"])
    result.update(status="passed",exact_outputs=True)
    return result,allowed


def correctness(args, launcher, provenance, identity, raw):
    isolated,isolated_allowed = isolated_natural(args,launcher,provenance,identity,raw)
    report = {"servers":{},"cases":list(isolated["cases"]),"isolated_natural":isolated,
              "fixture_goal_override_us":100,"timing_claimed":False}
    observed, recovery, positive_markers = {}, {}, {}
    with ExitStack() as stack:
        servers = {}
        for label,setting in (("disabled","0"),("enabled","1")):
            server = configured(launcher,args,"positive-"+label,setting,5,positive=True)
            stack.enter_context(server); servers[label] = server
        forced,base_a,base_b = diagnostic_bodies(servers["disabled"])
        report["forced_ascii_token"] = forced
        for label,server in servers.items():
            positive = paired(server,base_a,base_b,"positive-"+label,forced_token=forced)
            raw.append(positive); observed[label] = positive["semantics"]
            positive["markers"] = wait_positive_markers(server,positive["log_start"],require_adaptation=label == "enabled")
            positive_markers[label] = positive["markers"]
            if label == "enabled":
                validate_positive(positive["markers"])
            else: require(not positive["markers"]["active"] and not positive["markers"]["summaries"],"Disabled control admitted adaptive prefill")
            report["cases"].append(dict(case="forced-concurrent-"+label,exact_outputs=True,actual_markers=positive["markers"]))
        require(observed["disabled"] == observed["enabled"],"Forced concurrent stream outputs changed")
        report["positive_memory_peak_comparison"] = compare_memory_peaks(positive_markers["disabled"],positive_markers["enabled"])
        allowed,ascii_bias,ascii_evidence = printable_ascii_support(servers["disabled"])
        require(allowed == isolated_allowed,"Concurrent and fresh isolated printable support differs")
        require(forced in allowed,"Forced-token and printable-support fixture evidence disagree")
        report["unforced_ascii_support"] = ascii_evidence
        unforced = {}
        for label,server in servers.items():
            a,b = dict(base_a),dict(base_b)
            a["logit_bias"] = ascii_bias; b["logit_bias"] = ascii_bias
            pair = paired(server,a,b,"unforced-"+label)
            validate_ascii_support(pair,allowed)
            pair["markers"] = wait_positive_markers(server,pair["log_start"])
            raw.append(pair); unforced[label] = pair
            report["cases"].append(dict(case="unforced-ascii-support-concurrent-"+label,actual_markers=pair["markers"],
                allowed_candidates=len(allowed),model_greedy_ranking_preserved=True,strict_sse_token_conservation=True,
                distinct_output_tokens={name:len(set(pair[name]["tokens"])) for name in ("a","b")}))
        report["concurrent_natural_comparison"] = concurrent_natural_comparison(unforced["disabled"],unforced["enabled"])
        for label,server in servers.items():
            cancelled = paired(server,base_a,dict(base_b,return_progress=True),"cancel-"+label,cancel=True,forced_token=forced)
            cancelled["markers"] = wait_positive_markers(server,cancelled["log_start"])
            raw.append(cancelled)
            restored = []
            for slot in (0,1):
                start = server.logfile.stat().st_size
                body = dict(base_a,id_slot=slot,n_predict=16)
                request = Stream(server,body,"recover-"+label+"-"+str(slot)); request.start()
                value = request.result(); wait_idle(server)
                value["markers"] = wait_positive_markers(server,start)
                restored.append(semantics(value,body,forced)); raw.append(value)
            recovery[label] = restored
            report["cases"].append(dict(case="cancel-prefill-and-recover-"+label,
                cancelled_prompt_progress=cancelled["b"]["cancelled_progress"],both_slots_idle=True,recovery_exact=True))
            modules = provenance.verify_loaded_payload(server,identity)
            (server.directory/"modules.json").write_text(json.dumps(modules,indent=2)+"\n",encoding="utf-8")
            report["servers"][label] = dict(command=server.command,environment_controls={key:value for key,value in server.env.items()
                if key.startswith("WILDLLAMA_")},actual_markers=marker_rows(server),observed_modules=modules)
        require(recovery["disabled"] == recovery["enabled"],"Cancellation changed subsequent requests")
        report["complete_correctness_memory_peak_comparison"] = compare_memory_peaks(
            report["servers"]["disabled"]["actual_markers"],report["servers"]["enabled"]["actual_markers"])
    for label,server in servers.items(): report["servers"][label]["log_sha256"] = sha(server.logfile)
    report.update(status="passed",forced_outputs_exact=True,isolated_ascii_support_outputs_exact=True,
                  concurrent_natural_outputs_compared_without_equality_claim=True,
                  cancelled_during_observed_partial_prefill=True,actual_bounded_reuse=True,
                  adaptation_phase_growth_checked=True,ordinary_reservation_growth_retained_in_raw_markers=True,
                  ordinary_growth_definition="Cumulative maximum positive delta; before/after fields retain the LAST ordinary growth call, which may differ from the maximum call")
    return report,forced,base_a,base_b


def timings(args, launcher, provenance, identity, forced, a_body, b_body, raw):
    proofs,windows,warmups = {},[],[]
    expected = None
    with ExitStack() as stack:
        servers = {}
        for label,setting in (("disabled","0"),("enabled","1")):
            server = configured(launcher,args,"timing-"+label,setting,4)
            stack.enter_context(server); servers[label] = server
        for label,server in servers.items():
            require("WILDLLAMA_PREFILL_LATENCY_US" not in server.env,"Positive-path latency override leaked into timing")
            require("WILDLLAMA_PREFILL_TRACE" not in server.env,"Correctness memory tracing leaked into timing")
            for index in range(3):
                pair = paired(server,a_body,b_body,"warm-"+label+"-"+str(index),forced_token=forced)
                if expected is None: expected = pair["semantics"]
                require(pair["semantics"] == expected,"Timing warmup outputs changed")
                warmups.append(dict(label=label,index=index,metrics=pair["metrics"],actual_markers=pair["markers"]))
                raw.append(pair)
            modules = provenance.verify_loaded_payload(server,identity)
            proofs[label] = dict(command=server.command,observed_modules=modules)
        for cycle in range(args.cycles):
            order = ("disabled","enabled","enabled","disabled") if cycle%2 == 0 else ("enabled","disabled","disabled","enabled")
            for label in order:
                pair = paired(servers[label],a_body,b_body,"timed-"+label+"-"+str(len(windows)),forced_token=forced)
                require(pair["semantics"] == expected,"Timed concurrent streams changed outputs")
                windows.append(dict(label=label,cycle=cycle,metrics=pair["metrics"],actual_markers=pair["markers"]))
                raw.append(pair)
        for label,server in servers.items():
            after = provenance.verify_loaded_payload(server,identity)
            require(after["validated_required_modules"] == proofs[label]["observed_modules"]["validated_required_modules"],
                    "Timing process loaded different core modules")
            proofs[label]["observed_modules_after"] = after
            proofs[label]["actual_markers"] = marker_rows(server)
    for label,server in servers.items(): proofs[label]["log_sha256"] = sha(server.logfile)
    def medians(label):
        rows = [row["metrics"] for row in windows if row["label"] == label]
        result = {key:statistics.median(row[key] for row in rows) for key in
            ("pair_wall_ms","pair_end_to_end_tokens_per_second","a_first_content_ms","b_first_content_ms",
             "a_total_ms","b_total_ms","a_tokens_observed_during_b_prefill")}
        for name in ("a_intertoken","a_during_b_prefill"):
            result[name] = {key:statistics.median(row[name][key] for row in rows) for key in ("mean_ms","max_ms","p95_ms")}
        result["native_tokens_per_second"] = {name:statistics.median(row["native_tokens_per_second"][name] for row in rows) for name in ("a","b")}
        return result
    return dict(status="passed",cycles=args.cycles,order="ABBA",warmups_per_control=3,windows=windows,warmups=warmups,
        medians={label:medians(label) for label in servers},servers=proofs,log_verbosity=4,
        forced_latency_override=False,ordinary_relative_goal=True,client_timestamps="perf_counter_ns after SSE line read, before JSON parse",
        info_marker_io_included=True,info_marker_lines={label:dict(active=proofs[label]["actual_markers"]["active_lines"],
            summaries=proofs[label]["actual_markers"]["summary_lines"],memory=proofs[label]["actual_markers"]["memory_lines"]) for label in servers},
        correctness_memory_trace_enabled=False,
        actual_default_goal_activation_observed=bool(proofs["enabled"]["actual_markers"]["summaries"]),
        scope="Same-binary tiny random-weight two-slot controls; reports responsiveness/TTFT/throughput tradeoffs, not trained-model TPS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--gpu",action="store_true")
    parser.add_argument("--cycles",type=int,default=0)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(executed=False,scope="No build/model/server is started without --execute",cycles=args.cycles)))
        return
    args.executable,args.model,args.output_dir = (path.resolve() for path in (args.executable,args.model,args.output_dir))
    require(args.executable.is_file() and args.model.is_file(),"Explicit existing payload and fixture required")
    require(0 < args.model.stat().st_size <= 64*1024*1024,"Only local fixtures up to64MiB")
    if os.name == "nt":
        require(all(path.drive.upper() == "G:" for path in (args.executable,args.model,args.output_dir)),
                "The selected payload, fixture and all output must remain on G:")
    require(args.cycles == 0 or 4 <= args.cycles <= 20,"Use zero or4-20 ABBA timing cycles")
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()),"Use a fresh output directory")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    launcher_path = Path(__file__).with_name("native_execution_features_smoke.py")
    provenance_path = Path(__file__).with_name("native_payload_provenance.py")
    launcher = module(launcher_path,"adaptive_prefill_launcher")
    provenance = module(provenance_path,"adaptive_prefill_provenance")
    identity = provenance.manifest(args.executable)
    require(len(identity["files_sha256"]) == 39,"Select the complete39-file packaged payload")
    fixture_sha = sha(args.model)
    require(args.model.stat().st_size == FIXTURE_BYTES and fixture_sha == FIXTURE_SHA256,
            "Select the existing pinned random-weight dense fixture; no trained model is supported")
    report = dict(status="running",schema=1,gpu=args.gpu,fixture_bytes=args.model.stat().st_size,fixture_sha256=fixture_sha,
                  harness_sha256=sha(Path(__file__)),launcher_sha256=sha(launcher_path),provenance_harness_sha256=sha(provenance_path),
                  payload_identity=identity,actual_loaded_core_module_pins=False,model_downloads=0,
                  configured_context=8192,slots=2,batch=512,ubatch=128,graph_cache_mib_per_context=128,
                  automatic_feature_default_enabled=False)
    raw = []
    try:
        correctness_report,forced,a_body,b_body = correctness(args,launcher,provenance,identity,raw)
        report["correctness"] = correctness_report
        if args.cycles: report["timing"] = timings(args,launcher,provenance,identity,forced,a_body,b_body,raw)
        report.update(status="passed",actual_loaded_core_module_pins=True)
    except BaseException as error:
        report.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        provenance.unchanged(identity)
        require(sha(args.model) == fixture_sha,"Diagnostic fixture changed during validation")
        responses = args.output_dir/"responses.json"
        responses.write_text(json.dumps(raw,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        report["responses_sha256"] = sha(responses)
        (args.output_dir/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="passed",gpu=args.gpu,cases=len(report["correctness"]["cases"]),
                         timing_cycles=args.cycles,scope="Only tiny synthetic two-slot controls")))


if __name__ == "__main__":
    main()
