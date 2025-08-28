#!/usr/bin/env python3
"""
Async CAFA‑5 inference against a single vLLM server
- Builds batched /generate requests with protein sequences
- Streams to http://HOST:PORT/generate/
"""

import argparse
import asyncio
import json
import os
import time
from typing import Any

import aiohttp
import numpy as np
from bioreason2.dataset.cafa5.collate import _coords_from_cif, _coords_from_pdb
from bioreason2.dataset.cafa5.load import load_cafa5_dataset
from bioreason2.utils import str2bool


def add_structures_to_dataset(dataset, max_length_protein: int = 2048, num_proc: int = 192):
    """
    Add structure coordinates to dataset using the same logic as collate function.
    """

    def process_structure(example):
        struct_path = example.get("structure_path")

        # Same logic as collate function lines 122-142
        if struct_path is not None and os.path.exists(struct_path):
            try:
                if struct_path.endswith(".cif"):
                    coords = _coords_from_cif(struct_path)
                elif struct_path.endswith(".pdb"):
                    coords = _coords_from_pdb(struct_path)
                else:
                    raise ValueError(f"Unsupported structure format: {struct_path}")
            except Exception:
                # On error, fall back to empty coordinates
                coords = np.full((0, 3, 3), np.nan)
        else:
            coords = np.full((0, 3, 3), np.nan)

        # Truncate if number of residues exceeds max_length_protein
        if coords.shape[0] > max_length_protein:
            coords = coords[:max_length_protein]

        # For empty coordinates, return None (helps schema alignment across shards)
        example["structure_coords"] = None if coords.shape[0] == 0 else coords.tolist()

        # Fix sequence field name consistency
        if "sequence" in example and "protein_sequences" not in example:
            example["protein_sequences"] = [example["sequence"]]

        return example

    print(f"Adding structure coordinates to {len(dataset)} samples using {num_proc} processes...")
    print(f"Expected processing time: ~{len(dataset) / (num_proc * 80):.1f} minutes (estimated)")
    return dataset.map(process_structure, num_proc=num_proc, desc="Adding structures")


_dumps = lambda o: json.dumps(o, indent=4)


async def _post(session: aiohttp.ClientSession, url: str, payload: dict[str, Any], timeout_s: int):
    async with session.post(url, json=payload, timeout=timeout_s) as resp:
        resp.raise_for_status()
        return await resp.json()


def _flatten_user_messages_to_text(prompt) -> str:
    def _to_text(item):
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            if item.get("type") == "text":
                return item.get("text", "")
            if "content" in item:
                return _to_text(item["content"])
        if isinstance(item, list):
            return " ".join(_to_text(x) for x in item)
        return ""

    if isinstance(prompt, list):
        user_msgs = [m for m in prompt if m.get("role") == "user"]
        user_texts = []
        for m in user_msgs:
            user_texts.append(_to_text(m.get("content", "")))
        return "\n".join(t.strip() for t in user_texts if t and t.strip())
    return str(prompt)


def _flatten_assistant_messages_to_text(prompt) -> str:
    def _to_text(item):
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            if item.get("type") == "text":
                return item.get("text", "")
            if "content" in item:
                return _to_text(item["content"])
        if isinstance(item, list):
            return " ".join(_to_text(x) for x in item)
        return ""

    if isinstance(prompt, list):
        assistant_msgs = [m for m in prompt if m.get("role") == "assistant"]
        assistant_texts = []
        for m in assistant_msgs:
            assistant_texts.append(_to_text(m.get("reasoning_content", "")))
            assistant_texts.append("\n\n")
            assistant_texts.append(_to_text(m.get("content", "")))
            assistant_texts.append("\n\n")
        return "\n".join(t.strip() for t in assistant_texts if t and t.strip())
    return str(prompt)


def join_batch_input_output(batch_input: dict, batch_output: dict, batch_index: int, samples) -> list[dict]:
    """
    Join batch input and output data into individual sample records.
    """
    joined_samples = []

    # Get the completions from the batch output
    batch_result = batch_output.get("result", {})
    completion_ids = batch_result.get("completion_ids", [])
    completions = batch_result.get("completions", [])

    # Get input data
    protein_ids = batch_input.get("protein_ids", [])
    prompts = batch_input.get("prompts", [])
    protein_sequences = batch_input.get("protein_sequences", [])
    go_aspects = batch_input.get("go_aspects", [])
    structure_coords = batch_input.get("structure_coords", [])

    # Process each sample in the batch
    for i in range(len(protein_ids)):
        sample_record = {
            "sample_id": i,
            "batch_index": batch_index,
            "protein_id": protein_ids[i] if i < len(protein_ids) else f"unknown_{i}",
            "prompt": prompts[i] if i < len(prompts) else "",
            "protein_sequences": protein_sequences[i] if i < len(protein_sequences) else [],
            "go_aspect": go_aspects[i] if go_aspects and i < len(go_aspects) else None,
            "generated_response": completions[i] if i < len(completions) else "",
            "full_response": completions[i] if i < len(completions) else "",
            "structure_coords": structure_coords[i] if structure_coords and i < len(structure_coords) else None,
            "structure_loaded": bool(
                structure_coords and i < len(structure_coords) and structure_coords[i] is not None
            ),
            "success": i < len(completions) and bool(completions[i]),
            "completion_id": completion_ids[i] if i < len(completion_ids) else f"batch_{batch_index}_sample_{i}",
        }

        # Try to get additional metadata from original samples if available
        try:
            original_sample_idx = batch_index * len(protein_ids) + i
            if hasattr(samples, "__getitem__") and original_sample_idx < len(samples):
                original_sample = samples[original_sample_idx]
                sample_record["ground_truth"] = original_sample.get("ground_truth", "")
                sample_record["structure_path"] = original_sample.get("structure_path", "")
        except (IndexError, AttributeError):
            sample_record["ground_truth"] = ""
            sample_record["structure_path"] = ""

        joined_samples.append(sample_record)

    return joined_samples


def build_batches(
    samples, batch_size: int, temperature: float, top_p: float, max_new_tokens: int, repetition_penalty: float
):
    batches = []
    filtered_errors = []
    
    for i in range(0, len(samples), batch_size):
        end = min(i + batch_size, len(samples))
        batch_ds = samples.select(range(i, end))
        protein_ids, prompts, assistant_texts, protein_seqs, go_aspects, structure_coords = [], [], [], [], [], []

        for s in batch_ds:
            # Check protein sequence length before adding to batch
            seqs = s.get("protein_sequences", [s["sequence"]])
            max_len_in_sample = max(len(seq) for seq in seqs) if seqs else 0
            
            if max_len_in_sample > 1500:
                # Filter out long proteins - add to error list
                error_record = {
                    "protein_id": s["protein_id"],
                    "sequence_length": max_len_in_sample,
                    "error": f"Protein sequence too long ({max_len_in_sample} > 1500). Filtered to prevent OOM.",
                    "prompt": _flatten_user_messages_to_text(s["prompt"]),
                    "protein_sequences": seqs,
                    "ground_truth": s.get("ground_truth", ""),
                    "structure_path": s.get("structure_path", ""),
                    "success": False
                }
                filtered_errors.append(error_record)
                print(f"⚠️ Filtered protein {s['protein_id']} (length: {max_len_in_sample})")
            else:
                # Safe to include in batch
                protein_ids.append(s["protein_id"])
                prompts.append(_flatten_user_messages_to_text(s["prompt"]))
                assistant_texts.append(_flatten_assistant_messages_to_text(s["prompt"]))
                protein_seqs.append(seqs)
                go_aspects.append(s.get("go_aspect") or None)
                structure_coords.append(s.get("structure_coords") or None)

        # Only create payload if we have samples in this batch
        if protein_ids:
            payload = {
                "protein_ids": protein_ids,
                "prompts": prompts,
                "assistant_texts": assistant_texts,
                "protein_sequences": protein_seqs,
                "go_aspects": go_aspects if any(a is not None for a in go_aspects) else None,
                "structure_coords": structure_coords if any(c is not None for c in structure_coords) else None,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
                "generation_kwargs": {},
            }
            batches.append(payload)

    return batches, filtered_errors


async def main(args):
    server = f"http://{args.host}:{args.port}"
    print(f"🚀 Server → {server}")

    print("📥 Loading CAFA‑5 validation split …")
    _, val_ds, _ = load_cafa5_dataset(
        dataset=args.cafa5_dataset,
        dataset_name=args.cafa5_dataset_name,
        cache_dir=args.dataset_cache_dir,
        dataset_subset=args.cafa5_dataset_subset,
        max_length=args.max_length_protein,
        seed=args.seed,
        val_split_ratio=args.val_split_ratio,
        return_as_chat_template=True,
        split_go_aspects=args.split_go_aspects,
        structure_dir=args.structure_dir,
        include_go_defs=args.include_go_defs,
        interpro_dataset_name=args.interpro_dataset_name,
        include_protein_function_summary=args.include_protein_function_summary,
        interpro_in_prompt=args.interpro_in_prompt,
        ppi_in_prompt=args.ppi_in_prompt,
        debug=args.debug,
    )

    n = len(val_ds) if args.max_samples <= 0 else min(args.max_samples, len(val_ds))
    samples = val_ds.select(range(n))
    samples = add_structures_to_dataset(samples, max_length_protein=args.max_length_protein)
    print(f"📊 {len(samples)} samples loaded")

    # Check if batches already exist on disk
    if os.path.exists(args.batch_inputs_dir) and os.listdir(args.batch_inputs_dir):
        print(f"🔄 Found existing batch inputs in {args.batch_inputs_dir} - loading from disk...")
        
        # Load existing batches
        batches = []
        batch_files = sorted([f for f in os.listdir(args.batch_inputs_dir) if f.startswith("batch_") and f.endswith(".json")])
        
        for batch_file in batch_files:
            batch_path = os.path.join(args.batch_inputs_dir, batch_file)
            with open(batch_path, "r") as f:
                batch = json.load(f)
                batches.append(batch)
        
        # Load existing filtered errors if available
        filtered_errors = []
        error_dir = os.path.join(os.path.dirname(args.batch_inputs_dir), "error_logs")
        prefilter_filename = os.path.join(error_dir, "prefiltered_long_proteins.json")
        
        if os.path.exists(prefilter_filename):
            with open(prefilter_filename, "r") as f:
                filtered_errors = json.load(f)
        
        total_batched_samples = sum(len(b["protein_ids"]) for b in batches)
        print(f"📁 Loaded {len(batches)} batches ({total_batched_samples} samples) and {len(filtered_errors)} filtered proteins from disk")
        
    else:
        print(f"🔨 Creating new batches...")
        batches, filtered_errors = build_batches(
            samples=samples,
            batch_size=args.request_batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )

        # Save filtered errors (long proteins) - only if newly created
        if filtered_errors:
            error_dir = os.path.join(os.path.dirname(args.batch_inputs_dir), "error_logs")
            os.makedirs(error_dir, exist_ok=True)
            
            prefilter_filename = os.path.join(error_dir, "prefiltered_long_proteins.json")
            with open(prefilter_filename, "w") as f:
                json.dump(filtered_errors, f, indent=4)
            print(f"🚫 Saved {len(filtered_errors)} long proteins (>1500 residues) → {prefilter_filename}")

        # Save batches - only if newly created
        if batches:
            os.makedirs(args.batch_inputs_dir, exist_ok=True)
            for i, batch in enumerate(batches):
                batch_filename = os.path.join(args.batch_inputs_dir, f"batch_{i}.json")
                with open(batch_filename, "w") as f:
                    json.dump(batch, f, indent=4)
                print(f"💾 Saved batch {i} payload → {batch_filename}")
    
    # Updated summary
    total_samples = len(samples)
    batched_samples = sum(len(b["protein_ids"]) for b in batches)
    filtered_samples = len(filtered_errors)
    print(f"📊 Summary: {total_samples} total → {batched_samples} batched, {filtered_samples} filtered (length > 1500)")

    t0 = time.time()
    connector = aiohttp.TCPConnector(limit=args.concurrent_requests)
    async with aiohttp.ClientSession(connector=connector, json_serialize=_dumps) as sess:
        sem = asyncio.Semaphore(args.concurrent_requests)

        async def worker(payload):
            async with sem:
                try:
                    result = await _post(sess, f"{server}/generate/", payload, args.client_timeout_sec)
                    # Check if server returned an error response (new error handling)
                    if isinstance(result, dict) and "error" in result:
                        return {"success": False, "result": None, "error": result["error"]}
                    return {"success": True, "result": result, "error": None}
                except Exception as e:
                    return {"success": False, "result": None, "error": str(e)}

        tasks = [asyncio.create_task(worker(p)) for p in batches]
        done, _ = await asyncio.wait(tasks)
        results = [t.result() for t in done]

    dt = time.time() - t0
    print(f"⏱️  {len(samples)} samples | {dt:.2f}s | {len(samples) / dt:.2f} samples/s")

    if args.save_results:
        os.makedirs(args.batch_outputs_dir, exist_ok=True)
        os.makedirs(args.joined_outputs_dir, exist_ok=True)
        error_dir = os.path.join(os.path.dirname(args.batch_outputs_dir), "error_logs")
        os.makedirs(error_dir, exist_ok=True)

        successful_batches = 0
        failed_batches = 0

        for i, batch_response in enumerate(results):
            batch_size = len(batches[i]["protein_ids"]) if i < len(batches) else 0

            if batch_response["success"]:
                # Handle successful batch
                result_out = {
                    "config": vars(args),
                    "batch_index": i,
                    "batch_size": batch_size,
                    "time_sec": dt,
                    "result": batch_response["result"],
                }
                result_filename = os.path.join(args.batch_outputs_dir, f"batch_{i}_results.json")
                with open(result_filename, "w") as f:
                    json.dump(result_out, f, indent=4)
                print(f"💾 Saved batch {i} results → {result_filename}")

                # Create joined output for this batch
                batch_input = batches[i]
                joined_samples = join_batch_input_output(batch_input, result_out, i, samples)

                joined_filename = os.path.join(args.joined_outputs_dir, f"batch_{i}_joined.json")
                with open(joined_filename, "w") as f:
                    json.dump(joined_samples, f, indent=4)
                print(f"🔗 Saved batch {i} joined data → {joined_filename}")
                successful_batches += 1

            else:
                # Handle failed batch
                error_out = {
                    "config": vars(args),
                    "batch_index": i,
                    "batch_size": batch_size,
                    "time_sec": dt,
                    "error": batch_response["error"],
                    "protein_ids": batches[i]["protein_ids"] if i < len(batches) else [],
                }
                error_filename = os.path.join(error_dir, f"batch_{i}_error.json")
                with open(error_filename, "w") as f:
                    json.dump(error_out, f, indent=4)
                print(f"❌ Saved batch {i} error → {error_filename}")
                failed_batches += 1

        print(f"📊 Final Summary: {successful_batches} successful batches, {failed_batches} failed batches, {len(filtered_errors)} prefiltered proteins")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Async CAFA inference against vLLM server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)

    # Dataset options (aligned with training)
    p.add_argument("--cafa5_dataset", type=str, default="wanglab/cafa5")
    p.add_argument("--cafa5_dataset_name", type=str, default="cafa5_reasoning")
    p.add_argument("--cafa5_dataset_subset", type=str, default=None)
    p.add_argument("--dataset_cache_dir", type=str, default="/large_storage/goodarzilab/bioreason/data/")
    p.add_argument("--structure_dir", type=str, default="/large_storage/goodarzilab/bioreason/data/sequences/")
    p.add_argument("--include_go_defs", type=str2bool, default=False)
    p.add_argument("--interpro_dataset_name", type=str, default="interpro_metadata")
    p.add_argument("--split_go_aspects", type=str2bool, default=True)
    p.add_argument("--interpro_in_prompt", type=str2bool, default=True)
    p.add_argument("--ppi_in_prompt", type=str2bool, default=True)
    p.add_argument("--include_protein_function_summary", type=str2bool, default=True)
    p.add_argument("--val_split_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--debug", type=str2bool, default=False)

    # Backward-compatibility aliases
    p.add_argument("--cafa5_hf_repo", type=str, default=None)
    p.add_argument("--cafa5_config", type=str, default=None)
    p.add_argument("--interpro_config", type=str, default=None)

    # Eval controls
    p.add_argument("--max_samples", type=int, default=128)
    p.add_argument("--max_length_protein", type=int, default=500)
    p.add_argument("--request_batch_size", type=int, default=16)
    p.add_argument("--concurrent_requests", type=int, default=8)

    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.0)

    p.add_argument("--save_results", action="store_true")
    p.add_argument("--batch_inputs_dir", type=str, default="batch_inputs")
    p.add_argument("--batch_outputs_dir", type=str, default="batch_outputs")
    p.add_argument("--joined_outputs_dir", type=str, default="joined_outputs")
    p.add_argument("--client_timeout_sec", type=int, default=1800)

    args = p.parse_args()

    # Normalize legacy flags to the new names if provided
    if getattr(args, "cafa5_hf_repo", None):
        args.cafa5_dataset = args.cafa5_hf_repo
    if getattr(args, "cafa5_config", None):
        args.cafa5_dataset_name = args.cafa5_config
    if getattr(args, "interpro_config", None):
        args.interpro_dataset_name = args.interpro_config

    asyncio.run(main(args))
