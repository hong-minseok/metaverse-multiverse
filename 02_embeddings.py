import argparse
import base64
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = json.loads(Path(__file__).with_name("models.json").read_text())


def request_body(model, texts):
    provider, name, dim = model["provider"], model["model_id"], model["dimension"]
    if provider == "google":
        config = {"autoTruncate": False, "outputDimensionality": dim}
        if model["task"]:
            config["taskType"] = model["task"]
        return {"requests": [
            {"model": "models/" + name, "embedContentConfig": config,
             "content": {"parts": [{"text": model["prompt_text"] + text}]}}
            for text in texts
        ]}
    if provider == "cohere":
        body = dict(model=name, texts=texts, input_type=model["input_type"],
                    embedding_types=["float"], truncate="NONE")
        if model["request_dimensions"]:
            body["output_dimension"] = dim
        return body
    if provider == "amazon":
        if model["embedding_purpose"]:
            params = {"embeddingPurpose": model["embedding_purpose"], "embeddingDimension": dim,
                      "text": {"truncationMode": "NONE", "value": texts[0]}}
            return {"schemaVersion": "nova-multimodal-embed-v1", "taskType": model["task"],
                    "singleEmbeddingParams": params}
        return dict(inputText=texts[0], dimensions=dim, normalize=model["normalize_embeddings"],
                    embeddingTypes=["float"])
    body = {"model": name, "input": texts}
    if provider == "openai":
        body["encoding_format"] = "float"
        if model["request_dimensions"]:
            body["dimensions"] = dim
    elif provider == "voyage":
        body.update(truncation=False, output_dimension=dim, output_dtype="float")
    elif provider == "perplexity":
        body.update(dimensions=dim, encoding_format="base64_int8")
    elif provider == "naver":
        body.update(input=texts[0], encoding_format="float")
    return body


def hosted_batch(model, texts):
    provider = model["provider"]
    body = request_body(model, texts)
    if provider == "amazon":
        import boto3
        client = boto3.client("bedrock-runtime", region_name=model["region"])
        response = client.invoke_model(modelId=model["model_id"], body=json.dumps(body, ensure_ascii=False),
                                       accept="application/json", contentType="application/json")
        result = json.load(response["body"])
        if model["embedding_purpose"]:
            return [result["embeddings"][0]["embedding"]]
        return [result["embedding"]]
    import httpx
    key = os.environ[model["key_variable"]]
    headers = {"x-goog-api-key": key} if provider == "google" else {"Authorization": "Bearer " + key}
    response = httpx.post(model["endpoint"], headers=headers, json=body, timeout=120)
    response.raise_for_status()
    result = response.json()
    if provider == "google":
        return [e["values"] for e in result["embeddings"]]
    if provider == "cohere":
        return result["embeddings"]["float"]
    vectors = [e["embedding"] for e in sorted(result["data"], key=lambda e: e["index"])]
    if provider == "perplexity":
        return [np.frombuffer(base64.b64decode(v), dtype=np.int8) for v in vectors]
    return vectors


def sentence_encoder(model, device, dtype):
    from sentence_transformers import SentenceTransformer, models
    name, revision = model["model_id"], model["revision"]
    shared = dict(revision=revision, trust_remote_code=model["trust_remote_code"])
    weights = {"torch_dtype" if model["trust_remote_code"] else "dtype": dtype}
    config = dict(model.get("config_kwargs", {}))
    if model["code_revision"]:
        weights["code_revision"] = config["code_revision"] = model["code_revision"]
    if "task_adapter_layer" in model:
        from huggingface_hub import hf_hub_download
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        transformer = get_class_from_dynamic_module(
            model["task_adapter_layer"], name, revision=revision, code_revision=revision
        )
        layer = transformer(
            name, max_seq_length=model["maximum_input_length"], model_args={**shared, **weights},
            config_args={**shared, **config}, tokenizer_args=shared,
        )
        path = hf_hub_download(name, "1_Pooling/config.json", revision=revision)
        pooler = models.Pooling.load(str(Path(path).parent))
        modules = OrderedDict(transformer=layer, pooler=pooler, normalizer=models.Normalize())
        encoder = SentenceTransformer(modules=modules, device=device)
        encoder.module_kwargs = {"transformer": ["task"]}
    elif model.get("explicit_pooling"):
        checkpoint = {"revision": revision}
        layer = models.Transformer(
            name, max_seq_length=model["maximum_input_length"], model_args={**checkpoint, **weights},
            tokenizer_args=checkpoint, config_args=checkpoint,
        )
        pooler = models.Pooling(model["dimension"], pooling_mode=model["pooling"])
        encoder = SentenceTransformer(modules=[layer, pooler], device=device)
    else:
        extra = {}
        if "attention_backend" in model:
            weights["attn_implementation"] = model["attention_backend"]
        if "padding_side" in model:
            extra["processor_kwargs"] = {"padding_side": model["padding_side"]}
        encoder = SentenceTransformer(name, device=device, model_kwargs=weights,
                                      config_kwargs=config, **shared, **extra)
        if model.get("bidirectional_layers"):
            encoder[0].auto_model.config.use_cache = False
            for layer in encoder[0].auto_model.layers:
                layer.self_attn.is_causal = False
    encoder.max_seq_length = model["maximum_input_length"]
    return encoder.eval()


def float_vectors(output):
    import torch
    if isinstance(output, (list, tuple)) and isinstance(output[0], torch.Tensor):
        output = torch.stack(output)
    if isinstance(output, torch.Tensor):
        output = output.float().cpu().numpy()
    return np.asarray(output, dtype=np.float32)


def sentence_vectors(model, texts, device, dtype):
    size = model["batch_size"]
    encoder = sentence_encoder(model, device, dtype)
    prepared = [(model["prompt_text"] + t).strip() for t in texts]
    ids = encoder.tokenizer(prepared, truncation=False, padding=False)["input_ids"]
    if max(map(len, ids)) > model["maximum_input_length"]:
        raise ValueError("Speech exceeds the model input limit")
    if model["prompt_name"]:
        options = {"prompt_name": model["prompt_name"]}
    else:
        options = {"prompt": model["prompt_text"]}
    if model["task"]:
        options["task"] = model["task"]
    output = getattr(encoder, model["method"])(
        texts, batch_size=size, convert_to_tensor=True, show_progress_bar=False,
        normalize_embeddings=model["normalize_embeddings"], **options,
    )
    return float_vectors(output)


def transformer_vectors(model, texts, device, dtype):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    snapshot = snapshot_download(
        model["model_id"], revision=model["revision"],
        allow_patterns=["*.json", "*.py", "*.txt", "*.model", "*.tiktoken", "*.safetensors"],
        ignore_patterns=["onnx/*", "*/onnx/*"],
    )
    settings = dict(trust_remote_code=model["trust_remote_code"], local_files_only=True,
                    revision=model["revision"], code_revision=model["code_revision"])
    method, size, limit = model["method"], model["batch_size"], model["maximum_input_length"]
    precision = {"dtype" if method == "encode" else "torch_dtype": dtype}
    encoder = AutoModel.from_pretrained(snapshot, **settings, **precision)
    encoder = encoder.to(device=device, dtype=dtype).eval()
    if method == "last_token":
        tokenizer = AutoTokenizer.from_pretrained(snapshot, **settings)
    else:
        tokenizer = encoder.processor if method == "encode_text" else encoder.tokenizer
    vectors = []
    for start in range(0, len(texts), size):
        batch = texts[start:start + size]
        prepared = [model["prompt_text"] + text for text in batch]
        if method == "last_token":
            inputs = tokenizer(prepared, padding=True, truncation=False, return_tensors="pt").to(device)
            mask = inputs["attention_mask"]
            if mask.shape[1] > limit:
                raise ValueError("Speech exceeds the model input limit")
            hidden = encoder(**inputs, **model.get("forward_kwargs", {})).last_hidden_state
            positions = torch.arange(mask.shape[1], device=device).expand_as(mask)
            last = positions.masked_fill(mask == 0, -1).max(dim=1).values
            last_hidden = hidden[torch.arange(len(batch), device=device), last]
            output = torch.nn.functional.normalize(last_hidden, dim=1)
        else:
            ids = tokenizer(text=prepared, padding=False, truncation=False)["input_ids"]
            if max(map(len, ids)) > limit:
                raise ValueError("Speech exceeds the model input limit")
            options = dict(task=model["task"], prompt_name=model["prompt_name"],
                           max_length=limit, truncate_dim=None)
            if method == "encode_text":
                options.update(return_multivector=False, return_numpy=False, batch_size=size)
            output = getattr(encoder, method)(texts=batch, **options)
        vectors.append(float_vectors(output))
    return np.concatenate(vectors)


def embed(model, texts, device):
    size = model["batch_size"]
    if model["implementation"] == "api":
        batches = [hosted_batch(model, texts[i:i + size]) for i in range(0, len(texts), size)]
        return np.concatenate(batches).astype(model["storage_dtype"])
    import torch
    dtype = getattr(torch, model["inference_dtype"])
    with torch.inference_mode():
        if model["implementation"] == "transformers":
            vectors = transformer_vectors(model, texts, device, dtype)
        else:
            vectors = sentence_vectors(model, texts, device, dtype)
    if model["normalization_dtype"]:
        norms = np.linalg.norm(vectors.astype(model["normalization_dtype"]), axis=1, keepdims=True)
        vectors = (vectors / norms).astype(np.float32)
    return vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("participants", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("models", nargs="+", choices=MODELS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    data = pd.read_csv(args.participants, dtype={"sample_id": str})
    texts = data[["negative", "positive"]].to_numpy().ravel().tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    for key in args.models:
        vectors = embed(MODELS[key], texts, args.device)
        np.savez(args.output / (key + ".npz"), sample_id=data.sample_id.to_numpy(str),
                 negative=vectors[::2], positive=vectors[1::2])
