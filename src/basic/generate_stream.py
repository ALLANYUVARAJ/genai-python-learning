import requests, json, sys

with requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3.2", "prompt": "Stream this sentence, please."},
    stream=True,           # default behavior: NDJSON streaming
    timeout=120,
) as r:
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        obj = json.loads(line)  # each line is a JSON object
        chunk = obj.get("response", "")
        if chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        if obj.get("done"):
            break
print()
