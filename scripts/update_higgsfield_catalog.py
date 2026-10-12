"""Rebuild app/higgsfield_catalog.json from docs.higgsfield.ai.

Each video workflow page publishes its endpoint ID and complete JSON schema;
the Generate tab builds its forms from this file. Re-run when Higgsfield adds
or changes models:  python scripts/update_higgsfield_catalog.py
"""
import json
import os
import re
import sys
import urllib.request

DOCS = "https://docs.higgsfield.ai"
OUT = os.path.join(os.path.dirname(__file__), "..", "app", "higgsfield_catalog.json")


def fetch(path):
    with urllib.request.urlopen(DOCS + path + ".md", timeout=30) as r:
        return r.read().decode()


def parse_workflow(md):
    endpoint = re.search(r"\*\*Endpoint ID:\*\* `([^`]+)`", md)
    title = re.search(r"^# (.+)$", md, re.M)
    block = re.search(r'<Accordion title="Complete JSON schema">\s*```json[^\n]*\n(.*?)```', md, re.S)
    if not (endpoint and title and block):
        return None
    schema = json.loads("\n".join(l[2:] if l.startswith("  ") else l for l in block.group(1).splitlines()))
    notes = re.search(r"## Usage notes\n(.*?)\n## ", md, re.S)
    notes = [re.sub(r"\\(.)", r"\1", l.strip()[2:]) for l in (notes.group(1).splitlines() if notes else [])
             if l.strip().startswith("* ")]
    return {"endpoint": endpoint.group(1), "title": title.group(1).strip().replace(" API", ""),
            "notes": notes, "schema": schema}


def main():
    index = fetch("/docs/models/video-generation")
    families = []
    for m in re.finditer(r'href="/docs/models/([a-z0-9-]+)">.*?featured-model-meta"><span>([^<]+)</span>'
                         r'.*?featured-model-title">([^<]+)<', index, re.S):
        families.append({"slug": m.group(1), "vendor": m.group(2), "name": m.group(3)})
    catalog = []
    for fam in families:
        page = fetch(f"/docs/models/{fam['slug']}")
        paths = sorted(set(re.findall(rf"/docs/models/{fam['slug']}/[a-z0-9-]+", page)))
        workflows = []
        for p in paths:
            wf = parse_workflow(fetch(p))
            if wf:
                wf["doc"] = DOCS + p
                workflows.append(wf)
            else:
                print("  skipped (no schema):", p, file=sys.stderr)
        print(f"{fam['name']}: {len(workflows)} workflows", file=sys.stderr)
        catalog.append({**fam, "workflows": workflows})
    with open(OUT, "w") as f:
        json.dump(catalog, f, indent=1)
    print(f"wrote {sum(len(f['workflows']) for f in catalog)} workflows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
