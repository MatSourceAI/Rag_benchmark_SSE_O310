"""Record installed versions without reading environment variables or secrets."""
import importlib.metadata
import json
import platform
from pathlib import Path

names = ['google-genai','openai','pymilvus','requests','python-dotenv','ragas',
         'mineru','transformers','sentence-transformers','torch']
versions = {}
for name in names:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
out = Path(__file__).resolve().parents[1]/'outputs'
out.mkdir(exist_ok=True)
record = {'python':platform.python_version(), 'system':platform.platform(),
          'packages':versions,
          'status':'current_machine_only_not_historical_experiment_environment'}
(out/'environment_report.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
