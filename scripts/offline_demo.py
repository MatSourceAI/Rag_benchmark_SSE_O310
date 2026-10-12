"""No private files, credentials, network calls, model scores, or API charges."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sse_rag.preprocess import process

def main():
    text = ('# Artificial demonstration\n\n'
            'Formatting example: $\\mathrm { L i } _ { 1.3 } '
            '\\mathrm { A l } _ { 0.3 } \\mathrm { T i } _ { 1.7 }$.\n\n'
            'This is invented text for checking program execution. It reports no '
            'experimental material, synthesis condition, or ionic conductivity.\n\n'
            'Unsupported math is kept: $x^{2}+y=1$.')
    result = process(text)
    result['data_status'] = 'artificial_software_fixture_not_benchmark'
    out = Path(__file__).resolve().parents[1] / 'outputs'
    out.mkdir(exist_ok=True)
    (out/'offline_demo.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result
if __name__ == '__main__':
    main()
