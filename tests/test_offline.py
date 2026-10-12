import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sse_rag.preprocess import normalize_inline_formulae,chunk_text
from scripts.aggregate_scores import aggregate,METRICS

class ScientificIntegrity(unittest.TestCase):
    def test_stoichiometry(self):
        source=r'$\mathrm { L i } _ { 1.3 } \mathrm { A l } _ { 0.3 } \mathrm { T i } _ { 1.7 }$'
        self.assertEqual(normalize_inline_formulae(source),'Li1.3Al0.3Ti1.7')
    def test_preserve_unsupported_notation(self):
        for text in [r'$\mathrm{Li}^{+}$',r'$\frac{x}{y}$',r'$x_{i}=1$',
                     r'$300\,^{\circ}\mathrm{C}$',r'$$x^{2}$$',r'$\mathrm{Xx}\mathrm{O}$']:
            self.assertEqual(normalize_inline_formulae(text),text)
    def test_no_gap_and_limit(self):
        text=('Some sentence. Another sentence.\n\n'*400)
        chunks=chunk_text(text,120,25)
        self.assertEqual(chunks[0].start,0)
        self.assertEqual(chunks[-1].end,len(text))
        for a,b in zip(chunks,chunks[1:]):
            self.assertEqual(b.start,a.end-25)
        self.assertTrue(all(c.content==text[c.start:c.end] and len(c.content)<=120 for c in chunks))
    def test_aggregate_rejects_duplicate_and_missing(self):
        row=dict(phase='P1',configuration='test',run_id='r1',question_id='q1',
                 **dict.fromkeys(METRICS,'0.5'))
        self.assertEqual(aggregate([row],1)[0]['score'],0.5)
        with self.assertRaises(ValueError): aggregate([row,row],1)
        with self.assertRaises(ValueError): aggregate([row],30)
        bad=dict(row,faithfulness='nan')
        with self.assertRaises(ValueError): aggregate([bad],1)
if __name__=='__main__': unittest.main()
