import json, re, unittest
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs'
DATA=json.loads((DOC/'assets/case_data.json').read_text())
class ResearchSiteChecks(unittest.TestCase):
 def test_both_cases(self): self.assertEqual(set(DATA),{'CASE00001','CASE00002'})
 def test_four_phases(self):
  for k,d in DATA.items():self.assertEqual(set(d['phase_stats']),{'DRY','ART','VEN','DEL'})
 def test_slices_available(self):
  for k,d in DATA.items():
   self.assertEqual(len(d['slices']),21)
   for p in ['DRY','ART','VEN','DEL']:
    for i in range(21):self.assertTrue((DOC/f'assets/cases/{k}/{p}_{i:02}.webp').is_file())
 def test_annotations_by_case(self):
  self.assertEqual(set(DATA['CASE00001']['annotation_keys']), {'lesion','aphe','venous_washout','delayed_washout'})
  self.assertEqual(DATA['CASE00002']['annotation_keys'],['lesion'])
 def test_mean_hu_order(self):
  for k,d in DATA.items():self.assertTrue(d['phase_stats']['VEN']['lesion_mean_hu']>d['phase_stats']['DRY']['lesion_mean_hu'])
 def test_three_overlay_types(self):
  for case,d in DATA.items():
   for p in ['ART','VEN','DEL']:
    path=DOC/f'assets/cases/{case}/{p}_heat_10.png'
    self.assertTrue(path.exists())
    with Image.open(path) as im:self.assertEqual(im.mode,'RGBA')
 def test_js_mirror(self):
  t=(DOC/'assets/case_data.js').read_text();self.assertTrue(t.startswith('window.PHORA_CASES = {'));self.assertIn('CASE00002',t)
 def test_html_separation_of_claims(self):
  t=(DOC/'index.html').read_text()
  self.assertIn('SIGNED ENHANCEMENT',t);self.assertIn('FROM PHASE TO DECISION',t)
  self.assertIn('complete four-phase PHORA model',t)
  self.assertIn('papers.miccai.org/miccai-2026-sat/paper/AMPLIFAI_007.pdf',t)
 def test_no_broken_first_slice(self):
  t=(DOC/'index.html').read_text();refs=re.findall(r'(?:src|href)="(assets/[^"#]+)"',t)
  for x in refs:self.assertTrue((DOC/x).exists(),x)
 def test_saved_validation_records_are_distinct_from_showcase(self):
  js=(DOC/'assets/site.js').read_text()
  self.assertIn("CASE00015:{truth:'LR-5'",js)
  self.assertIn("CASE00025:{truth:'LR-5',pred:'LR-M'",js)
  self.assertNotIn("CASE00001:{truth",js)
  self.assertNotIn("CASE00002:{truth",js)
 def test_signed_maps(self):
  for k,d in DATA.items():
   self.assertEqual(set(d['signed_phase_stats']), {'ART','VEN','DEL'})
   for phase in ['ART','VEN','DEL']:
    for i in range(21):
     f=DOC/f'assets/cases/{k}/{phase}_signed_{i:02}.png'
     self.assertTrue(f.exists(),f)
     with Image.open(f) as im:self.assertEqual(im.mode,'RGBA')
 def test_saved_prediction_schema(self):
  js=(DOC/'assets/site.js').read_text()
  for k in ['CASE00015','CASE00025','CASE00035','CASE00084','CASE00099']:
   self.assertIn(k+':{truth:',js)
  self.assertIn('concepts:[',js)
  self.assertIn('burdens:[',js)
 def test_no_generated_labels_for_ct_examples(self):
  js=(DOC/'assets/site.js').read_text()
  for k in ['CASE00001','CASE00002']:
   self.assertNotIn(k+':{truth:',js)
 def test_fixed_liras_metric(self):
  qwk=.737732657614929;scr=.6052631578947368
  self.assertAlmostEqual(.85*qwk+.15*scr,.7178622326569002,delta=1e-10)
if __name__=='__main__':unittest.main()
