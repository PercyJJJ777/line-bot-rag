import unittest
from unittest.mock import Mock
from rag_core import build_catalog,identify_models,RagEngine

class NewProductRetrievalTests(unittest.TestCase):
    def setUp(self):
        names=['SOL7.4H','SOL7.4H-WP','SOL22H-WS','AMCP500-600T1','AMCP1200-1500','AMCL-D2','ADCU-M0']
        self.catalog=build_catalog([],{m:[m] for m in names})
        self.search=Mock();self.complete=Mock(return_value='ตอบตามคู่มือ [1]')
        self.engine=RagEngine(Mock(return_value=[.1]*3072),self.search,self.complete,self.catalog)

    def test_ev_suffix_wins_over_family_alias(self):
        self.assertEqual(identify_models('สเปก SOL7.4H-WP',self.catalog),(['SOL7.4H-WP'],False))

    def test_unknown_t1_variant_does_not_substitute_t2(self):
        self.engine.answer('สเปก AMCP1200-1500T1')
        self.search.assert_not_called()

    def test_t1_query_rejects_t2_document(self):
        self.search.return_value=[{'score':.9,'doc_type':'datasheet','models':['AMCP1200-1500'],
            'text':'Rated power 1500W, maximum input power 1800W and MPPT range 24-130V.', 'source':'T2.pdf','page':2}]
        self.engine.answer('สเปก AMCP500-600T1')
        self.complete.assert_not_called()

    def test_product_app_question_uses_manual_and_exact_model(self):
        self.search.return_value=[{'score':.9,'doc_type':'installation_manual','models':['SOL7.4H-WP'],
            'text':'Open the Solplanet app and configure Bluetooth and Wi-Fi for this EV charger.',
            'models_verified':True,'source':'QG0039.pdf','page':1}]
        answer=self.engine.answer('SOL7.4H-WP ใช้แอปอะไร')
        self.assertIn('QG0039.pdf หน้า PDF 1',answer)
        self.assertEqual(self.search.call_args.args[1]['$and'][0],{'models':{'$in':['SOL7.4H-WP']}})
        self.assertNotIn('training',self.search.call_args.args[1]['$and'][1]['doc_type']['$in'])

    def test_scanned_unreviewed_numeric_specs_remain_blocked(self):
        self.search.return_value=[{'score':.9,'doc_type':'datasheet','models':['AMCL-D2'],
            'text':'OCR maximum current may be 15A 20A 25A in adjacent table columns.',
            'requires_visual_review':True,'source':'scan.pdf','page':2}]
        self.engine.answer('สเปก AMCL-D2 กระแสเท่าไร')
        self.complete.assert_not_called()

if __name__=='__main__':unittest.main()
