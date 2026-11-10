import json
from pathlib import Path
import unittest
from unittest.mock import Mock
from rag_core import RagEngine,build_catalog,identify_models

class TrainingScopeTests(unittest.TestCase):
    def setUp(self):
        p=Path(__file__).with_name('model_aliases.json')
        self.review={'model_aliases':json.loads(p.read_text(encoding='utf-8'))}
        self.catalog=build_catalog([],self.review['model_aliases'])
        self.embed=Mock(return_value=[0.1]*3072)
        self.search=Mock()
        self.complete=Mock(return_value='อธิบายตามสไลด์ Training [1]')
        self.engine=RagEngine(self.embed,self.search,self.complete,self.catalog)
        self.training={'score':0.9,'text':'Error 01 communication fails. This training overview requires confirmation against the applicable product manual.',
            'source':'TBSH.pdf','page':7,'doc_type':'training','models':[],
            'models_verified':False,'requires_model_confirmation':True,'applicability':'general_topic',
            'mentioned_models':[],'topics':['troubleshooting']}

    def test_training_error_can_be_explained_with_scope_and_page(self):
        self.search.return_value=[self.training]
        answer=self.engine.answer('Training อธิบาย Error E01')
        self.assertIn('TBSH.pdf หน้า PDF 7',answer)
        self.assertIn('ต้องยืนยันรุ่นเพิ่มเติม: True',self.complete.call_args.args[1])
        self.assertIn('ห้ามอ้างว่าสเปก',self.complete.call_args.args[0])

    def test_training_does_not_prove_a_models_specification(self):
        self.search.return_value=[self.training]
        self.engine.answer('ASW5000-S-G2 แรงดันเท่าไร')
        self.complete.assert_not_called()

    def test_general_error_code_does_not_claim_a_product_model(self):
        self.engine.answer('Error E01 แก้อย่างไร')
        self.embed.assert_not_called()

    def test_ocr_technical_values_require_review(self):
        doc={**self.training,'doc_type':'datasheet','models':['ASW5000-S-G2'],'models_verified':True,'requires_visual_review':True}
        self.search.return_value=[doc]
        self.engine.answer('ASW5000-S-G2 แรงดันเท่าไร')
        self.complete.assert_not_called()

    def test_monitoring_topic_filters_training(self):
        self.search.return_value=[{**self.training,'topics':['monitoring']}]
        self.engine.answer('Solplanet monitoring มีอะไรบ้าง')
        filters=self.search.call_args.args[1]
        self.assertIn({'topics':{'$in':['monitoring']}},filters['$and'])
        self.complete.assert_called_once()

    def test_sales_aliases_resolve_to_document_models(self):
        for query,canonical in [('ASW10K-LT-G2 Pro','ASW10K-LT-G2-Pro'),('ASW30K-TH','ASW030K-TH'),('ASW010kH-T2','ASW10kH-T2'),('ASW5120M-HB-E-IP20','ASW5120M-HB-E')]:
            models,unknown=identify_models(query,self.catalog)
            self.assertFalse(unknown,query)
            self.assertEqual(models,[canonical],query)

    def test_known_pro_does_not_hide_an_unknown_model(self):
        models,unknown=identify_models('ASW10K-LT-G2 Pro กับ ASW9999-S-G9',self.catalog)
        self.assertTrue(unknown)
        self.assertIn('ASW9999-S-G9',models)

if __name__=='__main__': unittest.main()
