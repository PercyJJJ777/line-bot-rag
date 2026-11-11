import unittest
from unittest.mock import Mock
from rag_core import RagEngine, build_catalog, document_types, identify_models, meaningful


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.catalog = build_catalog([
            {"models": ["ASW5000-S-G2"]}, {"models": ["ASW5000-S"]},
            {"models": ["SUN2000-20KTL-MB0 (New Model) / ต่อแบตเตอรี่ LUNA S1 ได้"]},
            {"models": ["ASW010kH-T2"]},
        ])
        self.doc = {"score": .8, "models": ["ASW5000-S-G2"], "doc_type": "datasheet",
            "text": "Maximum input voltage is 600V. This model has two independent MPPT inputs.",
            "source": "Datasheet.pdf", "page": 2}
        self.embed = Mock(return_value=[.1]*3072)
        self.search = Mock(return_value=[self.doc])
        self.complete = Mock(return_value="แรงดันขาเข้าสูงสุด 600V [1]")
        self.engine = RagEngine(self.embed, self.search, self.complete, self.catalog)

    def test_exact_model_does_not_match_shorter_model(self):
        self.assertEqual(identify_models("ASW5000-S-G2", self.catalog), (["ASW5000-S-G2"], False))

    def test_filter_keeps_metadata_original_casing(self):
        self.assertEqual(identify_models("ASW010KH-T2", self.catalog), (["ASW010kH-T2"], False))

    def test_price_description_resolves_to_stored_metadata(self):
        models, unknown = identify_models("ราคา SUN2000-20KTL-MB0", self.catalog)
        self.assertFalse(unknown)
        self.assertIn("(New Model)", models[0])

    def test_unknown_model_never_falls_back_to_another_model(self):
        result = self.engine.answer("ราคา ASW5000-S-G3")
        self.assertIn("ยังไม่มีข้อมูล", result)
        self.embed.assert_not_called()

    def test_partial_model_asks_for_full_model(self):
        self.assertIn("ชื่อรุ่นเต็ม", self.engine.answer("ราคา ASW5000"))
        self.embed.assert_not_called()

    def test_manual_without_model_asks_first(self):
        self.assertIn("ชื่อรุ่นเต็ม", self.engine.answer("วิธีตั้งค่า error code"))
        self.embed.assert_not_called()

    def test_explicit_um_selects_user_manual_only(self):
        self.assertEqual(document_types("UM ASW5000-S-G2")[0], ["user_manual"])

    def test_installation_question_can_use_matching_user_manual(self):
        self.search.return_value = [{**self.doc, "doc_type":"user_manual", "source":"UM0023.pdf", "page":14,
            "text":"Mount the ASW5000-S-G2 inverter with the wall bracket and observe the minimum mounting clearances."}]
        answer = self.engine.answer("ติดตั้ง ASW5000-S-G2 อย่างไร")
        self.assertIn("UM0023.pdf หน้า PDF 14",answer)
        self.assertIn("user_manual",self.search.call_args.args[1]['$and'][1]['doc_type']['$in'])
        self.complete.assert_called_once()

    def test_error_code_is_not_mistaken_for_product_model(self):
        self.assertEqual(identify_models("ASW5000-S error E-33", self.catalog), (["ASW5000-S"], False))

    def test_spec_retrieval_has_filter_and_page_reference(self):
        answer = self.engine.answer("สเปก ASW5000-S-G2 แรงดันเท่าไร")
        self.assertEqual(self.search.call_args.args[1], {"$and":[{"models":{"$in":["ASW5000-S-G2"]}}, {"doc_type":{"$in":["datasheet"]}}]})
        self.assertIn("Datasheet.pdf หน้า PDF 2", answer)
        self.assertIn("600V", answer)

    def test_manual_of_other_model_is_rejected_even_if_adapter_ignores_filter(self):
        self.search.return_value = [{**self.doc, "models":["ASW5000-S"], "doc_type":"user_manual"}]
        self.assertIn("ตรงรุ่น", self.engine.answer("คู่มือใช้งาน ASW5000-S-G2"))
        self.complete.assert_not_called()

    def test_footer_only_page_is_rejected(self):
        self.assertFalse(meaningful("60 UM0009_ASW3000-5000-S_EN_V05_0223"))
        self.search.return_value=[{**self.doc,"text":"60 UM0009_ASW3000-5000-S_EN_V05_0223"}]
        self.engine.answer("สเปก ASW5000-S-G2")
        self.complete.assert_not_called()

    def test_low_similarity_does_not_call_llm(self):
        self.search.return_value=[{**self.doc,"score":.1}]
        self.engine.answer("สเปก ASW5000-S-G2")
        self.complete.assert_not_called()

    def test_existing_custom_reply_is_preserved(self):
        self.assertEqual(self.engine.answer("ใครหล่อที่สุด"), "คุณสเต๊ะ")
        self.embed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
