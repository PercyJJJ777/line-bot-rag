import base64
import hashlib
import hmac
import json
import unittest
from unittest.mock import Mock

from app import create_app, line_text


class WebhookTests(unittest.TestCase):
    def setUp(self):
        self.engine=Mock()
        self.engine.answer.return_value="ตอบจากเอกสาร [1]"
        self.reply=Mock()
        self.client=create_app(self.engine, self.reply, "test-secret").test_client()

    def post(self,payload,signature=True):
        body=payload if isinstance(payload,bytes) else json.dumps(payload,ensure_ascii=False).encode()
        headers={}
        if signature:
            headers['X-Line-Signature']=base64.b64encode(hmac.new(b'test-secret',body,hashlib.sha256).digest()).decode()
        return self.client.post('/webhook',data=body,headers=headers,content_type='application/json')

    def event(self,text="ASW5000-S-G2 ราคา"):
        return {'events':[{'type':'message','replyToken':'test-reply','message':{'type':'text','text':text}}]}

    def test_health_and_cron_endpoint_return_200(self):
        for path in ('/','/health'):
            response=self.client.get(path)
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json['dimension'],3072)
            self.assertEqual(response.json['namespace'],'rag-dt-7369758250fbe0306986')

    def test_unsigned_request_is_rejected(self):
        self.assertEqual(self.post(self.event(),False).status_code,400)
        self.engine.answer.assert_not_called()

    def test_tampered_signature_is_rejected(self):
        response=self.client.post('/webhook',data=b'{"events":[]}',headers={'X-Line-Signature':'fake'})
        self.assertEqual(response.status_code,400)

    def test_thai_utf8_signed_body_accepted(self):
        self.assertEqual(self.post(self.event()).status_code,200)
        self.reply.assert_called_once_with('test-reply','ตอบจากเอกสาร [1]')

    def test_line_verify_empty_events(self):
        self.assertEqual(self.post({'events':[]}).status_code,200)
        self.reply.assert_not_called()

    def test_signed_invalid_json_is_rejected(self):
        self.assertEqual(self.post(b'{invalid').status_code,400)

    def test_signed_wrong_event_schema_is_rejected(self):
        self.assertEqual(self.post({'events':'wrong'}).status_code,400)

    def test_non_text_event_is_ignored(self):
        self.assertEqual(self.post({'events':[{'type':'follow'}]}).status_code,200)
        self.engine.answer.assert_not_called()

    def test_provider_failure_sends_user_friendly_reply(self):
        self.engine.answer.side_effect=RuntimeError('private detail')
        self.assertEqual(self.post(self.event()).status_code,200)
        self.assertIn('ขัดข้อง',self.reply.call_args.args[1])
        self.assertNotIn('private detail',self.reply.call_args.args[1])

    def test_reply_failure_is_not_reported_as_success(self):
        self.reply.side_effect=RuntimeError('unavailable')
        self.assertEqual(self.post(self.event()).status_code,502)

    def test_utf16_message_limit(self):
        result=line_text('😀'*3000)
        self.assertEqual(len(result.encode('utf-16-le'))//2,4500)


if __name__=='__main__':
    unittest.main()
