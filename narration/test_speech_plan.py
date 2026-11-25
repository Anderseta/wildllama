import unittest
from types import SimpleNamespace as Mark
from speech_plan import speech_plan, caption_times, preview_chunk


class SpeechPlanTests(unittest.TestCase):
    def test_sentence_boundaries_and_pauses(self):
        script = "We saw Dr. Green. Another story begins right here.\n\nGood morning."
        chunks = speech_plan({"text": script}, limit=37)
        self.assertEqual([c["spoken"] for c in chunks], ["We saw Dr. Green.", "Another story begins right here.", "Good morning."])
        self.assertEqual([c["paragraph_end"] for c in chunks], [False, True, False])

    def test_preview_preserves_paragraph_and_pronunciation(self):
        cfg = {"text": "First paragraph.\n\nDr. Green uses SQL. And more.",
               "previewParagraph": 1, "pronunciations": [{"term": "SQL", "spoken": "sequel"}]}
        chunk = preview_chunk(cfg)
        self.assertEqual((chunk["paragraph"], chunk["spoken"]), (1, "Dr. Green uses sequel."))
        self.assertLessEqual(len(preview_chunk({"text": "a " * 500})["spoken"].split()), 24)

    def test_alias_preserves_original_caption_and_duration(self):
        cfg = {"text": "Use SQL now.", "pronunciations": [{"term": "SQL", "spoken": "S Q L"}]}
        chunk, = speech_plan(cfg)
        self.assertEqual(chunk["spoken"], "Use S Q L now.")
        marks = [Mark(text=s, start_time=i, end_time=i+1) for i,s in enumerate(["Use","S","Q","L","now"])]
        self.assertEqual(caption_times(chunk["units"], marks, 5, 2), [[2,3,"Use"],[3,6,"SQL"],[6,7,"now."]])
