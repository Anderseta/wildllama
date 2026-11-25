import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import worker


class VoiceSessionTest(unittest.TestCase):
    def test_preview_read_and_video_share_one_engine_until_eof(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = Mock()
            engine.model.device = "cuda:0"
            lines = []
            for i, stage in enumerate(("preview", "preview", "read", "synth")):
                job = root / str(i)
                job.mkdir()
                (job / "request.json").write_text(json.dumps({"engine": "qwen"}))
                lines.append(json.dumps({"stage": stage, "dir": str(job)}))
            with (patch.object(worker.sys, "stdin", io.StringIO("\n".join(lines))),
                  patch.object(worker, "make_voice", return_value=engine) as create,
                  patch.object(worker, "synth") as synth,
                  patch.object(worker, "read_aloud") as read,
                  patch("sys.stdout", new_callable=io.StringIO) as output):
                worker.voice_session(root, "tts")
            create.assert_called_once_with(root, False)
            self.assertEqual([call.kwargs["engine"] for call in synth.call_args_list], [engine] * 3)
            self.assertIs(read.call_args.args[-1], engine)
            self.assertEqual(output.getvalue().count('@voice-done {"device": "cuda:0"}'), 4)


if __name__ == "__main__":
    unittest.main()
