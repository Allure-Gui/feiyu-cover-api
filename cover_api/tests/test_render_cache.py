import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cover_api import render_cache


class CacheIdentityTests(unittest.TestCase):
    def test_pipeline_imports_both_cache_keys(self):
        from cover_api import pipeline
        self.assertIs(pipeline.dry_cache_key, render_cache.dry_cache_key)
        self.assertIs(pipeline.final_cache_key, render_cache.final_cache_key)

    def setUp(self):
        self.settings = {"pitch": 0, "index_rate": 0.6, "f0_method": "rmvpe"}

    def dry_key(self, index_digest):
        return render_cache.dry_cache_key(
            "song", "vocals", "model", index_digest, "spin-v2", self.settings
        )

    def test_index_choice_and_in_place_replacement_invalidate_dry_cache(self):
        with TemporaryDirectory() as tmp:
            index_a = Path(tmp) / "voice.index"
            index_b = Path(tmp) / "other.index"
            index_a.write_bytes(b"index-a")
            index_b.write_bytes(b"index-b")
            index_a_path_digest = render_cache.file_digest(index_a)
            index_b_digest = render_cache.file_digest(index_b)
            selected_index = self.dry_key(index_a_path_digest)
            other_index = self.dry_key(index_b_digest)
            self.assertNotEqual(selected_index, other_index)

            index_a.write_bytes(b"index-a-replaced-in-place")
            replaced_digest = render_cache.file_digest(index_a)
            self.assertNotEqual(index_a_path_digest, replaced_digest)
            self.assertNotEqual(selected_index, self.dry_key(replaced_digest))

    def test_no_index_or_rvc_setting_change_invalidates_dry_cache(self):
        indexed = self.dry_key("index-content")
        without_index = self.dry_key("no-index")
        self.assertNotEqual(indexed, without_index)
        self.settings["pitch"] = 3
        self.assertNotEqual(indexed, self.dry_key("index-content"))

    def test_mix_change_keeps_dry_key_but_changes_final_key(self):
        dry = self.dry_key("index-content")
        mix_a = {"eq_on": True, "presence_db": 2.0}
        mix_b = {"eq_on": False, "presence_db": 2.0}
        dry_again = self.dry_key("index-content")
        self.assertEqual(dry, dry_again)
        self.assertNotEqual(
            render_cache.final_cache_key(dry, "instrumental", mix_a),
            render_cache.final_cache_key(dry_again, "instrumental", mix_b),
        )

    def test_atomic_audio_and_pitch_metadata_roundtrip(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp3"
            source.write_bytes(b"audio-data")
            entry = root / "cache" / "render.zip"
            meta = {"effective_pitch": 5, "inst_pitch": 5}
            render_cache.store_render(entry, source, meta)
            target = root / "restored.mp3"
            self.assertEqual(render_cache.restore_render(entry, target), meta)
            self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_invalid_partial_cache_is_a_miss(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            entry = Path(tmp) / "partial.zip"
            entry.write_bytes(b"partial")
            self.assertIsNone(render_cache.restore_render(entry, Path(tmp) / "out"))


if __name__ == "__main__":
    unittest.main()
