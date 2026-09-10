"""Regression checks for annotation-only exceptions, with synthetic notebooks only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import validate_delivery as delivery


class AnnotationAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.original = {"nbformat": 4, "metadata": {}, "cells": [
            {"cell_type": "markdown", "source": ["Original note"], "metadata": {}, "id": "m"},
            {"cell_type": "code", "source": ["value = 1"], "metadata": {}, "id": "c",
             "execution_count": 1, "outputs": [{"output_type": "stream", "text": ["1"]}]},
        ]}
        self.current = copy.deepcopy(self.original)
        self.current["cells"][0]["source"] = ["Translated note"]
        self.name = "provenance/original_code/example.ipynb"
        self.write_notebook()
        self.record = {"file": self.name, "change_kind": "markdown_translation",
                       "original_sha256": "original-hash-not-replaced",
                       "current_sha256": delivery.sha(self.root / self.name),
                       "unchanged_content_sha256": delivery.notebook_digest(
                           self.original, exclude_markdown_source=True)}

    def write_notebook(self):
        path = self.root / self.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.current))

    def validate(self, records=None):
        path = self.root / "provenance/notebook_annotation_edits.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": 1,
                                    "records": records if records is not None else [self.record]}))
        with patch.object(delivery, "ROOT", self.root):
            return delivery.validate_annotation_edits()

    def test_reviewed_translation_passes(self):
        self.assertEqual(set(self.validate()), {self.name})

    def test_unreviewed_edit_fails_byte_check(self):
        self.current["cells"][0]["source"] = ["Another edit"]
        self.write_notebook()
        with self.assertRaisesRegex(RuntimeError, "Reviewed annotation file changed"):
            self.validate()

    def test_code_output_metadata_or_order_change_fails_even_with_new_byte_hash(self):
        for kind in ["source", "outputs", "metadata", "order"]:
            with self.subTest(kind=kind):
                self.current = copy.deepcopy(self.original)
                if kind == "order":
                    self.current["cells"].reverse()
                else:
                    self.current["cells"][1][kind] = ["changed"]
                self.write_notebook()
                self.record["current_sha256"] = delivery.sha(self.root / self.name)
                with self.assertRaisesRegex(RuntimeError, "protected content changed"):
                    self.validate()

    def test_duplicate_record_fails(self):
        with self.assertRaisesRegex(RuntimeError, "Duplicate annotation record"):
            self.validate([self.record, self.record])

    def test_scientific_output_directory_cannot_get_annotation_exception(self):
        self.name = "outputs/example.ipynb"
        self.write_notebook()
        self.record["file"] = self.name
        with self.assertRaisesRegex(RuntimeError, "allowed directory"):
            self.validate()

    def test_serialization_only_checks_markdown_too(self):
        self.name = "notebooks/example.ipynb"
        self.current = copy.deepcopy(self.original)
        self.write_notebook()
        self.record.update(file=self.name, change_kind="json_serialization_only",
                           current_sha256=delivery.sha(self.root / self.name),
                           unchanged_content_sha256=delivery.notebook_digest(self.original))
        self.validate()
        self.current["cells"][0]["source"] = ["Not a serialization change"]
        self.write_notebook()
        self.record["current_sha256"] = delivery.sha(self.root / self.name)
        with self.assertRaisesRegex(RuntimeError, "protected content changed"):
            self.validate()

    def test_record_cannot_point_outside_root(self):
        self.record["file"] = "../outside.ipynb"
        with self.assertRaisesRegex(RuntimeError, "leaves the package"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
