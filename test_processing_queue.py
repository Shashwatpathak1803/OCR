import tempfile
import unittest
from pathlib import Path

from processing_queue import ProcessingQueue, QueueWorker


class ProcessingQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.queue = ProcessingQueue(Path(self.temporary_directory.name))

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_hash_deduplication_and_persisted_completed_result(self):
        image_bytes = b"same image content"
        card, added = self.queue.add_image("first.jpg", image_bytes)
        self.assertTrue(added)

        duplicate, added = self.queue.add_image("renamed.png", image_bytes)
        self.assertFalse(added)
        self.assertEqual(duplicate["id"], card["id"])

        claimed = self.queue.claim_next()
        self.assertEqual(claimed["status"], "Processing")
        self.assertEqual(claimed["attempts"], 1)

        self.queue.complete(
            card["id"],
            "Completed",
            "Original OCR output",
            {"full_name_en": "A Person"},
            0.95,
            {"total_card_s": 1.25, "detection_s": 0.2},
        )

        restarted_queue = ProcessingQueue(Path(self.temporary_directory.name))
        records = restarted_queue.get_completed_records()
        self.assertEqual(records[0]["full_name_en"], "A Person")
        self.assertEqual(records[0]["_queue_id"], card["id"])
        self.assertEqual(restarted_queue.get_card(card["id"])["raw_text"], "Original OCR output")

        duplicate, added = restarted_queue.add_image("again.jpg", image_bytes)
        self.assertFalse(added)
        self.assertEqual(duplicate["status"], "Completed")
        self.assertEqual(restarted_queue.summary()["completed"], 1)

    def test_interrupted_and_failed_cards_can_be_recovered_and_retried(self):
        card, _ = self.queue.add_image("card.jpg", b"another image")
        self.queue.claim_next()
        self.assertEqual(self.queue.recover_interrupted(), 1)
        self.assertEqual(self.queue.get_card(card["id"])["status"], "Pending")

        claimed = self.queue.claim_next()
        self.queue.fail(claimed["id"], "OCR failed", {"ocr_s": 0.4})
        self.assertEqual(self.queue.retry_failed(), 1)
        self.assertEqual(self.queue.get_card(card["id"])["status"], "Pending")
        self.assertEqual(self.queue.claim_next()["attempts"], 3)

    def test_record_edits_are_saved_by_queue_id(self):
        card, _ = self.queue.add_image("editable.jpg", b"editable image")
        self.queue.claim_next()
        self.queue.complete(
            card["id"], "Awaiting Verification", "", {"city_en": "Delhi"}, 0.8, {}
        )

        self.queue.save_record(
            card["id"], {"_queue_id": card["id"], "city_en": "New Delhi"}
        )
        self.assertEqual(
            self.queue.get_completed_records()[0]["city_en"], "New Delhi"
        )

    def test_bulk_staging_keeps_one_database_row_per_unique_image(self):
        for index in range(1000):
            _, added = self.queue.add_image(
                f"card-{index}.jpg", f"unique image {index}".encode("ascii")
            )
            self.assertTrue(added)

        summary = self.queue.summary()
        self.assertEqual(summary["total"], 1000)
        self.assertEqual(summary["counts"]["Pending"], 1000)
        self.assertEqual(self.queue.claim_next()["attempts"], 1)

    def test_clearing_queue_keeps_completed_records_but_removes_staged_images(self):
        complete, _ = self.queue.add_image("saved.jpg", b"saved")
        self.queue.claim_next()
        self.queue.complete(
            complete["id"], "Completed", "saved OCR", {"full_name_en": "Saved"}, 0.9, {}
        )
        pending, _ = self.queue.add_image("pending.jpg", b"pending")
        pending_path = Path(pending["image_path"])

        self.queue.clear_queue()

        self.assertFalse(pending_path.exists())
        self.assertEqual(self.queue.summary()["total"], 1)
        self.assertEqual(self.queue.get_completed_records()[0]["full_name_en"], "Saved")
        self.assertIsNone(self.queue.get_image_path(complete["id"]))

    def test_idle_worker_marks_queue_paused_after_draining(self):
        worker = QueueWorker(self.queue)
        worker.start()
        thread = worker._thread
        assert thread is not None
        thread.join(timeout=5)

        self.assertFalse(worker.is_running())
        self.assertTrue(self.queue.is_paused())


if __name__ == "__main__":
    unittest.main()
