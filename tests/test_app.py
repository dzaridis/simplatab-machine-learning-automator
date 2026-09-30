import io
import unittest

import app as appmod
from werkzeug.test import Client


class TestUploadAndParameters(unittest.TestCase):
    def setUp(self):
        self.client = Client(appmod.application)

    def upload(self, train, test, train_name="Train.csv"):
        return self.client.post("/automl/upload", content_type="multipart/form-data", data={
            "train_file": (io.BytesIO(train.encode()), train_name),
            "test_file": (io.BytesIO(test.encode()), "Test.csv"),
        })

    def test_invalid_upload_shows_its_message(self):
        response = self.upload("a,Target\n1,0\n", "a,Target\n1,0\n", train_name="Train.txt")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/automl/")
        page = self.client.get("/automl/").get_data(as_text=True)
        self.assertIn("Invalid file type", page)

    def test_parameters_warns_about_target_labels(self):
        self.upload("a,Target\n1,1\n2,2\n3,1\n", "a,Target\n1,1\n")
        page = self.client.get("/automl/parameters").get_data(as_text=True)
        self.assertIn("Check the Target column", page)

        self.upload("a,Target\n1,0\n2,1\n3,0\n", "a,Target\n1,1\n")
        page = self.client.get("/automl/parameters").get_data(as_text=True)
        self.assertNotIn("Check the Target column", page)


if __name__ == "__main__":
    unittest.main()
