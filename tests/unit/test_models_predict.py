# -*- coding: utf-8 -*-
import os
import shutil
import unittest

import numpy as np
import torch

from comet import download_model, load_from_checkpoint
from comet.models import XCOMETMetric
from tests.data import DATA_PATH

TEST_SAMPLES = [
    {"lp": "it-en", "src": "Nel 1884, Tesla accettò un'offerta di lavoro presso la Edison Company di New York City e questo lo portò a trasferirsi negli Stati Uniti d’America.", "mt": "In 1884, Tesla accepted a job at the Pacific League of New York City and moved to the United States of America.", "ref": "In 1884, Tesla accepted a job with the Edison Company in New York City and moved to the United States of America.", "annotations": [{"start": 37, "end": 51, "text": "Pacific League", "category": "critical_id9_ne_replaced", "severity": "major"}], "score": 0.5833333333333333},
    {"lp": "it-en", "src": "Nel 1884, Tesla accettò un'offerta di lavoro presso la Edison Company di New York City e questo lo portò a trasferirsi negli Stati Uniti d’America.", "mt": "In 1884, Tesla accepted a job at the Edison Company of New York City and moved to the United States of America.", "ref": "In 1884, Tesla accepted a job with the Edison Company in New York City and moved to the United States of America.", "annotations": [], "score": 1.0},
    {"lp": "it-en", "src": "La Rivoluzione francese è stata una fonte di ispirazione anche per molti altri lavoratori oppressi di vari Paesi per iniziare la propria rivoluzione.", "mt": "The American Civil War was also an inspiration for many other oppressed workers from various countries to start their own revolution.", "ref": "The French Revolution also inspired many other repressed working class people of other country's to began their own revolutions.", "annotations": [{"text": "American Civil War", "start": 4, "end": 22, "category": "critical_id9_ne_replaced", "severity": "major"}], "score": 0.5454545454545454},
    {"lp": "it-en", "src": "La Rivoluzione francese è stata una fonte di ispirazione anche per molti altri lavoratori oppressi di vari Paesi per iniziare la propria rivoluzione.", "mt": "The French Revolution was also an inspiration for many other oppressed workers from various countries to start their own revolution.", "ref": "The French Revolution also inspired many other repressed working class people of other country's to began their own revolutions.", "annotations": [], "score": 1.0},
    {"lp": "it-en", "src": "Il Kundalini Yoga attiva l'energia Kundalini (energia dell'illuminazione) mediante l'impiego di posizioni yoga, esercizi di respirazione, mantra e visualizzazioni.", "mt": "Lokomotiv Moscow activates the Kundalini energy (energy of enlightenment) through the use of yoga postures, breathing exercises, mantras and visualizations.", "ref": "With Kundalini Yoga the Kundalini energy (enlightenment energy) is awakened through yoga postures, breathing exercises, mantras and visualizations.", "annotations": [{"start": 0, "end": 16, "text": "Lokomotiv Moscow", "category": "critical_id9_ne_replaced", "severity": "major"}], "score": 0.6},
    {"lp": "it-en", "src": "Il Kundalini Yoga attiva l'energia Kundalini (energia dell'illuminazione) mediante l'impiego di posizioni yoga, esercizi di respirazione, mantra e visualizzazioni.", "mt": "Kundalini Yoga activates the Kundalini energy (energy of enlightenment) through the use of yoga postures, breathing exercises, mantras and visualizations.", "ref": "With Kundalini Yoga the Kundalini energy (enlightenment energy) is awakened through yoga postures, breathing exercises, mantras and visualizations.", "annotations": [], "score": 1.0},
    {"lp": "it-en", "src": "Le isole dell'Africa orientale sono situate nell'Oceano Indiano, al largo della costa est dell'Africa.", "mt": "The East African islands are located in the Longxi River, off the east coast of Africa.", "ref": "The East African Islands are in the Indian Ocean off the eastern coast of Africa.", "annotations": [{"start": 44, "end": 56, "text": "Longxi River", "category": "critical_id9_ne_replaced", "severity": "major"}], "score": 0.4444444444444444},
    {"lp": "it-en", "src": "Le isole dell'Africa orientale sono situate nell'Oceano Indiano, al largo della costa est dell'Africa.", "mt": "The East African islands are located in the Indian Ocean, off the east coast of Africa.", "ref": "The East African Islands are in the Indian Ocean off the eastern coast of Africa.", "annotations": [], "score": 1.0},
    {"lp": "it-en", "src": "Le Falkland hanno la loro moneta ufficiale, la sterlina delle Falkland (FKP), il cui valore equivale a quello della sterlina britannica (GBP).", "mt": "The Falklands have their official currency, the Falklands Pound (FKP), the value of which is equivalent to that of American Dollar (GBP).", "ref": "The official Falklands currency is the Falkland pound (FKP) whose value is set equivalent to that of one British pound (GBP).", "annotations": [{"text": "of American Dollar", "start": 112, "end": 130, "category": "critical_id9_ne_replaced", "severity": "major"}], "score": 0.6551724137931034},
    {"lp": "it-en", "src": "Le Falkland hanno la loro moneta ufficiale, la sterlina delle Falkland (FKP), il cui valore equivale a quello della sterlina britannica (GBP).", "mt": "The Falklands have their official currency, the Falklands Pound (FKP), the value of which is equivalent to that of the British Pound (GBP).", "ref": "The official Falklands currency is the Falkland pound (FKP) whose value is set equivalent to that of one British pound (GBP).", "annotations": [], "score": 1.0},
    {"lp": "it-en", "src": "Nel suo uso popolare, il termine safari fa riferimento a viaggi svolti via terra, in particolare nella savana, per vedere la bellissima fauna selvatica africana.", "mt": "In its popular usage, the term safari refers to trips made by land, particularly in the savannah, to see beautiful Indonesian wildlife.", "ref": "The term safari in popular use refers to overland travel to view the stunning African wildlife, particularly on savanna.", "annotations": [{"text": "Indonesian", "start": 115, "end": 125, "category": "critical_id9_ne_replaced", "severity": "major"}], "score": 0.6153846153846154},
    {"lp": "it-en", "src": "Nel suo uso popolare, il termine safari fa riferimento a viaggi svolti via terra, in particolare nella savana, per vedere la bellissima fauna selvatica africana.", "mt": "In its popular usage, the term safari refers to trips made by land, particularly in the savannah, to see beautiful African wildlife.", "ref": "The term safari in popular use refers to overland travel to view the stunning African wildlife, particularly on savanna.", "annotations": [], "score": 1.0}
]

CONTEXT_TEST_SAMPLES = [
    {"lp": "it-en", "context_src": None,  "src": "Le isole dell'Africa orientale sono situate nell'Oceano Indiano, al largo della costa est dell'Africa.", "context_mt": None,  "mt": "The East African islands are located in the Indian Ocean, off the east coast of Africa.", "context_ref": None,  "ref": "The East African Islands are in the Indian Ocean off the eastern coast of Africa.", "annotations": [], "score": 1.0},
]

class TestUnifiedMetricPredict(unittest.TestCase):
   
    model = load_from_checkpoint(download_model("Unbabel/test-model-whimsical-whisper", saving_directory=DATA_PATH))
    gpus = 1 if torch.cuda.device_count() > 0 else 0
    
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(DATA_PATH, "models--Unbabel--test-model-whimsical-whisper"))

    def test_predict(self):
        model_output = self.model.predict(TEST_SAMPLES, batch_size=12, gpus=self.gpus)
        assert "error_spans" in model_output.metadata
        assert "src_scores" in model_output.metadata
        assert "ref_scores" in model_output.metadata
        assert "unified_scores" in model_output.metadata
        
        expected_scores = np.array(
            [model_output.metadata.src_scores, model_output.metadata.ref_scores, model_output.metadata.unified_scores]
        ).mean(axis=0)
        
        # Assert for almost equal Arrays or Numbers
        np.testing.assert_almost_equal(expected_scores, np.array(model_output.scores), decimal=5)
        np.testing.assert_almost_equal(model_output.system_score, expected_scores.mean(), 5)

    def test_context_predict(self):
        self.model.enable_context()
        assert self.model.use_context == False
    
    def test_length_batching(self):
        output_without_length_batching = self.model.predict(TEST_SAMPLES, batch_size=1, gpus=self.gpus, length_batching=False)
        output_with_length_batching = self.model.predict(TEST_SAMPLES, batch_size=1, gpus=self.gpus, length_batching=True)
        self.assertListEqual(output_without_length_batching.scores, output_with_length_batching.scores)
    
    def test_xcomet_predict(self):
        model = XCOMETMetric.load_from_checkpoint(
            checkpoint_path=download_model("Unbabel/test-model-whimsical-whisper", saving_directory=DATA_PATH),
            map_location=torch.device("cpu"),
            strict=False,
            **dict(self.model.hparams),
        )
        model.score_weights = [0.25, 0.25, 0.25, 0.25]
        model_output = model.predict(TEST_SAMPLES, batch_size=12, gpus=self.gpus)
        assert "mqm_scores" in model_output.metadata
        
        # on XCOMET we cap all scores at 1. and final score is a weighted average of 4 features.
        expected_scores = np.array([
            np.array(list(map(lambda x: 1.0 if x > 1.0 else x, model_output.metadata.src_scores))),
            np.array(list(map(lambda x: 1.0 if x > 1.0 else x, model_output.metadata.ref_scores))),
            np.array(list(map(lambda x: 1.0 if x > 1.0 else x, model_output.metadata.unified_scores))),
            model_output.metadata.mqm_scores
        ]).mean(axis=0)
        np.testing.assert_almost_equal(expected_scores, np.array(model_output.scores), decimal=5)

        # Put all the weight on MQM score.
        model.score_weights = [0, 0, 0, 1]
        model_output = model.predict(TEST_SAMPLES, batch_size=12, gpus=self.gpus)
        self.assertListEqual(model_output.scores, model_output.metadata.mqm_scores)


class TestRegressionMetricPredict(unittest.TestCase):
   
    model = load_from_checkpoint(download_model("Unbabel/eamt22-cometinho-da", saving_directory=DATA_PATH))
    gpus = 1 if torch.cuda.device_count() > 0 else 0
      
    def test_context_predict(self):
        # Enabling context should not change scores"
        model_output_context_disabled = self.model.predict(CONTEXT_TEST_SAMPLES, batch_size=2, gpus=self.gpus)
        self.model.enable_context()
        assert self.model.use_context == True
        model_output_context_enabled = self.model.predict(CONTEXT_TEST_SAMPLES, batch_size=2, gpus=self.gpus)
        np.testing.assert_almost_equal(np.array(model_output_context_disabled.scores), np.array(model_output_context_enabled.scores), decimal=5)
