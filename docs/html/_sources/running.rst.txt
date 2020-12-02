.. _running:
Running COMET
==============

Command Line Interface
################################

After installing COMET you can score you MT outputs with the following command:: 

   echo -e "Dem Feuer konnte Einhalt geboten werden\nSchulen und Kindergärten wurden eröffnet." >> src.de
   echo -e "The fire could be stopped\nSchools and kindergartens were open" >> hyp.en
   echo -e "They were able to control the fire.\nSchools and kindergartens opened" >> ref.en
   comet score -s src.de -h hyp.en -r ref.en

You can also specify a metric model using the `---model` flag (see :ref:`models:COMET Metrics` for available metric models):: 

   comet score -s src.de -h hyp.en -r ref.en --model wmt-large-da-estimator-1719

You can export your results to a JSON file using the `---to_json` flag::

   comet score -s src.de -h hyp.en -r ref.en --to_json output.json

Finally, if you don't have access to a GPU you can run COMET using CPU with the :code:`--cpu` flag but this will take some time to run.


Using Python
#############

Instead of using CLI you can also score your models in Python with the `predict` function::

   from comet.models import download_model
   model = download_model("wmt-large-da-estimator-1719", "path/where/to/save/models/")
   data = [
      {
         "src": "Dem Feuer konnte Einhalt geboten werden",
         "mt": "The fire could be stopped",
         "ref": "They were able to control the fire."
      },
      {
         "src": "Schulen und Kindergärten wurden eröffnet.",
         "mt": "Schools and kindergartens were open",
         "ref": "Schools and kindergartens opened"
      }
   ]
   model.predict(data, cuda=True, show_progress=True)

Scoring MT ouputs using lists::
   
   source = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
   hypothesis = ["The fire could be stopped", "Schools and kindergartens were open"]
   reference = ["They were able to control the fire.", "Schools and kindergartens opened"]
   data = {"src": source, "mt": hypothesis, "ref": reference}
   data = [dict(zip(data, t)) for t in zip(*data.values())]
   model.predict(data, cuda=True, show_progress=True)

**Note:** COMET corpus-level scores are obtained by averaging segment-level scores.