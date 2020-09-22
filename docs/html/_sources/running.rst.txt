.. _running:
Running COMET
==============

Command Line Interface
################################

After installing COMET you can score you MT outputs with the following command:: 

   comet score -s sources.txt -h hypothesis.txt -r references.txt --model wmt-large-da-estimator-1719

You can export your results to a JSON file using the `---to_json` flag::

   comet score -s sources.txt -h hypothesis.txt -r references.txt --model wmt-large-da-estimator-1719 --to_json output.json


Using Python
#############

Instead of using CLI you can score you models using Python with the `predict` function::

   from comet.models import download_model
   model = download_model("wmt-large-da-estimator-1719", "path/where/to/save/models")
   data = [
       {
           "src": "Hello world!",
           "mt": "Oi mundo!",
           "ref": "Olá mundo!"
       },
       {
           "src": "This is a sample",
           "mt": "este é um exemplo",
           "ref": "isto é um exemplo!"
       }
   ]
   model.predict(data)

Scoring MT ouputs using lists::
   
   source = ["Hello world!", "This is a sample"]
   hypothesis = ["Oi mundo!", "este é um exemplo"]
   reference = ["Olá mundo!", "isto é um exemplo!"]
   data = {"src": source, "mt": hypothesis, "ref": reference}
   data = [dict(zip(data, t)) for t in zip(*data.values())]
   model.predict(data)