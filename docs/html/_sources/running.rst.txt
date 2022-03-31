.. _running:
Running COMET
==============

Command Line Interface
################################

Our CLI supports 4 different commands:

- ``comet-score``: the `Scoring Command`_ is used to evaluate MT. 
- ``comet-compare``: the `Compare Command`_ is used too compare two MT systems using statistical significance tests.
- ``comet-mbr``: the `MBR Command`_  is used for Minimum Bayes Risk Decoding.
- ``comet-train``: used to `train your own evaluation Metric <https://unbabel.github.io/COMET/html/training.html>`_.

Before we get started please create the following *dummy test data*::

   echo -e "Dem Feuer konnte Einhalt geboten werden\nSchulen und Kindergärten wurden eröffnet." >> src.de
   echo -e "The fire could be stopped\nSchools and kindergartens were open" >> hyp1.en
   echo -e "The fire could have been stopped\nSchools and pre-school were open" >> hyp2.en
   echo -e "They were able to control the fire.\nSchools and kindergartens opened" >> ref.en

Scoring Command
++++++++++++++++

Basic scoring command:: 

   comet-score -s src.de -t hyp1.en -r ref.en

use  ``--gpus 0`` to test on CPU.

Scoring multiple systems::
   
   comet-score -s src.de -t hyp1.en hyp2.en -r ref.en

You can also test your system on public benchmarks such as WMT20 en-de via `SacreBLEU
<https://github.com/mjpost/sacrebleu/>`_::

   comet-score -d wmt20:en-de -t PATH/TO/TRANSLATIONS

The default setting of ``comet-score`` prints the score for each segment individually. 
If you are only interested in the score for the whole dataset (computed as the average of the segment scores), 
you can use the ``--quiet`` flag. E.g::

   comet-score -s src.de -t hyp1.en -r ref.en --quiet

COMET provides a list of different model/metrics that you can use to evaluate your systems. You can select which one you want using the ``--model`` flag. 
 
**NOTE:** For reference-free (QE-as-a-metric) models you don't need to pass a reference. E.g: ::

   comet-score -s src.de -t hyp1.en --model wmt20-comet-qe-da

Following our work on `Uncertainty-Aware MT Evaluation
<https://aclanthology.org/2021.findings-emnlp.330/>`_ you can use the ``--mc_dropout`` flag to get a variance/uncertainty value for each segment score. 
If this value is high, it means that the metric is less confident in that prediction.::

   comet-score -s src.de -t hyp1.en -r ref.en --mc_dropout 30

**Multi-GPU Inference:**

COMET is optimized to be used in a single GPU by taking advantage of length batching and embedding caching. When using Multi-GPU since data e spread 
across GPUs we will typically get fewer cache hits and the length batching samples is replaced by a `DistributedSampler
<https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#replace-sampler-ddp/>`_. 
Because of that, according to our experiments, using 1 GPU is faster than using 2 GPUs specially when scoring multiple systems 
for the same source and reference.

Nonetheless, if your data does not have repetitions and you have more than 1 GPU available, you can **run multi-GPU inference with the following command**::

   comet-score -s src.de -t hyp1.en -r ref.en --gpus 2 --quiet


**Changing Embedding Cache Size:**
You can change the cache size of COMET using the following env variable::

   export COMET_EMBEDDINGS_CACHE="2048"

by default the COMET cache size is 1024.

Compare Command
++++++++++++++++

When comparing MT systems **we encourage you to check how significant are the improvements** you observe for a given model.
This command receives two system outputs and performs statistical significant testing with *Paired T-Test* and *bootstrap resampling*.::

   comet-compare -s src.de -x hyp1.en -y hyp2.en -r ref.en

MBR Command
+++++++++++

Minimum Bayes-Risk (MBR) decoding aims to find the candidate hypothesis that has the least expected loss under a given metric. 
Recent studies showed that MBR with neural fine-tuned metrics such as COMET leads to significant improvement in automatic and 
human evaluations [`Freitag et al. 2022, <https://arxiv.org/abs/2111.09388/>`_ `Zhang et al. 2022 <https://arxiv.org/abs/2203.00201/>`_].

We find that this is a promissing direction for MT in general and for that reason we developed the ``comet-mbr`` command. Our implementation is
inspired by `Amrhein et al, 2022, <https://arxiv.org/abs/2202.05148/>`_  caching approach.

Example:::

   comet-mbr -s source.txt -t samples.txt --num_samples n -o output.txt

where ``source.txt`` is a text file with one source per line and ``samples.txt`` is a text file with ``n`` samples for each ``source.txt`` line. 
The command will write the best MT sample to the ``output.txt`` file. 


Example with 2 source and 3 samples:

+------------+----------------------------------+
| source.txt |        samples.txt               |
+============+==================================+
| Obama      | Obama empfängt Netanjahu         |
| receives   +----------------------------------+ 
| Netanyahu  | Obama empfing Netanjahu          |
|            +----------------------------------+
|            | Obama trifft Netanjahu           |
+------------+----------------------------------+
| Lamb grew  | Lamm wuchs in der Gegend auf.    |
| up in the  +----------------------------------+ 
|            | Lamb wuchs in der Gegend auf.    |
| area.      +----------------------------------+
|            | Lamb wuchs in dieser Gegend auf. |
+------------+----------------------------------+
