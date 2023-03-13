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
run the following command::

   comet-score -s src.de -t hyp1.en -r ref.en --quiet --only_system

COMET provides a list of different model/metrics that you can use to evaluate your systems. You can select which one you want using the ``--model`` flag. 
 
**NOTE:** For reference-free (QE-as-a-metric) models you don't need to pass a reference. E.g: ::

   comet-score -s src.de -t hyp1.en --model Unbabel/wmt20-comet-qe-da

Compare Command
++++++++++++++++

When comparing multiple MT systems we encourage you to run the ``comet-compare`` command to get **statistical significance** with Paired T-Test and bootstrap resampling.

   comet-compare -s src.de -t hyp1.en hyp2.en hyp3.en -r ref.en

MBR Command
++++++++++++++++

Minimum Bayes-Risk (MBR) decoding aims to find the candidate hypothesis that has the least expected loss under a given metric. 
Recent studies showed that MBR with neural fine-tuned metrics such as COMET leads to significant improvement in automatic and 
human evaluations [`Fernandes et al., NAACL 2022, <https://aclanthology.org/2022.naacl-main.100/>`_ `Zhang et al. 2022 <https://arxiv.org/abs/2203.00201/>`_].

Example::

   comet-mbr -s [SOURCE].txt -t [MT_SAMPLES].txt --num_sample [X] -o [OUTPUT_FILE].txt

If working with a very large candidate list you can use ``--rerank_top_k`` flag to prune the topK most promissing candidates according to a reference-free metric.

   comet-mbr -s [SOURCE].txt -t [MT_SAMPLES].txt -o [OUTPUT_FILE].txt --num_sample 1000 --rerank_top_k 100 --gpus 4 --qe_model Unbabel/wmt20-comet-qe-da

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
