Ollama Benchmark
~~~~~~~~~~~~~~~~

ollama-benchmark is a handy tool to measure the performance and efficiency of LLMs workloads.

.. contents:: Table of Contents
   :depth: 3
   :local:

Get started
===========

Install
-------

Simple as::

  pip install https://github.com/cloudmercato/ollama-benchmark/archive/refs/heads/main.zip

For monitoring you may install `Probes`_::

  pip install https://github.com/cloudmercato/Probes/archive/refs/heads/main.zip

Usage
-----

ollama-benchmark deliver several workloads:

- ``speed``: Evaluate chat speed performance
- ``embedding``: Evaluate embedding peformance
- ``load``: Evaluate model loading speed
- ``judge``: Evaluate answer quality with LLM-as-a-Judge
- ``chat``: Live evaluate performance while chatting

Please keep in mind the ollama server configuration during evaluation of results. See `this part of the FAQ <https://github.com/ollama/ollama/blob/8b920f35a46c6459e0fd48daa38bc80963bf6462/docs/faq.md#how-does-ollama-handle-concurrent-requests>`_  for more understanding of Ollama's performance.

All the `common Ollama parameters <https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter>`_ can be configured through command line options.

speed
@@@@@

This tool allow to run a set of simultaneous requests to the server. The question set is mix of `FastChat's MT-Bench dataset <https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl>`_ and Cloud Mercato's samples allowing computer vision evaluation.

Example::

  $ ollama-benchmark speed --question 81 --model llama3 --max-workers 1 --max_turns 1
  version: 0.1
  model: llama3
  question_ids: ["81"]
  max_workers: 1
  max_turns: 1
  mirostat: 0
  mirostat_eta: 0.1
  ...
  prompt_eval_duration_mean: 161.571
  prompt_eval_duration_stdev: 0.0
  prompt_eval_rate_mean: 198.05534409021422  <-- Valuable
  prompt_eval_rate_stdev: 0.0
  eval_count_mean: 128
  eval_count_stdev: 0.0
  prompt_eval_count_mean: 32
  prompt_eval_count_stdev: 0.0
  eval_duration_mean: 3966.014
  eval_duration_stdev: 0.0
  eval_rate_mean: 32.27421789232211
  eval_rate_stdev: 0.0
  total_duration: 4166.39425  <-- Valuable
  real_duration: 4356.656789779663  <-- Valuable

embedding
@@@@@@@@@

Evaluate the duration of embedding through different scale of client, different size of input and languages.

Example::

  $ ollama-benchmark embedding --model llama3 --max-workers 1 --num-tasks 3 --langs jp en --sample-sizes 32 64
  version: 0.1
  model: llama3
  question_ids: ["81"]
  max_workers: 1
  max_turns: 1
  mirostat: 0
  mirostat_eta: 0.1
  ...
  duration_min: 0.3955111503601074
  duration_max: 1.2217307090759277
  duration_mean: 0.6712129910786947
  duration_stdev: 0.47676253481630143
  duration_perc95: 1.2217307090759277
  total_duration: 2.013638973236084
  real_duration: 2014.2037868499756
  rate_min: 0.8185109800148703
  rate_max: 2.5283737236978374
  rate_mean: 1.9565358035939044
  rate_stdev: 0.9855624575889667
  rate_perc95: 2.5283737236978374
  errors: 0
  errors_per_worker_mean: 0
  errors_per_worker_stdev: 0.0

load
@@@@

Evaluate the duration of loading one or several models into memory.

Example::

  $ ollama-benchmark --host zulumini:11434 load qwen:0.5b
  qwen:0.5b
  version: 0.1
  models: ["qwen:0.5b"]
  max_workers: 1
  duration_min: 0.5746748447418213
  duration_max: 0.5746748447418213
  duration_mean: 0.5746748447418213
  duration_stdev: 0.0
  duration_perc95: 0.5746748447418213
  total_duration: 0.5746748447418213
  real_duration: 0.6157209873199463
  rate_min: 1.7401144475868968
  rate_max: 1.7401144475868968
  rate_mean: 1.7401144475868968
  rate_stdev: 0.0
  rate_perc95: 1.7401144475868968
  errors: 0

judge
@@@@@

Use LLM-as-a-Judge technic to evaluate quality of given response.

Example::

  $ ollama-benchmark judge --question 81 --judge-model llama3 --model qwen:1.8b --max_turns 1
  version: 0.1
  model: qwen:1.8b
  judge_model: llama3
  question_id: 81
  max_turns: 2
  mirostat: 0
  mirostat_eta: 0.1
  ...
  judge_top_k: 40
  judge_top_p: 0.9
  judge_min_p: 0.0
  message_duration: 1.4621801376342773
  judge_duration: 14.956491947174072
  work_duration: 16.41867208480835
  total_rating_mean: 30
  total_rating_stdev: 0.0
  total_ratings: [30]
  0;evaluation: The answer provides a general overview of the state of Hawaii and mentions two must-see attractions, Waikiki Beach and Haleakala National Park. However, it lacks cultural experiences and details about the trip.
  0;feedback: To improve this response, I would suggest providing more specific examples of cultural experiences had during the trip, such as visiting local markets, trying traditional Hawaiian cuisine, or attending a luau. Additionally, including more vivid descriptions of the natural attractions mentioned could make the post more engaging.

chat
@@@@

Make a live chat in command line and get live performance data.

Example::

  $ ollama-benchmark chat
  load_model_duration:  6.159428119659424
  > Hello world
  < A classic!

  "Hello, World!" is a traditional greeting in programming, often used to test if a program is working correctly. It's a simple yet iconic phrase that has been a part of computer culture for decades.

  So, what brings you here today? Are you looking for help with a programming problem or just wanting to say hello? Either way, I'm happy to chat!
  total_duration:  3.52207325
  load_duration:  0.032622416
  prompt_eval_count:  12
  prompt_eval_duration:  1.094229
  eval_count:  78
  eval_duration:  2.393477
  request_duration:  3.6268999576568604
  > \q

Special command are available with the prefix ``\``, type ``\help`` to get more informations.

Monitoring
@@@@@@@@@@

ollama-benchmark includes a built-in monitoring tool running the time of each workloads. Use following option to control it:

- ``--monitoring-interval``: Define the interval between each probe
- ``--monitoring-probers``: Define probers as Python path (ie: `path.to.my.Prober`), see `Probes' documentation <https://github.com/cloudmercato/Probes/blob/main/README.rst>`_
- ``--monitoring-output``: Define path to the JSON output
- ``--disable-monitoring``: Completly disable monitoring

While we try to keep a minimal computational overhead, some probes may incur a duration during starting and stopping.

Common
@@@@@@

You can list questions with the following command::

  $ ollama-benchmark questions
  ID | Category | # Turns | Turns
  81 | writing  |   2 | ['Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.', 'Rewrite your previous response. Start every sentence with the letter A.']
  82 | writing  |   2 | ["Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn. Keep the email short and to the point.", 'Take a moment to evaluate and critique your own response.']
  83 | writing  |   2 | ['Imagine you are writing a blog post comparing two popular smartphone models. Develop an outline for the blog post, including key points and subheadings to effectively compare and contrast the features, performance, and user experience of the two models. Please answer in fewer than 200 words.', 'Take your previous response and rephrase it as a limerick.']
  84 | writing  |   2 | ['Write a persuasive email to convince your introverted friend, who dislikes public speaking, to volunteer as a guest speaker at a local event. Use compelling arguments and address potential objections. Please be concise.', 'Can you rephrase your previous answer and incorporate a metaphor or simile in each sentence?']
  85 | writing  |   2 | ['Describe a vivid and unique character, using strong imagery and creative language. Please answer in fewer than two paragraphs.', 'Revise your previous response and incorporate an allusion to a famous work of literature or historical event in each sentence.']
  ...

Just pulling models is also doable::

  ollama-benchmark pull_model llama3 phi3
                         
External links
--------------

ollama-benchmark has been used for the following evaluations:

- `Ollama benchmark Q2 2024 - Exoscale A40 <https://projector.cloud-mercato.com/projects/exoscale-a40-gpus>`_


Contribute
----------

This project is created with ❤️ for free by `Cloud Mercato`_ under BSD License. Feel free to contribute by submitting a pull request or an issue.

.. _`Probes`: https://github.com/cloudmercato/Probes
.. _`Cloud Mercato`: https://www.cloud-mercato.com/
