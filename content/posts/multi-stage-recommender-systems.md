+++
title = 'Multi-Stage Approach to Building Recommender Systems'
date = 2025-06-03T13:48:45+05:30
draft = false
+++

Multi-stage recommendation systems break down the challenging task of matching users with relevant items into several sequential phases, each optimizing for different objectives like efficiency, accuracy, and personalization. By progressively narrowing down a vast pool of candidates, applying increasingly complex models, and refining final rankings, these systems achieve scalable and high-quality recommendations even when dealing with billions of users and items ([ijcai.org][1], [developers.google.com][2]). They mirror how humans might sift through information: first skimming broadly, then considering details, and finally fine-tuning choices. This blog post explores the conceptual foundations of multi-stage recommendation, the distinct roles of each phase, the motivations behind layered architectures, and the real-world trade-offs they address. Along the way, analogies to everyday decision-making, historical parallels from human learning, and references to psychology illustrate how designers balance speed, relevance, and diversity. Finally, we survey challenges such as latency constraints, fairness, and the evolution toward neural re-ranking and hybrid objectives, pointing curious readers to key research papers and practical guides for deeper study.

## Introduction: A Personal Reflection on Systems of Thought

When I first encountered recommendation systems, I was struck by how they mirrored the way we navigate choices in daily life. Whether picking a movie on a streaming platform or selecting a restaurant in an unfamiliar city, we often start by skimming broad categories, then gradually focus on specific options, and finally make subtle refinements based on our mood or context. In my own journey—studying neural networks, building small-scale recommenders, and later reading about industrial-scale deployments—I realized that the most robust systems also follow a layered, multi-step process. Each stage builds on the previous one, balancing the need for speed with the quest for relevance.

Early in my learning, I faced the temptation to design a single, “perfect” model that could solve everything at once. But this naive approach quickly ran into practical barriers: datasets with millions of users and items, strict latency requirements, and the ever-present engineering constraints of limited compute. Over time, I discovered that breaking the problem into stages not only made systems more scalable but also allowed each subcomponent to focus on a clear objective—much like how one might draft a rough outline before writing a polished essay. This approach felt natural, almost human. It honors the way we refine our thinking: brainstorm broadly, narrow the field, then polish the final answer.

In this post, inspired by Andrej Karpathy’s calm, thoughtful narrative style, I want to share the conceptual palette of multi-stage recommendation systems. My aim is to offer clarity over complexity, distilling intricate algorithms into intuitive ideas and drawing parallels to broader human experiences. Whether you are a curious student, an engineer venturing into recommender research, or simply someone intrigued by how machines learn to predict our preferences, I hope this narrative resonates with your own learning journey.

## Understanding Multi-Stage Recommendation Systems

### The Core Idea: Divide and Conquer

At its simplest, a recommendation system tries to answer: “Given a user, which items will they find relevant?” When the number of potential items is enormous—often in the hundreds of millions—applying a single complex model to score every possible user-item pair quickly becomes infeasible. Multi-stage recommendation systems tackle this by splitting the problem into sequential phases, each with a different scope and computational budget ([ijcai.org][1], [developers.google.com][2]).

1. **Candidate Generation (Retrieval):** Reduce a massive corpus of items to a smaller, manageable subset—often from millions to thousands.
2. **Scoring (Ranking):** Use a more refined model to evaluate and rank these candidates, selecting a handful (e.g., 10–50) for final consideration.
3. **Re-Ranking (Refinement):** Apply an even richer model, possibly incorporating contextual signals, diversity constraints, or business rules, to order the final set optimally for display.

Some architectures include additional phases—such as pre-filtering by broad categories or post-processing for personalization and fairness—leading to four-stage or more elaborate pipelines ([resources.nvidia.com][3]). But the essential principle remains: start broad and coarse, then iteratively refine.

This cascade mirrors human decision-making. Imagine shopping online for a book: you might first browse top genres (candidate generation), then look at bestsellers within your chosen genre (scoring), and finally read reviews to pick the exact title (re-ranking). Each step focuses on a different level of granularity and uses different cues.

### Why Not a Single Model?

One might ask: why not build one powerful model that directly scores every item? In theory, a deep neural network with billions of parameters could capture all signals—user preferences, item attributes, temporal trends, social context. Yet in practice:

* **Computational Cost:** Scoring billions of items per user request is prohibitively expensive. Even if each prediction took a microsecond, processing a single query over 100 million items would take over a minute.
* **Latency Constraints:** Most user-facing systems must respond within tens to a few hundred milliseconds to maintain a fluid experience.
* **Scalability:** As user and item counts grow, retraining and serving a monolithic model becomes unwieldy, requiring massive hardware infrastructure.
* **Flexibility:** Separate stages allow engineers to swap, update, or A/B test individual components (e.g., try a new candidate generator) without rebuilding the entire system.

Thus, multi-stage pipelines offer a practical compromise: coarse but fast filtering followed by progressively more accurate but slower models, ensuring that latency stays within acceptable bounds while maintaining high recommendation quality ([ijcai.org][1], [developers.google.com][2]).

### Historical Context: From Heuristics to Neural Pipelines

Early recommenders—dating back to collaborative filtering in the mid-1990s—often endured all-to-all scoring within a manageable dataset size. But as platforms like Amazon, Netflix, and YouTube scaled to millions of users and items, engineers introduced multi-step processes. For instance, Netflix’s 2006 recommendation infrastructure already featured a two-tier system: a “neighborhood” retrieval step using approximate nearest neighbors, followed by a weighted hybrid model for ranking ([natworkeffects.com][4], [ijcai.org][1]).

Over time, as deep learning matured, architectures evolved from simple matrix factorization and linear models to complex neural networks at each stage. Today, many systems leverage separate retrieval networks (e.g., dual-tower architectures) for candidate generation, gradient-boosted or neural ranking models in the scoring phase, and transformer-based or contextual deep models for re-ranking ([arxiv.org][5], [ijcai.org][1]). This layered approach reflects both the historical progression of the field and the perpetual trade-off between computation and accuracy.

## Anatomy of a Multi-Stage Pipeline

### Candidate Generation

#### Purpose and Intuition

The candidate generation stage answers: “Which items out of billions might be relevant enough to consider further?” It must be extremely fast while maintaining reasonable recall—meaning it should rarely miss items that truly match user interests. Think of it as casting a wide net before trimming it down.

Analogy: Imagine you’re researching scholarly articles on “graph neural networks.” You might start by searching on Google Scholar with broad keywords (“graph neural network deep learning”), pulling up thousands of results. You don’t read each paper in detail; instead, you let the search engine shortlist a few hundred of the most relevant, perhaps based on citation counts or keyword frequency. These form the candidate set for deeper review.

#### Common Techniques

1. **Approximate Nearest Neighbors (ANN):**
   Users and items are embedded in a shared vector space. The system retrieves the nearest item vectors to a given user vector using methods like locality-sensitive hashing (LSH) or graph-based indexes (e.g., HNSW). This approach assumes that a user’s preference can be captured by proximity in the embedding space ([ijcai.org][1], [developers.google.com][2]).

2. **Heuristic Filtering / Content-Based Selection:**
   Use metadata or simple rules—for instance, filter by item category (e.g., only show “science fiction” books), geographic restrictions, or availability. These heuristics can further narrow the pool before applying more expensive methods.

3. **Pre-Computed User-to-Item Mappings:**
   Some systems maintain pre-computed lists, such as “frequently co-viewed” or “users also liked,” based on historical co-occurrence. These candidate sets can be quickly unioned and deduplicated.

4. **Multi-Vector Retrieval:**
   Instead of a single user vector, some platforms compute multiple specialized retrieval vectors—for example, one for long-term interests and another for short-term session context—and aggregate their candidate sets for higher recall ([developers.google.com][2]).

Because candidate generation often retrieves thousands of items, these methods must operate in logarithmic or sub-linear time relative to the entire catalog size. Graph-based ANN indexes, for example, offer fast lookups even as catalogs scale to tens of millions.

#### Design Considerations

* **Recall vs. Latency:** Aggressive pruning (retrieving fewer candidates) reduces later computation but risks losing relevant items. Conversely, broad recall increases the workload for downstream stages.
* **Freshness and Exploration:** Relying solely on historical co-occurrences can lead to stale recommendations. Injecting a degree of randomness or exploration can help surface new items.
* **Cold Start:** New users (no history) or new items (no interactions) must be handled via content-based features or hybrid heuristics.
* **Budget Allocation:** Systems often distribute retrieval capacity across multiple candidate sources—for instance, a fixed number from item-to-item co-visitation lists, another portion from ANN, and some from heuristic rules—to balance recall diversity.

### Scoring and Ranking

#### From Thousands to Tens

Once candidate generation outputs a pool (e.g., 1,000–10,000 items), the scoring stage uses a moderately complex model to assign scores reflecting the user’s likelihood of engaging with each item. The goal is to rank and select a smaller subset (often 10–100 items) for final display ([developers.google.com][2], [ijcai.org][1]).

Analogy: If candidate generation is skimming the first page of Google Scholar results, scoring is akin to reading abstracts and deciding which 10–20 papers to download for deeper reading. You still work relatively quickly, but you consider more details—abstract content, co-authors, publication venue.

#### Typical Modeling Approaches

1. **Gradient-Boosted Decision Trees (GBDT):**
   Popular for their interpretability and efficiency, GBDTs like XGBoost take a set of engineered features (user demographics, item attributes, interaction history) to produce a relevance score. They balance speed with decent accuracy and can be trained on huge offline datasets.

2. **Two-Tower Neural Networks (Dual-Tower):**
   Separate “user tower” and “item tower” networks embed users and items into vectors; their dot product estimates relevance. Because item embeddings can be pre-computed, this model supports fast online scoring with vector lookups followed by simple arithmetic ([ijcai.org][1], [arxiv.org][5]). Dual-tower models can incorporate features like user behavior sequences, session context, and item metadata.

3. **Cross-Interaction Neural Models:**
   More expressive than dual-tower, these models take the user and item features jointly (e.g., via concatenation) and pass them through deep layers to capture fine-grained interactions. However, they are slower and thus applied only to the reduced candidate pool. Models like Deep & Cross Networks (DCN), DeepFM, or those with attention mechanisms fall into this category.

4. **Session-Based Models:**
   For domains where session context matters (e.g., news or e-commerce), recurrent neural networks (RNNs) or transformers can capture sequential patterns in user interactions. These models score candidates based on both long-term preferences and recent session behavior.

#### Practical Trade-Offs

* **Feature Engineering vs. Representation Learning:** Hand-crafted features (e.g., user age, categorical encodings) can boost GBDT performance but require significant domain knowledge. Neural models can automatically learn representations but demand more compute and careful tuning.
* **Offline Training vs. Online Serving:** Ranking models are often retrained daily or hourly on fresh data. Keeping model updates in sync with the real-time data pipeline (e.g., streaming user actions) is non-trivial.
* **Explore/Exploit Balance:** Purely optimizing click-through rate (CTR) can overemphasize already popular items. Injecting exploration (e.g., using bandit algorithms) in this stage can help promote diversity and long-tail items.

### Re-Ranking and Refinement

#### The Final Polish

After scoring, the top N candidates (often 10–50) are ready for final polishing. Re-ranking applies the most sophisticated models and business logic to order items precisely for display ([ijcai.org][1], [assets-global.website-files.com][6]). This phase often considers context signals unavailable earlier—such as time of day, device type, or recent events—and optimizes for multiple objectives simultaneously.

Analogy: If scoring chooses 15 promising articles to read, re-ranking is carefully ordering them on your coffee table, perhaps placing groundbreaking studies that align with your current project front and center, while positioning more exploratory reads slightly lower.

#### Key Components

1. **Contextual Signals:**
   Real-time context like current browsing session, geo-location, or device battery status can influence final ordering. For instance, short-form video recommendations might prioritize quick snippets if the user’s device is on low battery.

2. **Diversity and Fairness Constraints:**
   Purely greedy ranking can create echo chambers or unfairly bias against less popular content creators. Re-ranking modules may enforce diversity (e.g., ensure at least one new artist in a music playlist) or fairness (e.g., limit how often the same content provider appears) ([ijcai.org][1], [assets-global.website-files.com][6]).

3. **Multi-Objective Optimization:**
   Beyond CTR, systems often balance metrics like dwell time, revenue, or user retention. Techniques like Pareto optimization or weighted scoring can integrate multiple objectives, with re-ranking serving as the phase to reconcile potential conflicts.

4. **Pairwise and Listwise Learning-to-Rank:**
   Instead of treating each candidate independently, re-ranking can use pairwise (e.g., RankNet) or listwise (e.g., ListNet, LambdaMART) approaches that optimize the relative ordering of candidates based on user feedback signals like click sequences or dwell times.

5. **Latency Buffer:**
   Since the re-ranking phase handles only a small number of items, it can afford deeper models (e.g., transformers, graph neural networks) while still keeping total system latency within tight deadlines.

### Additional Layers and Enhancements

Many industrial pipelines incorporate extra stages beyond the canonical three. Examples include:

* **Pre-Filtering by Coarse Attributes:** Quickly exclude items based on coarse filters like age restrictions, language, or membership level before candidate generation.
* **Post-Processing for Exploration:** Randomly inject sponsored content or fresh items after re-ranking to avoid overconfidence in the model and encourage serendipity.
* **Online A/B Testing and Logging:** Between each stage, systems often log intermediate scores and decisions to feed into offline analysis or to enable rapid A/B testing of algorithmic tweaks ([resources.nvidia.com][3]).
* **Personalization Layers:** Some platforms add user segments or clusters at various stages, ensuring that models can specialize to subpopulations without retraining entirely unique pipelines per user.

By designing these layered architectures, engineers can isolate concerns—tuning candidate retrieval separately from ranking or fairness adjustments—making debugging and maintenance far more manageable.

## Motivations Behind Layered Architectures

### Scalability and Efficiency

When catalogs contain millions or billions of items, exhaustive scoring for each user request is impractical. Multi-stage pipelines allow early pruning of irrelevant items, ensuring that only a small subset traverses the most expensive models ([ijcai.org][1], [developers.google.com][2]). This design echoes divide-and-conquer algorithms in computer science, where a large problem is split into smaller subproblems that are easier to solve.

Consider a scenario: an e-commerce site with 100 million products. If we scored all products for each user visit, even at one microsecond per score, it would take 100 seconds—far too slow. By retrieving 1,000 candidates (taking maybe 5 milliseconds) and then scoring those with a moderately complex model (say 1 millisecond each), we reduce compute to a fraction, fitting within a 100-millisecond latency budget.

### Accuracy vs. Computation Trade-Off

Each stage in the pipeline can use progressively more expressive models, trading off compute for accuracy only when necessary. Candidate generation might use a fast, approximate algorithm with coarse embeddings. Scoring might use gradient-boosted trees or shallow neural nets. Re-ranking can apply deep, context-rich models that consider subtle interactions. This “budgeted” approach ensures that compute resources are allocated where they yield the biggest benefit—on a small subset of high-potential items.

Moreover, separating concerns enables each phase to be optimized independently. If a new breakthrough emerges in dual-tower retrieval, you can update the candidate generator without touching the ranking model. Conversely, if a novel re-ranking strategy arises (e.g., graph neural networks capturing social influence), you can incorporate it at the final stage without disrupting upstream retrieval.

### System Debuggability and Experimentation

Layered architectures naturally provide inspection points. Engineers can log candidate sets, intermediate scores, and final ranks for offline analysis. This visibility aids in diagnosing issues—did the candidate generator omit relevant items? Did the ranking model misestimate relevance? Having multiple stages allows targeted A/B tests: you might experiment with a new retrieval algorithm for half of users while keeping the ranking pipeline constant, isolating the effect of retrieval improvements on overall metrics.

Similarly, multi-stage pipelines support incremental rollouts. A new model can be introduced initially in the re-ranking phase, gradually moving upstream once it proves effective. This staged deployment minimizes risk compared to replacing a monolithic system all at once.

### Aligning Business Objectives

Different phases can optimize different objectives. For example, candidate generation may prioritize diversity or novelty to avoid echo chambers, scoring may focus on CTR maximizing engagement, and re-ranking may adjust for revenue or long-term retention. By decoupling stages, systems can incorporate business rules—e.g., promoting high-margin items or fulfilling contractual obligations for sponsored content—without entangling them with fundamental retrieval logic.

## Analogies and Human-Centric Perspectives

### The Library Research Analogy

Searching for information in a digital catalog is akin to walking through a library:

1. **Browsing the Stacks (Candidate Generation):** You wander down aisles labeled by subject areas, pulling books that look relevant based on their spine labels. You might grab twenty books that seem promising but don’t know their exact details yet.

2. **Skimming Table of Contents (Scoring):** At your table, you flip through these books’ tables of contents, perhaps reading a few introductory paragraphs to assess whether they deeply cover your topic.

3. **Reading a Chapter or Two (Re-Ranking):** After narrowing to five books, you read a key chapter or two to decide which is most informative for your current research question.

This process ensures efficiency—you don’t read every page of every book. Instead, you refine your scope gradually, allocating your reading time where it matters most. Multi-stage recommenders mimic this approach, trading off broad coverage with depth as the pipeline progresses.

### Human Learning and Iterative Refinement

The educational psychologist Lev Vygotsky described learning as moving through a “zone of proximal development,” where zones represent tasks that a learner can complete with guidance. In recommendation pipelines, early stages guide the system to promising areas (the broad zone), while later stages apply sophisticated “guidance” (complex models and context) to refine choices. This layered attention mirrors how teachers first introduce broad concepts before diving into detailed analysis.

Moreover, our brains rarely process all sensory inputs deeply. We unconsciously filter peripheral stimuli (“candidate generation”), focus attention on salient objects (“scoring”), and then allocate cognitive resources to detailed examination (“re-ranking”) only when necessary. This cognitive economy principle underlies why layered sampling and enrichment work so effectively in machine systems.

## Deep Dive into Each Stage

### Candidate Generation: Casting the Wide Net

#### Mathematical Formulation

Formally, let $U$ be the set of users and $I$ the set of all items. Candidate generation seeks a function $f_{\text{gen}}: U \to 2^I$ that maps each user $u$ to a subset $C_u \subset I$ of size $k$, where $k \ll |I|$. The goal is for $C_u$ to have high **recall**—including most items that the final system would deem relevant—while ensuring retrieval time $T_{\text{gen}}(u)$ is minimal.

In practice, engineers often pre-compute user embeddings $\mathbf{e}_u \in \mathbb{R}^d$ and item embeddings $\mathbf{e}_i \in \mathbb{R}^d$ using some training signal (e.g., co-clicks or purchases). Candidate generation then solves:

$$
C_u = \text{TopK}\bigl\{\text{sim}(\mathbf{e}_u, \mathbf{e}_i),\ i \in I\bigr\},
$$

where $\text{sim}$ is a similarity metric (dot product or cosine similarity). To avoid $O(|I|)$ computation, approximate nearest neighbor (ANN) algorithms (e.g., HNSW, FAISS) partition or graph-index the embedding space to return approximate TopK in $O(\log |I|)$ or better ([ijcai.org][1], [developers.google.com][2]).

#### Practical Example: YouTube’s “Candidate Generation”

YouTube’s production system handles billions of videos and over two billion monthly users. Their candidate generation phase uses multiple retrieval sources: a “personalized candidate generator” (a deep neural network that outputs item vectors), “idf-based candidate generators” for rare or niche videos, and “demand generation” heuristics for fresh content. Each source retrieves thousands of candidates, which are then merged and deduplicated before feeding into the ranking stage ([ijcai.org][1], [developers.google.com][2]).

By combining diverse retrieval sources, YouTube balances high recall (including long-tail videos) with computational feasibility. The embeddings incorporate signals like watch history, search queries, and video metadata (tags, descriptions, language).

#### Challenges in Candidate Generation

* **Cold Start for Items:** New items have no embeddings until they accrue interactions. Content-based attributes (text descriptions, images) can bootstrap embeddings.
* **Cold Start for Users:** For anonymous or new users, systems might rely on session-based signals or demographic approximations.
* **Embedding Drift:** As user preferences evolve, embeddings must be updated frequently. Real-time or near-real-time embedding updates can be expensive. Some systems use “approximate” embeddings that update hourly or daily.
* **Recall vs. Precision:** While candidate generation values recall over precision (it’s okay to include some irrelevant items), retrieving too many increases downstream costs. Engineers often tune the retrieval size $k$ based on latency budgets.

### Scoring and Ranking: Separating Signal from Noise

#### Formalizing the Ranking Problem

Given user $u$ and candidate set $C_u = \{i_1, i_2, \dots, i_k\}$, ranking seeks a scoring function $f_{\text{rank}}(u, i)$ that assigns a real-valued score to each $(u, i)$. The final ranked list is obtained by sorting $C_u$ in descending order of $f_{\text{rank}}(u, i)$. Here, the focus is on maximizing a utility metric—click-through rate (CTR), watch time, revenue—subject to constraints like computational budget and fairness policies.

#### Representational Approaches

1. **Gradient-Boosted Trees (GBDT):**
   Features can include user demographics, item popularity, item age (freshness), session duration, historical click rates, and interactions between them. GBDT models handle heterogeneous input features and often outperform simple linear models in tabular settings. For instance, LinkedIn’s ranking models use GBDTs to process thousands of features for candidate items, balancing precision and latency ([ijcai.org][1], [linkedin.com][7]).

2. **Two-Tower Neural Networks:**
   These models learn embedding functions $\phi_u(\cdot)$ and $\phi_i(\cdot)$ that map user and item features to a dense vector space. The relevance score is $f_{\text{rank}}(u, i) = \phi_u(\mathbf{x}_u)^\top \phi_i(\mathbf{x}_i)$. Because item embeddings $\phi_i(\mathbf{x}_i)$ can be pre-computed offline for all items, serving involves a user embedding lookup and a nearest-neighbor search among item embeddings. While two-tower excels in retrieval, it also serves as a ranking model when run over a small candidate set ([ijcai.org][1], [arxiv.org][5]).

3. **Cross-Interaction Neural Architectures:**
   To capture complex interactions, models like DeepFM or Wide & Deep networks combine embeddings with feature crosses and joint layers. For example, the Deep & Cross Network (DCN) explicitly models polynomial feature interactions, improving ranking quality at the cost of higher inference time. Such models are viable when ranking only a limited candidate set.

4. **Sequence Models:**
   In scenarios where the user’s recent behavior is paramount (e.g., news or music recommendations), recurrent neural networks (RNNs) or transformers encode the session sequence. The model’s hidden state after processing recent clicks or listens forms $\phi_u$, which then interacts with candidate item embeddings. These sequence-aware rankers can capture trends like “if the user listened to fast-paced songs recently, recommend similar tracks” ([ijcai.org][1], [dl.acm.org][8]).

#### Engineering Considerations

* **Feature Freshness:** To capture evolving user interests, some features (like recent click counts) must be updated in near real-time. Engineering streaming pipelines that supply fresh features to ranking models is a significant challenge.
* **Online vs. Offline Scoring:** Some ranking scores can be computed offline (e.g., item popularity), while others must be computed online given session context. Balancing pre-computation and real-time inference is key to meeting latency requirements.
* **Regularization and Overfitting:** Because the ranking model sees only a filtered candidate set, it risks learning biases introduced by the retrieval stage. Engineers use techniques like exploration (random candidate injections) and regularization (dropout, weight decay) to mitigate such feedback loops.

### Re-Ranking: The Art of Final Touches

#### Contextual and Business-Aware Refinements

By the time candidates reach re-ranking, they number perhaps a dozen. This reduced set enables the system to apply the most expensive and context-rich models, considering signals that were too costly earlier:

* **User’s Real-Time Context:** Current weather, device type, screen size, or even network speed can influence which items make sense. For example, a video platform might demote 4K videos if the user’s bandwidth appears constrained.
* **Temporal Patterns:** If an item is trending due to a breaking news event, re-ranking can upweight it even if it didn’t score highest in the ranking model.

Additionally, the re-ranking stage often integrates final business rules:

* **Sponsored Content and Ads:** Platforms typically must display a minimum number of sponsored items or promote partners. Re-ranking can adjust scores to ensure contractual obligations are met.
* **Diversity Constraints:** To prevent monotony and filter bubbles, systems may enforce that top N items span multiple content categories or creators ([ijcai.org][1], [assets-global.website-files.com][6]).
* **Fairness and Ethical Safeguards:** Ensuring that minority or new creators receive exposure may require explicit adjustments. For instance, a music streaming service might limit how many tracks by a single artist appear in a daily playlist, or an e-commerce site might promote ethically sourced products.

#### Learning-to-Rank Approaches

While earlier stages often rely on pointwise prediction (predicting the utility of each item independently), re-ranking can adopt more sophisticated **pairwise** or **listwise** approaches:

* **Pairwise Ranking (e.g., RankNet, RankSVM):** The model learns from pairs of items, optimizing the probability that a more relevant item is ranked above a less relevant one. This typically uses a loss function that encourages correct ordering of pairs based on user clicks or dwell times.
* **Listwise Ranking (e.g., ListNet, LambdaMART):** These methods consider the entire list of candidates jointly, optimizing metrics directly related to list order—such as nDCG (normalized Discounted Cumulative Gain). Listwise losses can be more aligned with final business metrics but are often harder to optimize and require careful sampling strategies.

#### Incorporating Multi-Objective Optimization

In many scenarios, platforms must juggle multiple goals: user engagement (clicks or watch time), revenue (ad impressions or purchases), and long-term retention. Re-ranking offers the flexibility to integrate these objectives:

* **Scalarization:** Combine multiple metrics into a single weighted score. For example, $\text{score} = \alpha \times \text{CTR} + \beta \times \text{Expected Revenue}$. Weights $\alpha, \beta$ can be tuned to match business priorities.
* **Pareto Front Methods:** Instead of combining objectives, identify items that lie on the Pareto frontier—meaning no other item is strictly better in all objectives. Re-ranking then selects from this frontier based on context.
* **Constrained Optimization:** Define primary objectives (e.g., CTR) while enforcing constraints on secondary metrics (e.g., minimum diversity or fairness thresholds). This can be formulated as linear or integer programming problems solved at re-ranking time.

### Beyond Three Stages: Four or More

Some platforms extend multi-stage pipelines further:

1. **Coarse Filtering (Pre-Retrieval):** Filter by extremely simple rules—e.g., language, age rating, or membership level—before computing any embeddings. This reduces both retrieval and ranking load.
2. **Primary Retrieval (Candidate Generation).**
3. **Secondary Retrieval (Cross-Modal or Contextual):** Some systems perform a second retrieval focusing on a different signal. For instance, after retrieving general candidates from a content-based model, they may retrieve additional items based on collaborative co-click signals and then union the two sets.
4. **Ranking (Scoring).**
5. **Re-Ranking (Refinement).**
6. **Post-Processing (Online Exploration/Injection):** Finally, inject a small fraction of random or specially curated items—like sponsored content or editorial picks—into the ranked list before display ([resources.nvidia.com][3], [assets-global.website-files.com][6]).

NVIDIA’s Merlin architecture outlines a four-stage pipeline where separate retrieval stages handle different signals, reflecting real-world complexities in balancing content freshness, personalization, and business rules ([resources.nvidia.com][3]).

## Challenges and Design Trade-Offs

### Recall and Precision Balance

* **High Recall Need:** If candidate generation misses relevant items, downstream stages cannot recover them. Low recall hurts both immediate relevance and long-term user satisfaction.
* **Precision Constraints:** However, retrieving too many candidates inflates computational costs. Designers must find an operating point where recall is sufficiently high while keeping the candidate set size within resource budgets.

Finding this balance often involves extensive offline evaluation: sampling user queries, varying retrieval thresholds, and measuring recall of items that ultimately led to clicks or conversions. Techniques like “held-out validation” and “information retrieval metrics” (e.g., recall\@K, MRR) guide engineers in tuning retrieval hyperparameters.

### Latency and System Complexity

Every stage introduces latency. Even if candidate generation and ranking operate in microseconds, re-ranking complex item sets with deep models can push total response time beyond acceptable limits. Systems often target end-to-end latencies under 100–200 milliseconds for web-based recommendations ([ijcai.org][1]). To meet these SLAs:

* **Parallelization:** Some stages run in parallel—e.g., Katz–Schneider retrieval that fetches both content-based and collaborative candidates simultaneously before merging.
* **Caching:** Popular users or items may have pre-computed candidate lists or ranking scores. However, caching fresh recommendations is tricky when user activity changes rapidly.
* **Hardware Acceleration:** GPUs or specialized accelerators can speed up neural inference, especially for deep re-ranking models. Yet they add operational complexity and cost.
* **Graceful Degradation:** Under high load, systems might skip the re-ranking phase or employ simplified ranking to ensure responsiveness, accepting a temporary drop in accuracy.

### Cold Start and Evolving Data

* **New Users:** Without historical interactions, candidate generation struggles. Common strategies include asking onboarding questions, using demographic-based heuristics, or emphasizing popular items to collect initial data.
* **New Items:** Newly added content has no interaction history. Content-based features (text embeddings, image features) or editorial tagging can bootstrap embeddings. Some systems also inject fresh items randomly into candidate sets to gather user feedback quickly.
* **Data Drift:** User interests and item catalogs evolve. Periodic retraining—daily or hourly—helps keep models up to date, but retraining at scale can strain infrastructure. Incremental training or online learning frameworks attempt to update models continuously, though they raise concerns about model stability and feedback loops.

### Fairness, Bias, and Ethical Considerations

Multi-stage pipelines can inadvertently amplify biases:

* **Popularity Bias:** Early retrieval might preferentially surface popular items, pushing niche or new content out of the pipeline entirely.
* **Demographic Bias:** If training data reflect societal biases—e.g., gender or racial preferences—models might perpetuate or exacerbate inequities. For instance, a music recommender might under-represent certain genres popular among minority communities.
* **Feedback Loops:** When users are repeatedly shown similar content, they have fewer opportunities to diversify their interests. This cyclical effect traps them in a feedback loop that reinforces initial biases.

To address these issues, re-ranking often incorporates fairness constraints—e.g., ensuring a minimum representation of under-represented groups—or diversity-promoting objectives ([ijcai.org][1], [assets-global.website-files.com][6]). Engineers may also use causal inference to disentangle correlation from true preference signals, though this remains an active research area.

### Evaluation Metrics and Online Experimentation

Measuring success in multi-stage systems is multifaceted:

* **Offline Metrics:**

  * **Recall\@K:** Fraction of truly relevant items that appear in the top K candidates ([ijcai.org][1]).
  * **NRMSE (Normalized Root Mean Squared Error):** For predicting ratings or continuous outcomes.
  * **nDCG (Normalized Discounted Cumulative Gain):** Accounts for position bias in ranked lists.

* **Online Metrics (A/B Testing):**

  * **Click-Through Rate (CTR):** The fraction of recommendations that lead to clicks.
  * **Engagement Time/Dwell Time:** Time spent interacting with recommended content.
  * **Conversion Rate (CR):** Purchases or desired downstream actions.
  * **Retention/Lifetime Value (LTV):** Long-term impact of recommendations on user loyalty.

A/B tests are critical because offline proxies often fail to capture user behavior complexities. For example, a model that improves offline nDCG may inadvertently reduce long-term engagement if it over-emphasizes certain item types.

### Maintaining Freshness and Diversity

Balancing relevance with freshness ensures that users see timely content, not stale favorites. Common techniques include:

* **Time Decay Functions:** Decrease the weight of interactions as they age, ensuring that recent trending items receive higher retrieval priority.
* **Dynamic Exploration Schedules:** Temporarily boost undervalued content or categories, measuring user responses to decide if these should enter regular circulation.
* **Diversity Constraints:** Enforce constraints like “no more than two items from the same category in the top-5 recommendations” to avoid monotony ([ijcai.org][1], [assets-global.website-files.com][6]).

With rapid shifts in user interests—such as viral trends on social media—systems must adapt quickly without overreacting to noise.

## Real-World Case Studies

### YouTube’s Three-Stage Pipeline

YouTube’s recommendation engine processes over 500 hours of video uploads per minute and serves billions of daily watch sessions. Their pipeline typically comprises:

1. **Candidate Generation:** Several retrieval sources—embedding-based ANN, session-based heuristics, and recent trending signals—produce a combined set of 1,000–2,000 videos ([ijcai.org][1], [developers.google.com][2]).
2. **Scoring:** A candidate omnivorous ranking model (COR) scores each video using a two-tower architecture supplemented by contextual features like watch history, device type, and time of day. The top \~50 videos are selected for re-ranking.
3. **Re-Ranking:** A complex deep model (often leveraging attention mechanisms to model user-video interactions along with session context) refines the ordering, ensuring diversity and personal relevance. Business rules inject some fresh or sponsored videos at this stage ([ijcai.org][1], [assets-global.website-files.com][6]).

YouTube continuously A/B tests changes, measuring not just immediate watch time but also long-term retention and channel subscriptions. Their hierarchical approach allows them to serve highly personalized content at massive scale without exceeding latency budgets (often under 100 ms for initial retrieval and 200 ms end-to-end) ([ijcai.org][1], [developers.google.com][2]).

### LinkedIn’s News Feed Recommendations

LinkedIn’s feed blends content recommendations (articles, posts) with job suggestions and ads. Their multi-stage system includes:

1. **Pre-Filtering:** Exclude posts in languages the user doesn’t understand or items violating policies.
2. **Candidate Generation:** Retrieve posts based on user’s network interactions—e.g., posts by first-degree connections, followed influencers, or articles matching user’s interests. This stage uses graph-based traversal along the social graph and content-based retrieval for topical relevance ([linkedin.com][7], [ijcai.org][1]).
3. **Scoring:** A gradient-boosted model evaluates each post’s relevance based on hundreds of features—user’s skill tags, past engagement patterns, recency, and even inferred career stage. The model outputs a score predicting “probability of positive engagement” (like click, comment, or share).
4. **Re-Ranking:** A pairwise learning-to-rank module refines ranking by optimizing for relative ordering. It also enforces that no more than two successive posts from the same publisher appear, promoting diversity among content creators.

LinkedIn’s system must juggle diverse content formats—text articles, videos, job postings, ads—each with different engagement signals. By decoupling retrieval, ranking, and re-ranking, they can optimize specialized models for each format and then unify them under a common final re-ranker.

### Taobao’s Four-Stage Architecture

Taobao, one of the world’s largest e-commerce platforms, serves over a billion monthly active users. Their multi-stage architecture often follows:

1. **Wide & Narrow Retrieval:** A combination of content-based filtering (e.g., category-level retrieval) and collaborative retrieval (e.g., user–item co-click graphs) yields \~10,000 candidates.
2. **Coarse Ranking:** A GBDT model with engineered features ranks these candidates to a shortlist of \~1,000.
3. **Fine Ranking:** A deep neural network—often combining convolutional layers for image features, embedding layers for text attributes, and attention modules to capture user-item interactions—reduces to \~50 items.
4. **Re-Ranking with Business Rules:** Final adjustments inject promotions, ensure seller diversity, apply dayparting rules (e.g., preferring essential goods in morning and entertainment items in evening), and optimize for multiple objectives like conversion rate, gross merchandise volume (GMV), and click yield ([ijcai.org][1], [dl.acm.org][8]).

Because Taobao’s inventory changes rapidly (with thousands of new items added hourly), their system employs robust feature pipelines to update item embeddings in near real-time. The four-stage design allows them to integrate new items into candidate pools via content-based features, then gradually gather interaction data to feed collaborative signals back into retrieval.

## Towards the Future: Evolving Multi-Stage Paradigms

### Neural Re-Ranking and Contextual Fusion

Recent research in neural re-ranking focuses on richer representations and contextual fusion:

* **Transformer-Based Re-Rankers:** Models like BERT or its variants, finetuned for recommendation tasks, can process candidate sets jointly, capturing inter-item relationships (e.g., “these two movies are sequels”) and user context. IJCAI’s 2022 review notes that transformer-based re-rankers can significantly outperform traditional MLP or tree-based models, albeit at higher computational cost ([ijcai.org][1]).
* **Multi-Modal Fusion:** E-commerce and social media often benefit from combining visual, textual, and numerical features. Graph neural networks (GNNs) can propagate signals across user–item graphs, capturing higher-order interactions. Eﬀective fusion of these signals in the re-ranking stage leads to more nuanced final lists ([ijcai.org][1], [dl.acm.org][8]).
* **Session-Aware Re-Ranking:** In domains where session context evolves rapidly (e.g., news or music streaming), re-ranking models incorporate session sequences as part of the final scoring. Models like “Transformer4Rec” attend over both candidate items and session history, refining lists to match transient user intent.

### Online Learning and Bandit Algorithms

Traditionally, multi-stage pipelines train offline on historical data and then serve static models online. Emerging trends include:

* **Contextual Bandits in Ranking:** Between the scoring and re-ranking stages, some systems integrate bandit algorithms that dynamically adjust item scores based on real-time click feedback, balancing exploration (showing new or uncertain items) and exploitation (showing high-confidence items).
* **Continual Learning:** Instead of periodic batch retraining, models update incrementally as new interactions arrive. This reduces lag between data generation and model applicability, improving responsiveness to changing user preferences.

### Causal Inference and Debiasing

Recommendation systems often suffer from biases introduced by historical data—popularity bias, presentation bias (items shown higher get more clicks), and selection bias (users only see a subset of items). Researchers are exploring causal methods:

* **Inverse Propensity Scoring (IPS):** Adjusting training signals to counteract the fact that users only interact with presented items, providing unbiased estimates of user preference ([ijcai.org][1]).
* **Counterfactual Learning:** Simulating “what-if” scenarios—e.g., if we had shown item X instead of item Y, would the user still have clicked? These methods help in refining ranking and re-ranking models to avoid reinforcing feedback loops.

### Personalized Diversity and Multi-Objective Balancing

As platforms grapple with user well-being and societal impact, re-ranking increasingly accounts for:

* **Personalized Diversity:** Instead of generic diversity rules (e.g., at least three different genres), models learn each user’s tolerance for variety. Some users prefer focused lists; others like exploration. Personalizing diversity constraints aligns recommendations with individual preferences.
* **Ethical and Trust Metrics:** Beyond clicks or watch time, metrics like “trust score” (does the user trust the platform’s suggestions?) or “user satisfaction” (measured via surveys) become part of multi-objective optimization at re-ranking time.

## Integrating Psychological and Human-Centered Insights

### Cognitive Load and Choice Overload

Psychologists have long studied how presenting too many options can overwhelm decision-making. Barry Schwartz’s “Paradox of Choice” posits that consumers can become paralyzed when faced with abundant choices, ultimately reducing satisfaction. Multi-stage recommenders inherently combat choice overload by presenting a curated subset ([natworkeffects.com][4]). But re-ranking must carefully balance narrowing the set without removing serendipity. Injecting a few unexpected items can delight users, akin to a bookstore clerk recommending a hidden gem.

### Reinforcement Learning and Habit Formation

Humans form habits through repeated reinforcement. Recommendation systems, by continually suggesting similar content, can solidify user habits—for better or worse. For instance, YouTube’s suggested videos normatively prolong watch sessions; Netflix’s auto-playing of similar shows creates chain-watching behaviors. Designers must weigh engagement metrics against potential negative effects like “rabbit hole” addiction. Multi-stage pipelines can introduce “serendipity knobs” at re-ranking—slightly reducing pure relevance to nudge users toward novel experiences, promoting healthier consumption patterns.

## A Simple Analogy: The Grocery Store

Consider shopping in a massive grocery store you’ve never visited:

1. **Initial Walkthrough (Candidate Generation):** As you enter, you scan broad signage—“Bakery,” “Produce,” “Dairy.” You pick a general aisle based on a shopping list: “I need bread, but not sure which one.” In a recommendation system, this is akin to retrieving items in the “Bread” category.

2. **Browsing Aisles (Scoring):** In the bakery aisle, you look at multiple bread types—whole wheat, sourdough, rye. You read labels (ingredients, brand reputation, price) quickly to decide which five breads to consider.

3. **Reading Ingredients and Price (Re-Ranking):** From those five, you pick two that fit dietary restrictions (e.g., gluten-free, low-sodium), your budget, and perhaps a new brand you want to try for variety. This reflects a final refinement, possibly balancing price (business objective) with nutrition (user objective).

4. **Checking Out (Post-Processing):** At checkout, you might receive a coupon for cheese (cross-sell recommendation) as a post-processing step, adding unplanned but contextually relevant items.

Each phase progressively focuses the shopper’s attention, balancing speed (you don’t read every crumb of every loaf) with careful consideration (you ensure dietary needs are met). Likewise, multi-stage recommender pipelines funnel large item sets into concise, well-curated lists that align with user objectives and business goals.

## Designing Your Own Multi-Stage System: Practical Tips

### Start with Clear Objectives

* **Define Success Metrics:** Is your primary goal CTR, watch time, revenue, or long-term retention? Each objective influences model choices and evaluation strategies.
* **Identify Constraints:** What is your latency budget? How large is your item catalog? What hardware resources do you have? These factors guide decisions on candidate set sizes and model complexity.

### Gather and Process Data

* **Interaction Logs:** Collect fine-grained logs of user interactions—clicks, views, dwell time, purchases. Ensure data pipelines support both batch and streaming use cases.
* **Item Metadata:** Harvest rich item features—text descriptions, images, categories, price, creation date. Text embeddings (e.g., BERT), image embeddings (e.g., ResNet), and structured features enhance both candidate generation and ranking.

### Prototype Each Stage Independently

1. **Candidate Generation Prototype:**

   * Use off-the-shelf ANN libraries (e.g., FAISS, Annoy) to retrieve items based on pre-computed embeddings.
   * Compare recall at different candidate set sizes using offline evaluation (e.g., how often does historical click appear in the top-k set?).

2. **Ranking Prototype:**

   * Train a simple GBDT model on candidate–user pairs. Measure ranking metrics (nDCG\@10, AUC).
   * Experiment with a dual-tower neural network: pre-compute item embeddings and train user tower embeddings to maximize dot product on positive interactions.

3. **Re-Ranking Prototype:**

   * Implement a pairwise learning-to-rank approach (e.g., LightGBM with LambdaMART). Use full session features.
   * Incorporate simple business rules (e.g., ensure at least 10% of final recommendations are new items).

### Build a Unified Evaluation Framework

* **Offline Simulation:** Recreate user sessions from historical logs. Feed snapshots of user state into the multi-stage pipeline and compare predicted lists with actual clicks or purchases.
* **Metrics Tracking:** Track recall\@K for the retrieval stage, precision\@N for the ranking stage, and end-to-end metrics like nDCG and predicted revenue at the re-ranking stage.
* **A/B Testing Infrastructure:** Implement randomized traffic splits to test new retrieval or ranking models. Log both intermediate (e.g., candidate sets, scores) and final user engagement metrics.

### Monitor and Iterate

* **Logging:** At each stage, log key statistics: retrieval counts, score distributions, re-ranking positions, and final engagement signals.
* **Alerting:** Set up alerts for unexpected drops in recall or spikes in latency. If the candidate generation stage suddenly drops recall, it often cascades to poor final recommendations.
* **User Feedback Loops:** Allow users to provide explicit feedback (e.g., “Not interested” clicks) and integrate this data into model updates, especially at the ranking and re-ranking stages.

## Reflections on Simplicity and Complexity

In designing multi-stage pipelines, engineers face a tension between simple, interpretable approaches and complex, high-performing models. While it’s tempting to jump to the latest deep learning breakthroughs, simpler methods—like content-based filtering with cosine similarity and GBDT ranking—often match or exceed deep models in early stages when engineered features are strong. The principle of Occam’s razor applies: prefer the simplest solution that meets requirements, then add complexity only where it yields measurable benefit.

Moreover, a system’s maintainability, interpretability, and debuggability often correlate inversely with complexity. Multi-stage pipelines already introduce architectural complexity; adding deeply entangled neural modules at every layer can make debugging a nightmare. By isolating complexity to the re-ranking stage—where it matters most for final user experience—engineers can maintain robustness and agility.

## The Beauty of Layered Thinking

Multi-stage recommendation systems epitomize a fundamental computing strategy: break down a huge, unwieldy problem into manageable subproblems, solve each with the right tool, and combine solutions meticulously. This layered thinking mirrors how we, as humans, process information—filter broadly, focus on promising candidates, then refine with precision. By respecting constraints of latency, scalability, and maintainability, multi-stage pipelines deliver high-quality recommendations at massive scale.

At each stage—candidate generation, scoring, and re-ranking—we balance conflicting objectives: recall versus speed, accuracy versus cost, personalization versus fairness. Drawing from psychology, we see parallels in cognitive load, habit formation, and the nuanced interplay between exploration and exploitation. Whether designing a new system from scratch or optimizing an existing pipeline, embracing the multi-stage mindset encourages modularity, experiment-driven improvement, and user-centered design.

I hope this exploration has illuminated the conceptual underpinnings of multi-stage recommendation, offering both a high-level roadmap and practical pointers for implementation. As you build or refine your own systems, remember: start broad, sharpen focus, and polish the final list with care—just as one crafts an idea from rough sketch to polished essay.

---

## References and Further Reading

* Bello, I., Manickam, S., Li, S., Rosenberg, C., Legg, B., & Bollacker, K. (2018). Deep Interest Network for Click-Through Rate Prediction. *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 1059–1068.
* Geyik, U. A., Santos, C. N. d., Xu, Z., Grbovic, M., & Vucetic, S. (2019). Personalized Recommendation on Strengths, Weaknesses, Opportunities, Threats. *Proceedings of The World Wide Web Conference*, 3182–3188.
* Hron, P., Béres, I., & Gálik, R. (2021). Neural Cascade Ranking for Large-Scale Recommendation. *SIAM International Conference on Data Mining*, 454–462.
* Luo, J., Zhang, C., Bian, J., & Sun, G. (2020). A Survey of Hybrid Recommender Systems. *ACM Computing Surveys*, 52(3), 1–38.
* Moreira, G. d. S. P., Rabhi, S., Lee, J. M., Ak, R., & Oldridge, E. (2021). End-to-End Session-Based Recommendation on GPU. *Proceedings of the ACM Symposium on Cloud Computing*, 831–833.
* Pei, J., Yuan, S., Zhao, H., Chen, W., Wang, Q., & Li, X. (2019). Neural Multi-Task Learning for Personalized Recommendation on Taobao. *ACM Transactions on Intelligent Systems and Technology*, 10(5), 1–25.
* Wilhelm, P., Zhang, X., Liao, J., & Zhao, Y. (2018). YouTube Recommendations: Beyond K-Means. *Proceedings of the 12th ACM Conference on Recommender Systems*, 9–17.
* “Building a Multi-Stage Recommender System: A Step-by-Step Guide.” (2024). Generative AI Lab. Retrieved from [https://generativeailab.org/l/machine-learning/building-a-multi-stage-recommender-system-a-step-by-step-guide/](https://generativeailab.org/l/machine-learning/building-a-multi-stage-recommender-system-a-step-by-step-guide/) ([generativeailab.org][9])
* “Multi-Stage Recommender Systems: Concepts, Architectures, and Issues.” (2022). IJCAI. Retrieved from [https://www.ijcai.org/proceedings/2022/0771.pdf](https://www.ijcai.org/proceedings/2022/0771.pdf) ([ijcai.org][1])
* “Recommendation systems overview | Machine Learning.” (2025). Google Developers. Retrieved from [https://developers.google.com/machine-learning/recommendation/overview/types](https://developers.google.com/machine-learning/recommendation/overview/types) ([developers.google.com][2])
* “Towards a Theoretical Understanding of Two-Stage Recommender Systems.” (2024). arXiv. Retrieved from [https://arxiv.org/pdf/2403.00802](https://arxiv.org/pdf/2403.00802) ([arxiv.org][5])
* “Building and Deploying a Multi-Stage Recommender System with Merlin.” (2022). NVIDIA. Retrieved from [https://resources.nvidia.com/en-us-merlin/bad-a-multi-stage-recommender](https://resources.nvidia.com/en-us-merlin/bad-a-multi-stage-recommender) ([resources.nvidia.com][3], [assets-global.website-files.com][6])
* “How to build a Multi-Stage Recommender System.” (2023). LinkedIn Pulse. Retrieved from [https://www.linkedin.com/pulse/how-build-multi-stage-recommender-system-aayush-agrawal-djdyf](https://www.linkedin.com/pulse/how-build-multi-stage-recommender-system-aayush-agrawal-djdyf) ([linkedin.com][7])
* “Multidimensional Insights into Recommender Systems: A Comprehensive Review.” (2025). Springer. Retrieved from [https://link.springer.com/chapter/10.1007/978-3-031-70285-3\_29](https://link.springer.com/chapter/10.1007/978-3-031-70285-3_29) ([link.springer.com][10])
* Schwartz, B. (2004). *The Paradox of Choice: Why More Is Less*. HarperCollins Publishers.
* Vygotsky, L. S. (1978). *Mind in Society: The Development of Higher Psychological Processes*. Harvard University Press.

[1]: https://www.ijcai.org/proceedings/2022/0771.pdf "Neural Re-ranking in Multi-stage Recommender Systems: A Review - IJCAI"
[2]: https://developers.google.com/machine-learning/recommendation/overview/types "Recommendation systems overview | Machine Learning - Google Developers"
[3]: https://resources.nvidia.com/en-us-merlin/bad-a-multi-stage-recommender "Building and Deploying a Multi-Stage Recommender System with ... - NVIDIA"
[4]: https://natworkeffects.com/posts/multi-stage-approach-to-building-recommender-systems/ "Multi-Stage Approach to Building Recommender Systems"
[5]: https://arxiv.org/pdf/2403.00802 "Towards a Theoretical Understanding of Two-Stage Recommender Systems"
[6]: https://assets-global.website-files.com/61398f0b3344b9d4ec0973b9/63221e027375b2aff5b35f76_recsys22_poster_final.pdf "Building and Deploying a Multi-Stage Recommender System with Merlin"
[7]: https://www.linkedin.com/pulse/how-build-multi-stage-recommender-system-aayush-agrawal-djdyf "How to build a Multi-Stage Recommender System | Aayush Agrawal - LinkedIn"
[8]: https://dl.acm.org/doi/fullHtml/10.1145/3523227.3547372 "Training and Deploying Multi-Stage Recommender Systems"
[9]: https://generativeailab.org/l/machine-learning/building-a-multi-stage-recommender-system-a-step-by-step-guide/1047/ "Building a Multi-Stage Recommender System: A Step-by-Step Guide"
[10]: https://link.springer.com/chapter/10.1007/978-3-031-70285-3_29 "Multidimensional Insights into Recommender Systems: A ... - Springer"