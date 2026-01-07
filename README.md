# Experimental and movie tasks reveal unique trajectories of hippocampal functional development in the same participants
* aka movie_scan
  
---

### Highlights:
* Compared neural activity during naturalistic movie-viewing and a controlled experimental task within the same children
* ML approach to quantifying hippocampal functional maturity via adult-trained models that generalize to children across development
* Demonstrated that familiar and novel task contexts reveal fundamentally different aspects of neural development
* End-to-end reproducible pipeline provided
* Figure below illustrates age-related generalization of adult-trained hippocampal models and effect of task-dependent differences in functional maturity on memory.


### Methods at a glance:
* Supervised out-of-sample classification and AUC-based evaluation
* Time-series similarity metrics (ISC)
* ROI-based feature engineering
* Cross-participant modeling

### Repo description: 
This repository includes end-to-end code and documentation for all steps of analysis, from raw fMRI preprocessing through final statistical analyses and figure generation.
Anonymized, subject-level datasets derived from preprocessed fMRI data are provided for both tasks, enabling full replication of all reported analyses and figures from the manuscript.
For step-by-step, manuscript-aligned analyses, see **[full_manuscript_analyses.md]()** which includes embedded links to modular markdown files and Jupyter/R notebooks, as well as example function calls optimized for high-performance computing environments (see [bin]() for code). 
Markdown files also include detailed analytical rationale and implementation logic; for conceptual motivation and hypothesis justification, see **[main manuscript]()**

---

### Project description:
Studies in which children watch movies while undergoing fMRI are becoming increasingly popular approaches to studying neural function in developmental populations because they minimize motion, engage children's attention, and provide task demands which children understand and are familiar with (Vanderwal et al., 2019). However, no work to date has directly compared neural activity during movie-viewing to traditional experiments in the same participants. To interpret developmental cognitive neuroscience findings amidst the field's shift towards movie-viewing, it is critical to compare functional neural maturity in both contexts.

One way that movie-viewing tasks likely significantly differ from traditional experiments in the neural processes they engage is task familiarity. Children are familiar with the demands of watching and discussing a movie, but often have no prior exposure to experimental tasks they are asked to perform in the scanner. Since familiarity with a task or context significantly affects both memory performance and neural activity, particularly in children (Hudson & Nelson, 1986; Bruer & Pozzulo, 2014), the inferences we draw about neural development may depend strongly on task context.

Here, we directly capture neural activity within the same participants while viewing movies and while completing an experimental statistical learning task. We compare neural function in hippocampus, a critical region supporting memory and demonstrating age-related differences in engagement during memory tasks. Importantly, defining an *a priori* anatomical region of interest provides a controlled and interpretable comparison across task contexts. Since anterior hippocampus demonstrates protracted development relative to posterior (Calabro et al., 2020), we predicted that familiar (movie-viewing) and novel (statistical learning task) tasks would reveal distinct developmental trajectories of functional maturity within anterior hippocampus. Specifically, while we predicted functional anterior hippocampal maturity (defined as more adult-like activity) would facilitate memory in both familiar and novel contexts, neural maturity would emerge along a more extended trajectory for the novel task relative to the familiar task.

To quantify alignment between child and adult neural activity, we employed two complementary approaches:
1) An out-of-sample generalization framework, in which classifiers trained on adult neural activity were evaluated on children of different ages (AUC), and
2) Inter-subject correlation analyses, measuring similarity between child and adult neural time series during time-locked movie viewing.

We found that, in anterior hippocampus, adult-trained classifiers generalized increasingly well to children with age and reliably predicted memory performance, consistent with the emergence of more mature neural representations. In contrast, we observed no systematic age differences in neural maturity during movie-viewing, though individual differences in anterior hippocampal maturity were associated with enhanced memory in older participants. Together, these findings suggest that familiar and novel task contexts offer complementary, but fundamentally different, insights into functional neural development. Novel experimental tasks reveal systematic age-related changes in functional maturity and their behavioral consequences, whereas familiar movie-viewing tasks highlight how idiosyncratic individual differences in neural maturity relate to behavior once broadly adult-like patterns are already present.

---
**add figure here**

Feel free to reach out to corresponding author Owen Friend at ofriend@utexas.edu with any questions!
