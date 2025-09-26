+++
title = "Glossary"
weight = 6
+++

### set
{{% expand "set" %}}  A *set* is a collection of elements or members. Sets are defined by the elements they do or do not contain. The elements are listed with commas between them and "$\\{$" denotes the start of a set and "$\\}$" the end of a set. Note that the elements of a set are unique.{{% /expand %}}

### event
{{% expand "event" %}} An *event* is a set that is none, some, or all of the possible outcomes. {{% /expand %}}

### cardinality
{{% expand "cardinality" %}} The *cardinality* or *size* of a set is the number of elements it contains. If $A = \\{H, T\\}$, then the cardinality $A$ is $|A|=2$. {{% /expand %}}

### probability
{{% expand "probability" %}} The *probability* of an event $A$ relative to an outcome space $\Omega$ is the ratio of their sizes or $\frac{|A|}{|\Omega|}$ {{% /expand %}}

### random variable
{{% expand "random variable" %}} A *random variable* is a function that maps from the set of possible outcomes to some set or space. The output or range of the function could be the set of outcomes again, a whole number based on the outcome (e.g., counting the number of Tonkatsu), or something more complex (e.g., the world's friendship matrix, an 8-billion by 8-billion, binary matrix where $N$ where $N_{1,100}=1$ if person 1 is friends with person 100). Technically the output must be *measurable*. You shouldn't worry about that distinction unless your random variable's output gets really, really big (like continuous). We'll talk more about probabilities over continuous random variables later. {{% /expand %}}

### conditional probability
{{% expand "conditional probability" %}}The *conditional probability* is the probability of an event conditioned on knowledge of another event. Conditioning on an event means that the possible outcomes in that event form the set of possibilities or outcome space. We then calculate probabilities as normal within that *restricted* outcome space. Formally, this is written as $P(A \mid B) = \frac{|A|}{|B|}$, where everything to the left of the $\mid$ is what we're interested in knowing the probability of and everything to the right of the $\mid$ is what we know to be true. {{% /expand %}}

### dependence
{{% expand "dependence" %}}  When knowing the outcome of one random variable or event influences the probability of another, those variables or events are called *dependent*. This is denoted as $A \not\perp B$. When they do not influence each other, they are called *independent*. This is denoted as $A \perp B$.  {{% /expand %}}

### marginal probability
{{% expand "marginal probability" %}} A *marginal probability* is the probability of a random variable that has been calculated by summing over the possible values of one or more other random variables.  {{% /expand %}}

### joint probability
{{% expand "joint probability" %}} The *joint probability* is the probability of all considered events. This corresponds to the intersection of the events.  {{% /expand %}}

### Bayes theorem
{{% expand "Bayes Theorem" %}} *Bayes Theorem* is a rule for reversing the order that variables are conditioned -- how to go from $P(A \mid B)$ to $P(B \mid A)$ {{% /expand %}}

### generative process
{{% expand "generative process" %}} A *generative process* defines the probabilities for possible outcomes according to an algorithm with random choices. {{% /expand %}}

### probabilistic computing
{{% expand "probabilistic computing" %}} *Probabilistic computing* is a programming language for specifying probabilistic models and built to calculate different probabilities according to this model in an efficient manner {{% /expand %}}